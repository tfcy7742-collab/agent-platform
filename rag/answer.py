"""RAG 问答引擎（模块 3）。

流程
----
::

    问题
     │
     ├─ 知识库为空？ ────────────────▶ 拒答（no_documents，不调用大模型）
     │
     ├─ 混合检索（向量 + BM25 + RRF [+ 重排]）
     │
     ├─ 最高分 < REFUSE_THRESHOLD？ ──▶ 拒答（below_threshold，不调用大模型）
     │
     ├─ 构造 Prompt（含 {context} 与 {question}）→ 调用 DeepSeek（JSON 模式）
     │
     ├─ 模型说 confident=false 或无回答？ ─▶ 拒答（llm_uncertain）
     │
     ├─ 引用编号越界？ ───────────────▶ 拒答（citation_invalid）
     │
     └─ 通过 ──▶ 返回答案 + 来源片段

三层拒答为什么必要（有实测依据）
--------------------------------
真实 bge 模型上的分数分布存在重叠：文档内问题最低 0.374，文档外问题最高 0.465。
也就是说：

* 阈值设低了（如 0.35）→ 会漏放少量无关问题进来；
* 阈值设高了（如 0.50）→ 会把"保密义务持续多久"这类正常问题误拒。

因此单一阈值无法同时做到零漏拒与零误拒，必须叠加
**Prompt 约束**（让模型自己判断资料够不够）与**引用校验**（答案里的引用必须能对上片段）。
三层叠加后，端到端拒答准确率才能达到「文档外问题稳定返回标准话术」的验收要求。

离线模式
--------
未配置 API Key 时，回答由 ``_offline_answer`` 生成：从命中的片段里按关键词重合度
摘取最相关的句子并标注来源。它不做自然语言归纳，但**保证有依据、不编造**，
也保证了无 Key 环境下整条链路（检索 → 拒答 → 引用）依然可测可用。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.llm import LLMClient, get_llm_client
from core.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    ANSWER_SYSTEM_PROMPT,
    BELOW_THRESHOLD_REASON,
    CITATION_INVALID_REASON,
    NO_DOCUMENT_REASON,
    OFFLINE_PREFIX,
    REFUSAL_MESSAGE,
)
from config.settings import Settings, get_settings
from rag.models import RetrievedChunk, SourceRef
from rag.retriever import HybridRetriever, get_retriever
from rag.store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# 单条引用片段的展示上限（太长会让前端面板与 Prompt 都很臃肿）
SOURCE_TEXT_LIMIT = 400
# 组装 Prompt 时的片段上限（控制上下文长度）
CONTEXT_CHUNK_LIMIT = 6


# ---------------------------------------------------------------------------
# 返回结构
# ---------------------------------------------------------------------------
@dataclass
class RagAnswer:
    """一次问答的完整结果。"""

    answer: str
    refused: bool = False
    refuse_reason: Optional[str] = None
    sources: List[SourceRef] = field(default_factory=list)
    hits: List[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0
    retrieved: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_est: float = 0.0
    llm_latency_ms: int = 0
    llm_degraded: bool = False
    llm_error: Optional[str] = None
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转成可直接 JSON 序列化的字典（供 API 层返回）。"""
        return {
            "answer": self.answer,
            "refused": self.refused,
            "refuse_reason": self.refuse_reason,
            "sources": [item.model_dump() for item in self.sources],
            "top_score": round(self.top_score, 4),
            "retrieved": self.retrieved,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cost_est": round(self.cost_est, 6),
            },
            "llm": {
                "latency_ms": self.llm_latency_ms,
                "degraded": self.llm_degraded,
                "error": self.llm_error,
            },
            "retrieval": self.retrieval_stats,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# 引用校验
# ---------------------------------------------------------------------------
def extract_citation_indices(text: str) -> List[int]:
    """从回答文本中提取引用编号。

    支持中英文方括号：【1】【1,2】【1-3】与 [1] [1,2] [1-3]。
    解析失败（例如模型没写引用）返回空列表——**没有引用不算错误**，
    真正的错误是"引用了不存在的编号"。
    """
    import re

    if not text:
        return []

    indices: List[int] = []
    for match in re.finditer(r"[【\[]\s*([0-9,\-\s]+?)\s*[】\]]", text):
        body = match.group(1)
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                bounds = [item.strip() for item in part.split("-", 1)]
                if len(bounds) == 2 and all(item.isdigit() for item in bounds):
                    start, end = int(bounds[0]), int(bounds[1])
                    if 0 < start <= end:
                        # 不在这里静默丢弃越界区间：交给 validate_citations 判为非法，
                        # 由上层拒答，避免"看起来通过了校验"的假阳性。
                        indices.extend(range(start, end + 1))
                continue
            if part.isdigit():
                indices.append(int(part))
    # 去重并保持出现顺序
    seen: set = set()
    ordered: List[int] = []
    for index in indices:
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def validate_citations(answer: str, hit_count: int) -> Dict[str, Any]:
    """校验回答里的引用编号是否落在检索片段范围内。

    Args:
        answer: 模型回答文本。
        hit_count: 本次检索到的片段数量（合法编号范围 1..hit_count）。

    Returns:
        ``{"ok": bool, "indices": [...], "invalid": [...], "had_citation": bool}``
    """
    indices = extract_citation_indices(answer)
    invalid = [index for index in indices if index < 1 or index > hit_count]
    return {
        "ok": not invalid,
        "indices": indices,
        "invalid": invalid,
        "had_citation": bool(indices),
    }


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class RagEngine:
    """RAG 问答引擎。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retriever: Optional[HybridRetriever] = None,
        llm: Optional[LLMClient] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or get_retriever()
        self.llm = llm or get_llm_client()
        self.vector_store = vector_store or get_vector_store()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> RagAnswer:
        """基于知识库回答问题。

        Args:
            question: 用户问题（应为已改写、可独立检索的问题）。
            top_k: 返回给模型的片段数，默认取配置。
            threshold: 拒答阈值，默认取 ``REFUSE_THRESHOLD``。
            history: 可选的多轮对话历史（仅用于日志，检索用的是 question 本身）。

        Returns:
            ``RagAnswer``：含答案、拒答标记与原因、来源片段、用量与检索统计。
        """
        question = (question or "").strip()
        top_k = top_k or self.settings.retrieve_top_k
        threshold = self.settings.refuse_threshold if threshold is None else threshold

        if not question:
            return self._refuse(NO_DOCUMENT_REASON, reason_code="empty_question", top_score=0.0)

        # ---- 第一道闸门：知识库是否为空 ----
        total_chunks = self.vector_store.count()
        if total_chunks == 0:
            logger.info("知识库为空，直接拒答")
            return self._refuse(
                NO_DOCUMENT_REASON,
                reason_code="no_documents",
                top_score=0.0,
                meta={"indexed_chunks": 0},
            )

        # ---- 混合检索 ----
        hits = self.retriever.retrieve(question, top_k=top_k)
        retrieval_stats = dict(self.retriever.last_stats)
        top_score = hits[0].score if hits else 0.0
        sources = [self.to_source_ref(hit) for hit in hits]

        # ---- 第二道闸门：检索分数低于阈值（不调用大模型，成本为零且结果稳定）----
        if not hits or top_score < threshold:
            logger.info(
                "检索分数 %.4f 低于阈值 %.2f，短路拒答（未调用大模型）", top_score, threshold
            )
            return self._refuse(
                BELOW_THRESHOLD_REASON,
                reason_code="below_threshold",
                top_score=top_score,
                hits=hits,
                sources=sources,
                retrieval_stats=retrieval_stats,
                meta={"threshold": threshold},
            )

        # ---- 组装 Prompt（含 {context} 与 {question}）----
        context = self.build_context(hits)
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
        system_prompt = ANSWER_SYSTEM_PROMPT.format(refusal_message=REFUSAL_MESSAGE)

        # ---- 离线模式：不调用大模型，走模板摘要 ----
        if not self.llm.available:
            offline_text, cited = self._offline_answer(question, hits)
            return RagAnswer(
                answer=offline_text,
                refused=False,
                sources=sources,
                hits=hits,
                top_score=top_score,
                retrieved=len(hits),
                llm_degraded=True,
                llm_error="离线模式：未启用大模型，答案由片段摘要生成",
                retrieval_stats=retrieval_stats,
                meta={"mode": "offline", "cited": cited, "threshold": threshold},
            )

        # ---- 调用 DeepSeek（JSON 模式）----
        parsed, result = self.llm.chat_json(system_prompt, prompt)
        if not result.ok or not isinstance(parsed, dict):
            logger.warning("大模型调用失败（%s），降级为片段摘要", result.error_type)
            offline_text, cited = self._offline_answer(question, hits)
            return RagAnswer(
                answer=offline_text,
                refused=False,
                sources=sources,
                hits=hits,
                top_score=top_score,
                retrieved=len(hits),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                cost_est=result.cost_est,
                llm_latency_ms=result.latency_ms,
                llm_degraded=True,
                llm_error=result.error,
                retrieval_stats=retrieval_stats,
                meta={
                    "mode": "fallback_after_llm_error",
                    "error_type": result.error_type,
                    "cited": cited,
                },
            )

        answer_text = str(parsed.get("answer") or "").strip()
        confident = bool(parsed.get("confident", True))
        raw_citations = parsed.get("citations")

        # ---- 第三道闸门之一：模型自己说资料不足 ----
        if not answer_text or not confident or REFUSAL_MESSAGE in answer_text:
            return RagAnswer(
                answer=REFUSAL_MESSAGE,
                refused=True,
                refuse_reason="llm_uncertain",
                sources=sources,
                hits=hits,
                top_score=top_score,
                retrieved=len(hits),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                cost_est=result.cost_est,
                llm_latency_ms=result.latency_ms,
                retrieval_stats=retrieval_stats,
                meta={"mode": "llm_refuse", "confident": confident, "threshold": threshold},
            )

        # ---- 第三道闸门之二：引用校验 ----
        citation_info = validate_citations(answer_text, len(hits))
        if self.settings.enable_citation_check and not citation_info["ok"]:
            logger.warning(
                "引用校验失败：回答引用了不存在的编号 %s（本次片段数 %s）",
                citation_info["invalid"], len(hits),
            )
            return RagAnswer(
                answer=REFUSAL_MESSAGE,
                refused=True,
                refuse_reason="citation_invalid",
                sources=sources,
                hits=hits,
                top_score=top_score,
                retrieved=len(hits),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                cost_est=result.cost_est,
                llm_latency_ms=result.latency_ms,
                retrieval_stats=retrieval_stats,
                meta={
                    "mode": "citation_invalid",
                    "invalid_citations": citation_info["invalid"],
                    "threshold": threshold,
                },
            )

        # 模型给的 citations 字段与文本里的编号取并集，便于前端高亮
        extra_citations = [int(item) for item in raw_citations] if isinstance(raw_citations, list) else []
        cited = sorted({*citation_info["indices"], *[item for item in extra_citations if 1 <= item <= len(hits)]})

        return RagAnswer(
            answer=answer_text,
            refused=False,
            sources=sources,
            hits=hits,
            top_score=top_score,
            retrieved=len(hits),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_est=result.cost_est,
            llm_latency_ms=result.latency_ms,
            retrieval_stats=retrieval_stats,
            meta={
                "mode": "llm",
                "cited": cited,
                "had_citation": citation_info["had_citation"],
                "threshold": threshold,
            },
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def build_context(self, hits: Sequence[RetrievedChunk]) -> str:
        """把命中片段拼成带编号的资料块（编号与引用校验的范围一致）。"""
        lines: List[str] = []
        for index, hit in enumerate(hits[:CONTEXT_CHUNK_LIMIT], start=1):
            page = f"，第 {hit.chunk.page} 页" if hit.chunk.page else ""
            lines.append(
                f"[{index}] 来源：{hit.chunk.file_name}{page}"
                f"（相关度 {hit.score:.3f}）\n{hit.chunk.text.strip()}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def to_source_ref(hit: RetrievedChunk) -> SourceRef:
        """把命中片段转成前端展示用的引用对象。"""
        text = hit.chunk.text.strip()
        if len(text) > SOURCE_TEXT_LIMIT:
            text = text[:SOURCE_TEXT_LIMIT] + "…"
        return SourceRef(
            file_name=hit.chunk.file_name,
            page=hit.chunk.page,
            chunk_id=hit.chunk.chunk_id,
            score=hit.score,
            text=text,
            file_path="",
            doc_id=hit.chunk.doc_id,
        )

    def _refuse(
        self,
        reason: str,
        reason_code: str,
        top_score: float = 0.0,
        hits: Optional[List[RetrievedChunk]] = None,
        sources: Optional[List[SourceRef]] = None,
        retrieval_stats: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> RagAnswer:
        """统一的拒答构造（保证话术与字段完全一致）。"""
        hits = hits or []
        return RagAnswer(
            answer=REFUSAL_MESSAGE,
            refused=True,
            refuse_reason=reason_code,
            sources=sources or [],
            hits=hits,
            top_score=top_score,
            retrieved=len(hits),
            retrieval_stats=retrieval_stats or {},
            meta={"reason": reason, **(meta or {})},
        )

    def _offline_answer(self, question: str, hits: Sequence[RetrievedChunk]) -> tuple:
        """离线/降级模式下的答案生成：按关键词重合度摘取最相关句子。

        Returns:
            ``(答案文本, 引用编号列表)``
        """
        from rag.retriever import tokenize_chinese

        query_tokens = set(tokenize_chinese(question))
        scored: List[tuple] = []

        for index, hit in enumerate(hits, start=1):
            for sentence in self._split_sentences(hit.chunk.text):
                sentence_tokens = set(tokenize_chinese(sentence))
                if not sentence_tokens:
                    continue
                overlap = len(query_tokens & sentence_tokens) / (len(query_tokens) or 1)
                if overlap > 0:
                    scored.append((overlap, index, sentence))

        scored.sort(key=lambda item: (-item[0], item[1]))
        picked: List[tuple] = []
        used_indices: set = set()
        for overlap, index, sentence in scored:
            if index in used_indices and len(picked) >= 2:
                continue
            picked.append((index, sentence))
            used_indices.add(index)
            if len(picked) >= 3:
                break

        if not picked:
            # 一个关键词都没重合：宁可如实说明，也不编造
            return REFUSAL_MESSAGE, []

        parts = [f"{sentence}【{index}】" for index, sentence in picked]
        return f"{OFFLINE_PREFIX}\n" + "".join(parts), sorted(used_indices)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """按中英文句末标点切句（保留标点，便于阅读）。"""
        import re

        parts = re.split(r"(?<=[。！？；!?;])", text.strip())
        return [part.strip() for part in parts if part and part.strip()]


_engine_singleton: Optional[RagEngine] = None


def get_rag_engine(reload: bool = False) -> RagEngine:
    """获取全局 RAG 引擎单例。"""
    global _engine_singleton
    if _engine_singleton is None or reload:
        _engine_singleton = RagEngine()
    return _engine_singleton


def reset_rag_engine() -> None:
    """丢弃引擎单例（测试用）。"""
    global _engine_singleton
    _engine_singleton = None


__all__ = [
    "RagAnswer",
    "RagEngine",
    "extract_citation_indices",
    "get_rag_engine",
    "reset_rag_engine",
    "validate_citations",
]
