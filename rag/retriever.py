"""混合检索模块（模块 3 的检索部分）。

检索策略：**向量召回 + BM25 召回 → RRF 融合 →（可选）CrossEncoder 重排**。

为什么必须做混合检索
--------------------
纯向量检索对"语义相近"很强（"手机断网还能记笔记吗" 能命中"离线编辑"），
但对**精确关键词**偏弱：型号、编号、专有名词、政策条款号这类字面串，
向量模型经常给不出高分，而 BM25 一命中即准。两者互补是工业界的默认做法。

为什么用 RRF 融合而不是加权求和
------------------------------
向量分数（0~1 余弦）与 BM25 分数（无上界、随语料变化）**量纲不同**，
加权求和需要调权重，且换一批文档就得重调。
RRF（Reciprocal Rank Fusion）只看**名次**：

    score = Σ 1 / (k + rank_i)      # k 默认 60

它无需归一化、对分数尺度免疫，是"少调参、跨数据集稳定"的选择。

分数语义（关键设计）
--------------------
融合分只用来**排序**，对外暴露的 ``score`` 仍取**向量余弦相似度**。
原因：拒答阈值是在真实 bge 模型上按余弦相似度标定的
（文档内 0.374~0.770 / 文档外 0.205~0.465，见 scripts/verify_embedding_model.py），
如果换成 RRF 分数，阈值必须重新标定，而且失去了"跨查询可比"的含义。
这个约定在 ``_fuse`` 里有显式注释，改动前请先读它。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.settings import Settings, get_settings
from rag.embeddings import EmbeddingService, get_embedding_service
from rag.models import Chunk, RetrievedChunk
from rag.store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# BM25 索引的进程内缓存：(指纹 -> (tokenized_corpus, chunks))
# 指纹由"块数 + 最大 chunk_id 哈希"构成，语料变化时自动失效。
_bm25_cache: Dict[str, Tuple[List[List[str]], List[Chunk]]] = {}
_bm25_lock = threading.Lock()

# 领域词典：把与"检索精度"直接相关的业务词钉成固定切分。
# 关闭 HMM 后 jieba 完全依赖词典，像"年假""事假""病假"这类词会被切成单字，
# 单字匹配会引入噪声（"假"能匹配到所有假期条款）。加入自定义词后，
# 这些词成为独立词项，BM25 的区分度明显提升。
DOMAIN_WORDS = [
    "年假", "事假", "病假", "婚假", "产假", "陪产假", "丧假", "调休", "加班",
    "考勤", "打卡", "补卡", "报销", "差旅", "餐补", "发票", "预算", "审批",
    "保密", "知识产权", "离职", "交接", "试用期", "转正", "绩效", "调薪",
    "离线编辑", "双向链接", "协作空间", "模板市场", "检索增强",
    "备份", "恢复", "灰度发布", "压测", "缓存", "索引", "中间件", "队列",
]
_domain_dict_loaded = False
_domain_dict_lock = threading.Lock()

# 中文停用词 / 功能词。
#
# 为什么必须过滤：BM25 不知道"哪些词没有信息量"。中文里"如何""怎么""可以"
# "使用"这类功能词在任意文档里都高频出现，若不过滤，一个完全无关的问题
# （如"如何用 Python 写快速排序"）会因为"如何""用"命中运维文档里的
# "如何做性能调优""使用压测工具"而拿到虚高的 BM25 分——实测该问题的
# Top-1 分数达到了 0.469，与真正的问题（"年假有几天" 0.471）几乎相同，
# 直接导致拒答阈值失效。
#
# 这份词表覆盖：疑问词、连接词、泛化动词、量词与常见虚词。
STOPWORDS = {
    # 疑问 / 指代
    "如何", "怎么", "怎样", "什么", "哪些", "哪个", "多少", "为什么", "是否", "能否",
    "可以", "需要", "应该", "可能", "有没有", "这个", "那个", "这些", "那些", "我们",
    "你们", "他们", "它", "我", "你", "他", "她",
    # 泛化动词
    "使用", "进行", "实现", "提供", "支持", "包含", "属于", "作为", "成为", "通过",
    "处理", "完成", "获取", "设置", "查看", "了解", "知道", "告诉", "介绍", "说明",
    "包括", "以及", "并且", "或者", "但是", "因为", "所以", "如果", "那么", "就是",
    # 单字虚词（jieba 关闭 HMM 后常把虚词切成单字）
    "的", "了", "和", "与", "及", "或", "在", "是", "有", "无", "不", "也", "都",
    "就", "而", "被", "把", "从", "到", "对", "为", "以", "并", "等", "将", "让",
    "会", "能", "要", "该", "这", "那", "个", "些", "么", "呢", "吧", "啊", "呀",
    "用", "做", "好", "多", "少", "大", "小", "上", "下", "里", "中", "前", "后",
    # 常见量词 / 时间泛词
    "一个", "一种", "一次", "一些", "时候", "情况", "问题", "方式", "方法", "内容",
    "今天", "明天", "昨天", "现在", "最近", "一般", "通常", "主要", "重要", "相关",
}


def _ensure_domain_dict() -> None:
    """把领域词加入 jieba 词典（幂等，仅首次生效）。"""
    global _domain_dict_loaded
    if _domain_dict_loaded:
        return
    with _domain_dict_lock:
        if _domain_dict_loaded:
            return
        try:
            import jieba

            for word in DOMAIN_WORDS:
                jieba.add_word(word, freq=20000)
        except ImportError:  # pragma: no cover
            pass
        _domain_dict_loaded = True


# ---------------------------------------------------------------------------
# 中文分词（BM25 用）
# ---------------------------------------------------------------------------
def tokenize_chinese(text: str) -> List[str]:
    """中文分词，供 BM25 使用。

    使用 jieba 的**精确模式且关闭 HMM**（``HMM=False``）：HMM 会对词典外的
    连续汉字做新词发现，结果依赖首次调用时的字典加载顺序，同一句话可能出现
    不同的切分（例如"年假"有时切成一个词、有时切成"年"+"假"）。
    BM25 的词项统计需要**可复现**，因此这里选择确定性优先。

    未安装 jieba 时退化为"单字 + 字母数字串"，保证功能不中断（只是召回略降）。
    """
    if not text:
        return []
    try:
        import jieba

        _ensure_domain_dict()
        tokens = [token.strip() for token in jieba.lcut(text, HMM=False)]
    except ImportError:  # pragma: no cover - requirements 里已包含 jieba
        tokens = []
        buffer: List[str] = []
        for char in text:
            if char.isalnum():
                buffer.append(char)
            else:
                if buffer:
                    tokens.append("".join(buffer))
                    buffer.clear()
                tokens.append(char)
        if buffer:
            tokens.append("".join(buffer))

    # 过滤标点、空白与停用词，统一小写
    return [
        token.lower()
        for token in tokens
        if token.strip() and any(c.isalnum() for c in token) and token.lower() not in STOPWORDS
    ]


# ---------------------------------------------------------------------------
# 检索器
# ---------------------------------------------------------------------------
class HybridRetriever:
    """混合检索器：向量 + BM25 + RRF（+ 可选重排）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self._reranker: Any = None
        self._reranker_error: Optional[str] = None
        # 检索耗时统计（供 trace 与指标展示）
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BM25 分支
    # ------------------------------------------------------------------
    def _corpus_fingerprint(self) -> str:
        """语料指纹：用于判断 BM25 索引缓存是否失效。

        由「逻辑库标识 + 块数 + 最近 chunk_id」组成。

        为什么不能只用块数：**不同知识库可能块数相同但内容不同**
        （例如两个都恰好 10 块的库），只按块数判断会让 BM25 复用上一个库的索引，
        召回出根本不属于当前库的内容——这是实际踩到的 bug。

        为什么加上"最近 chunk_id"：进程内的语料版本号是全局的，切换库（测试、
        多知识库场景）时并不会重置，单靠版本号无法区分不同库。
        取集合里最后一个 chunk_id 作为内容指纹，可以低成本地识别"库换了/内容变了"。
        """
        from rag.store import get_corpus_version

        stats = self.vector_store.stats()
        return (
            f"v{get_corpus_version()}"
            f"|{self.vector_store.persist_directory}"
            f"|{stats.get('collection', '')}"
            f"|{stats.get('chunks', 0)}"
            f"|{self._latest_chunk_id()}"
        )

    def _latest_chunk_id(self) -> str:
        """取集合中最后一个 chunk_id（作为内容指纹的一部分，代价极低）。"""
        try:
            result = self.vector_store.collection.get(limit=1, include=[])
            ids = result.get("ids") or []
            return str(ids[0]) if ids else ""
        except Exception:  # noqa: BLE001 - 读不到就退化为空串
            return ""

    def _load_corpus(self) -> List[Chunk]:
        """从向量库取出全部块（用于构建 BM25 索引）。"""
        collection = self.vector_store.collection
        result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        chunks: List[Chunk] = []
        for index, chunk_id in enumerate(ids):
            text = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            if not text:
                continue
            chunks.append(
                Chunk.from_metadata(chunk_id=str(chunk_id), text=str(text), metadata=dict(metadata or {}))
            )
        return chunks

    def _get_bm25_index(self) -> Tuple[Optional[Any], List[Chunk]]:
        """获取（或构建）BM25 索引。

        索引在进程内缓存，语料指纹不变时直接复用——否则每次检索都重算分词，
        在几百个块时就会带来几百毫秒的无谓开销。
        """
        if not self.settings.enable_bm25:
            return None, []

        fingerprint = self._corpus_fingerprint()
        with _bm25_lock:
            cached = _bm25_cache.get(fingerprint)
            if cached is not None:
                return cached[0], cached[1]

            chunks = self._load_corpus()
            if not chunks:
                return None, []

            try:
                from rank_bm25 import BM25Okapi

                tokenized = [tokenize_chinese(chunk.text) for chunk in chunks]
                # 过滤空 token 序列（否则 BM25Okapi 会因平均长度计算异常而报警告）
                tokenized = [tokens or ["空"] for tokens in tokenized]
                index = BM25Okapi(tokenized)
            except Exception as exc:  # noqa: BLE001 - BM25 失败时静默降级为纯向量
                logger.warning("构建 BM25 索引失败，已降级为纯向量检索：%s", exc)
                return None, []

            _bm25_cache[fingerprint] = (index, chunks)
            logger.info("BM25 索引已构建：%s 个块", len(chunks))
            return index, chunks

    def invalidate_cache(self) -> None:
        """清空 BM25 缓存（文档增删后调用，避免召回已删除的内容）。"""
        with _bm25_lock:
            _bm25_cache.clear()

    def _bm25_search(self, query: str, top_n: int) -> List[Tuple[Chunk, float]]:
        """BM25 检索，返回 (块, 原始分) 列表（按分数降序）。"""
        index, chunks = self._get_bm25_index()
        if index is None or not chunks:
            return []

        tokens = tokenize_chinese(query)
        if not tokens:
            return []
        try:
            scores = index.get_scores(tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 打分失败：%s", exc)
            return []

        ranked = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        results: List[Tuple[Chunk, float]] = []
        for position in ranked[:top_n]:
            score = float(scores[position])
            if score <= 0:
                continue  # 零分表示一个词都没命中，纳入融合只会引入噪声
            results.append((chunks[position], score))
        return results

    # ------------------------------------------------------------------
    # 重排分支
    # ------------------------------------------------------------------
    def _get_reranker(self) -> Any:
        """懒加载 CrossEncoder 重排模型（默认关闭，需下载约 1GB）。"""
        if not self.settings.enable_rerank:
            return None
        if self._reranker is not None or self._reranker_error is not None:
            return self._reranker
        try:
            from config.settings import ensure_model_endpoint
            from sentence_transformers import CrossEncoder

            ensure_model_endpoint()
            logger.info("正在加载重排模型：%s", self.settings.rerank_model)
            self._reranker = CrossEncoder(
                self.settings.rerank_model,
                max_length=512,
                cache_folder=str(self.settings.model_cache_path),
            )
        except Exception as exc:  # noqa: BLE001 - 重排是可选增强，失败要能降级
            self._reranker_error = f"{type(exc).__name__}: {exc}"
            logger.warning("重排模型加载失败，跳过重排：%s", exc)
        return self._reranker

    def _rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """用 CrossEncoder 对候选重新打分（失败则原样返回）。"""
        model = self._get_reranker()
        if model is None or not candidates:
            return candidates

        limited = candidates[: self.settings.rerank_top_n]
        try:
            pairs = [(query, item.chunk.text) for item in limited]
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("重排失败，使用融合结果：%s", exc)
            return candidates

        # CrossEncoder 输出是 logits，用 sigmoid 压到 0~1，保持与阈值语义一致
        def to_probability(value: float) -> float:
            import math

            try:
                return round(1.0 / (1.0 + math.exp(-float(value))), 6)
            except OverflowError:  # pragma: no cover - 极端 logits
                return 0.0 if value < 0 else 1.0

        reranked: List[RetrievedChunk] = []
        for item, raw_score in zip(limited, scores):
            probability = to_probability(raw_score)
            reranked.append(
                item.model_copy(
                    update={
                        "score": probability,
                        "vec_score": item.vec_score,
                        "retriever": "rerank",
                    }
                )
            )
        reranked.sort(key=lambda entry: -entry.score)
        # 未被重排的候选（超出 rerank_top_n）按原顺序追加在后面
        reranked.extend(candidates[len(limited) :])
        return reranked

    # ------------------------------------------------------------------
    # 融合
    # ------------------------------------------------------------------
    def _fuse(
        self,
        vector_hits: List[RetrievedChunk],
        bm25_hits: List[Tuple[Chunk, float]],
    ) -> List[RetrievedChunk]:
        """RRF 融合两路召回结果。

        重要约定（勿轻易修改）：返回结果的 ``score`` 字段是**向量余弦相似度**，
        不是 RRF 分数。原因见模块文档字符串——拒答阈值是按余弦相似度标定的，
        保持 score 语义不变，阈值才能跨查询复用、也不需要重新标定。
        RRF 只决定**排序**（``rank_fused``）。
        """
        k = self.settings.rrf_k
        merged: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            entry = merged.setdefault(hit.chunk.chunk_id, {"chunk": hit.chunk, "rrf": 0.0})
            entry["rrf"] += 1.0 / (k + rank)
            entry["vec_score"] = hit.score
            entry["vector_rank"] = rank

        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            entry = merged.setdefault(chunk.chunk_id, {"chunk": chunk, "rrf": 0.0})
            entry["rrf"] += 1.0 / (k + rank)
            entry["bm25_score"] = score
            entry["bm25_rank"] = rank

        ordered = sorted(merged.values(), key=lambda item: -item["rrf"])
        results: List[RetrievedChunk] = []
        for position, entry in enumerate(ordered, start=1):
            vec_score = entry.get("vec_score")
            bm25_score = entry.get("bm25_score")
            # score 的口径：优先用向量相似度；只有 BM25 命中（向量未召回）时，
            # 退化为"由 BM25 名次换算的保守估计值"，并如实标注 retriever=bm25。
            if vec_score is not None:
                score = float(vec_score)
                retriever = "hybrid" if bm25_score is not None else "vector"
            else:
                score = round(1.0 / (k + entry.get("bm25_rank", k)), 6)
                retriever = "bm25"

            results.append(
                RetrievedChunk(
                    chunk=entry["chunk"],
                    score=round(score, 6),
                    vec_score=vec_score,
                    bm25_score=bm25_score,
                    rank_fused=position,
                    rank_final=position,
                    retriever=retriever,
                )
            )
        return results

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidates: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        """执行一次混合检索。

        Args:
            query: 用户问题（内部不加 bge 查询前缀，前缀由 Embedding 服务负责）。
            top_k: 最终返回条数，默认 ``RETRIEVE_TOP_K``（3）。
            candidates: 每路召回候选数，默认 ``RETRIEVE_CANDIDATES``（20）。
            min_score: 低于该分数的结果直接丢弃（默认 0 = 不过滤，由上层做拒答判断）。

        Returns:
            按相关度排序的 ``RetrievedChunk`` 列表。
        """
        query = (query or "").strip()
        top_k = top_k or self.settings.retrieve_top_k
        candidates = candidates or self.settings.retrieve_candidates

        stats: Dict[str, Any] = {
            "query": query,
            "bm25_enabled": bool(self.settings.enable_bm25),
            "rerank_enabled": bool(self.settings.enable_rerank),
            "vector_hits": 0,
            "bm25_hits": 0,
            "fused": 0,
            "returned": 0,
            "latency_ms": 0,
        }
        started = time.perf_counter()

        if not query:
            self.last_stats = stats
            return []

        if self.vector_store.count() == 0:
            self.last_stats = stats
            return []

        # ---- 两路召回 ----
        vector_hits = self.vector_store.search_with_scores(query, k=candidates)
        stats["vector_hits"] = len(vector_hits)
        bm25_hits = self._bm25_search(query, candidates)
        stats["bm25_hits"] = len(bm25_hits)

        # ---- 融合 ----
        fused = self._fuse(vector_hits, bm25_hits)
        stats["fused"] = len(fused)

        # ---- 可选重排 ----
        if self.settings.enable_rerank and fused:
            fused = self._rerank(query, fused)

        # ---- 过滤与截断 ----
        if min_score > 0:
            fused = [item for item in fused if item.score >= min_score]
        results = fused[:top_k]
        # 重新编号最终名次（引用编号与展示顺序以它为准）
        results = [
            item.model_copy(update={"rank_final": index}) for index, item in enumerate(results, start=1)
        ]

        stats["returned"] = len(results)
        stats["latency_ms"] = int((time.perf_counter() - started) * 1000)
        stats["top_score"] = round(results[0].score, 4) if results else 0.0
        self.last_stats = stats
        return results

    def describe(self) -> Dict[str, Any]:
        """检索能力描述（供 /health 展示）。"""
        return {
            "vector": {"backend": "chromadb", "chunks": self.vector_store.count()},
            "bm25": {
                "enabled": bool(self.settings.enable_bm25),
                "tokenizer": "jieba",
                "indexed": self._get_bm25_index()[0] is not None,
            },
            "rerank": {
                "enabled": bool(self.settings.enable_rerank),
                "model": self.settings.rerank_model if self.settings.enable_rerank else None,
                "available": self._reranker is not None,
                "error": self._reranker_error,
            },
            "fusion": {"method": "rrf", "k": self.settings.rrf_k},
            "top_k": self.settings.retrieve_top_k,
            "candidates": self.settings.retrieve_candidates,
        }


_retriever_singleton: Optional[HybridRetriever] = None


def get_retriever(reload: bool = False) -> HybridRetriever:
    """获取全局检索器单例。"""
    global _retriever_singleton
    if _retriever_singleton is None or reload:
        _retriever_singleton = HybridRetriever()
    return _retriever_singleton


def reset_retriever_singleton() -> None:
    """丢弃单例并清空 BM25 缓存（测试与文档变更后使用）。"""
    global _retriever_singleton
    _retriever_singleton = None
    with _bm25_lock:
        _bm25_cache.clear()


__all__ = [
    "HybridRetriever",
    "get_retriever",
    "reset_retriever_singleton",
    "tokenize_chinese",
]
