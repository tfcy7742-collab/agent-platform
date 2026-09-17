"""knowledge_search 工具：把 RAG 问答引擎包装成 Agent 可调用的工具。

设计要点
--------
* **工具返回的是"结论 + 来源"，不是原始片段列表**：RAG 引擎已经做了
  三层拒答与引用校验，工具层直接复用，避免 Agent 绕过拒答逻辑自己拼接片段；
* **拒答是正常结果而不是错误**：``ok=True`` 但 ``data.refused=True``，
  并在 display 里明确写出"资料中没有相关信息"。这样 Planner 知道
  "检索工具用过了、但没有资料"，而不是误以为工具坏了去重试；
* **来源结构化返回**：``data.sources`` 带文件名/页码/chunk_id/分数，
  让 Agent 在最终回答里能给出可溯源的引用。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.tools.base import BaseTool, ToolResult
from rag.answer import RagEngine, get_rag_engine

# 工具返回给模型的片段上限（避免上下文爆炸）
MAX_SOURCES_IN_DATA = 5


class KnowledgeSearchTool(BaseTool):
    """企业知识库检索工具。"""

    name = "knowledge_search"
    description = (
        "在企业知识库（已上传的 PDF/Word/TXT/Markdown 文档）中检索并回答问题。"
        "适用于：查询公司制度（年假、报销、考勤、保密）、产品需求、运维手册等"
        "**已有文档里的信息**。如果知识库中没有相关内容，工具会明确返回"
        "「根据现有资料，我无法回答这个问题」，此时不要编造答案，如实告知用户。"
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索用的问题，应该是完整、可独立理解的问句（不要把多轮追问的代词原样传进来）",
            },
            "top_k": {
                "type": "integer",
                "description": "参与回答的片段数量，默认 3",
                "default": 3,
                "minimum": 1,
                "maximum": 10,
            },
            "threshold": {
                "type": "number",
                "description": "拒答阈值（0~1），低于该分数判定为资料中无相关内容；不传则用系统默认值",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["query"],
    }
    timeout_s = 60.0      # 含一次大模型调用
    est_cost = "medium"   # 会消耗一次 LLM 调用
    est_latency = "medium"
    retryable = True

    def __init__(self, engine: Optional[RagEngine] = None) -> None:
        super().__init__()
        self._engine = engine

    @property
    def engine(self) -> RagEngine:
        """延迟获取引擎单例（避免导入期就加载模型）。"""
        return self._engine or get_rag_engine()

    # ------------------------------------------------------------------
    def _run(self, query: str, top_k: int = 3, threshold: Optional[float] = None, **_: Any) -> ToolResult:
        """执行检索问答。"""
        result = self.engine.answer(query, top_k=int(top_k), threshold=threshold)

        sources = [
            {
                "file_name": item.file_name,
                "page": item.page,
                "chunk_id": item.chunk_id,
                "score": item.score,
                "text": item.text,
            }
            for item in result.sources[:MAX_SOURCES_IN_DATA]
        ]

        data: Dict[str, Any] = {
            "query": query,
            "answer": result.answer,
            "refused": result.refused,
            "refuse_reason": result.refuse_reason,
            "sources": sources,
            "top_score": round(result.top_score, 4),
            "retrieved": result.retrieved,
        }

        # 给前端/日志看的简短说明
        if result.refused:
            display = f"知识库检索：{result.answer}（原因：{result.refuse_reason}）"
        else:
            cite = "、".join(
                f"{item.file_name}{f' 第 {item.page} 页' if item.page else ''}"
                for item in result.sources[:3]
            )
            display = f"知识库检索到 {result.retrieved} 条相关片段（来源：{cite}）\n答案：{result.answer}"

        return ToolResult(
            ok=True,
            data=data,
            display=display,
            degraded=result.llm_degraded,
            meta={
                "refused": result.refused,
                "refuse_reason": result.refuse_reason,
                "top_score": round(result.top_score, 4),
                "sources": len(result.sources),
                "retrieval": result.retrieval_stats,
                "tokens": result.total_tokens,
                "cost_est": result.cost_est,
                "mode": result.meta.get("mode"),
            },
        )


__all__ = ["KnowledgeSearchTool"]
