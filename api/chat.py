"""问答接口（B3 交付，非流式版本）。

接口
----
``POST /api/chat``
    请求：``{"question": "...", "session_id": "...", "top_k": 3, "threshold": 0.35, "use_history": false}``
    响应：``{answer, refused, refuse_reason, sources, usage, retrieval, trace_id, session_id}``

B3 先交付**非流式**版本，用于自动化测试与命令行联调；B4/B5 会在同一路径上
增加 SSE 流式输出（``Accept: text/event-stream`` 或 ``stream=true``），
内部复用同一个 ``RagEngine``，接口语义保持一致。

审计：每次问答都会写入 SQLite 的 ``runs`` / ``steps`` 表（含检索分数、
拒答原因、引用编号、token 用量），可通过 ``/api/traces/{run_id}`` 回看。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.prompts import REFUSAL_MESSAGE
from infra import db
from infra.trace import TraceRecorder
from rag.answer import RagEngine, get_rag_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["问答"])

# 历史消息加载条数（仅在 use_history=true 时生效）
HISTORY_LIMIT = 20


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """问答请求。"""

    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 id；不传则自动生成")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="参与回答的片段数")
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="拒答阈值，覆盖配置中的 REFUSE_THRESHOLD"
    )
    use_history: bool = Field(
        False, description="是否读取会话历史（B4 接入问题改写后建议开启）"
    )
    mode: str = Field(
        "rag",
        description=(
            "执行模式：rag=直接走 RAG 问答管道（单工具、更快、延迟与 token 可预测，默认）；"
            "agent=走 Agent 主循环，Planner 自主路由，可调用知识库/旅行规划等多个工具。"
            "需要自动判断「该查文档还是该规划行程」时用 agent。"
        ),
    )
    max_steps: Optional[int] = Field(None, ge=1, le=12, description="Agent 步数预算（仅 mode=agent）")


class SourceItem(BaseModel):
    """引用片段。"""

    file_name: str
    page: Optional[int] = None
    chunk_id: str
    score: float
    text: str
    doc_id: str = ""


class ChatResponse(BaseModel):
    """问答响应。"""

    ok: bool = True
    answer: str
    refused: bool = False
    refuse_reason: Optional[str] = None
    sources: List[SourceItem] = Field(default_factory=list)
    top_score: float = 0.0
    retrieved: int = 0
    usage: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    llm: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    trace_id: str = ""
    session_id: str = ""
    latency_ms: int = 0
    # ---- B4：Agent 模式的额外信息 ----
    mode: str = "agent"
    status: str = "success"
    steps: int = 0
    tools_used: List[str] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    rewritten_question: Optional[str] = None
    events: List[Dict[str, Any]] = Field(
        default_factory=list, description="本次执行的事件流（前端可据此回放决策过程）"
    )


# ---------------------------------------------------------------------------
# 接口实现
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse, summary="问答（Agent 或 RAG 模式）")
async def chat(request: ChatRequest) -> ChatResponse:
    """回答问题，返回答案与来源引用。

    两种模式：

    * ``mode=rag``（默认）：直接走 RAG 管道（单工具）。**为什么默认不是 agent**：
      这是接口原有的公开行为，改默认值属于破坏性变更；而且单工具模式的延迟与
      token 消耗可预测，作为回归基线更稳（B3 的验收测试就依赖它的精确指标）；
    * ``mode=agent``：走 Agent 主循环。Planner 自主决定调用哪个工具——
      问文档内容会走 ``knowledge_search``，说"帮我规划北京三日游"会走 ``trip_planner``，
      不需要任何 if-else 硬编码。

    **拒答**在两种模式下都一致：知识库为空、检索分数低于阈值、模型判定资料不足、
    引用校验失败，都会返回统一话术``根据现有资料，我无法回答这个问题``。
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")

    session_id = request.session_id or db.new_id("sess_")
    mode = (request.mode or "rag").lower()

    # 会话历史（问题改写与审计都需要）
    history: List[Dict[str, str]] = []
    if request.use_history or mode == "agent":
        history = [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in db.get_history(session_id, limit=HISTORY_LIMIT)
        ]

    recorder = TraceRecorder(session_id=session_id, question=question)
    recorder.start()

    if mode == "rag":
        response = await _run_rag_mode(request, question, session_id, history, recorder)
    else:
        response = await _run_agent_mode(request, question, session_id, history, recorder)

    # 会话消息写入（多轮对话与审计都需要）
    db.add_message(session_id, "user", question, run_id=response.trace_id)
    db.add_message(session_id, "assistant", response.answer, run_id=response.trace_id)
    return response


async def _run_rag_mode(
    request: ChatRequest,
    question: str,
    session_id: str,
    history: List[Dict[str, str]],
    recorder: TraceRecorder,
) -> ChatResponse:
    """RAG 单工具模式（B3 的原有路径）。"""
    from fastapi.concurrency import run_in_threadpool

    engine: RagEngine = get_rag_engine()
    try:
        result = await run_in_threadpool(
            engine.answer, question, request.top_k, request.threshold, history
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("RAG 问答失败：%s", exc)
        recorder.fail(str(exc), error_type=type(exc).__name__)
        recorder.finish(answer="", status="failed")
        raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc

    recorder.plan(
        thought=(
            f"直接检索知识库（top_k={request.top_k or engine.settings.retrieve_top_k}，"
            f"阈值={request.threshold if request.threshold is not None else engine.settings.refuse_threshold}）"
        ),
        action="knowledge_search",
        tool="knowledge_search",
        args={"query": question, "top_k": request.top_k},
        latency_ms=int(result.retrieval_stats.get("latency_ms", 0) or 0),
        detail={
            "vector_hits": result.retrieval_stats.get("vector_hits"),
            "bm25_hits": result.retrieval_stats.get("bm25_hits"),
            "top_score": round(result.top_score, 4),
        },
    )
    if result.refused:
        recorder.refuse(
            reason=str(result.refuse_reason),
            top_score=result.top_score,
            detail={"mode": result.meta.get("mode"), "threshold": result.meta.get("threshold")},
        )
    else:
        recorder.respond(
            answer=result.answer,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_est=result.cost_est,
            latency_ms=result.llm_latency_ms,
            degraded=result.llm_degraded,
            detail={
                "cited": result.meta.get("cited", []),
                "sources": len(result.sources),
                "mode": result.meta.get("mode"),
            },
        )
    summary = recorder.finish(answer=result.answer)
    payload = result.to_dict()
    return _build_response(
        mode="rag",
        answer=payload["answer"],
        refused=payload["refused"],
        refuse_reason=payload["refuse_reason"],
        sources=payload["sources"],
        top_score=payload["top_score"],
        retrieved=payload["retrieved"],
        usage=payload["usage"],
        retrieval=payload["retrieval"],
        llm=payload["llm"],
        meta=payload["meta"],
        trace_id=summary["run_id"],
        session_id=session_id,
        latency_ms=int(summary["total_ms"]),
    )


async def _run_agent_mode(
    request: ChatRequest,
    question: str,
    session_id: str,
    history: List[Dict[str, str]],
    recorder: TraceRecorder,
) -> ChatResponse:
    """Agent 模式：Planner 自主路由 + 工具执行 + 回答汇总。"""
    from fastapi.concurrency import run_in_threadpool

    from core.runtime.agent import get_agent_runtime

    runtime = get_agent_runtime()
    try:
        result = await run_in_threadpool(
            runtime.run,
            question,
            session_id,
            history,
            request.max_steps,
            None,           # 事件回调（SSE 流式在 B5 接入）
            recorder.run_id,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Agent 执行失败：%s", exc)
        recorder.fail(str(exc), error_type=type(exc).__name__)
        recorder.finish(answer="", status="failed")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{exc}") from exc

    # ---- 把 Agent 的决策与工具调用写入 trace（可观测性）----
    # 每个决策记一步：前端"这次回答是怎么产生的"面板就是靠这些步骤还原的。
    for index, event in enumerate(result.events):
        if event.type == "plan":
            recorder.plan(
                thought=str(event.payload.get("thought") or ""),
                action=str(event.payload.get("action") or "route"),
                tool=event.payload.get("tool"),
                args=event.payload.get("args") or {},
                latency_ms=int(event.payload.get("latency_ms") or 0),
                detail={"step": event.payload.get("step"), "source": event.payload.get("source")},
            )
    for call in result.tool_calls:
        recorder.tool(
            call.get("tool", "unknown"),
            call.get("args", {}),
            _ToolCallView(call),
            latency_ms=int(call.get("latency_ms", 0) or 0),
            detail={
                "step": call.get("step"),
                "retried": call.get("retried"),
                "attempts": call.get("attempts"),
            },
        )
    if result.refused:
        recorder.refuse(reason=str(result.refuse_reason), top_score=0.0, detail={"mode": "agent"})
    else:
        recorder.respond(
            answer=result.answer,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_est=result.cost_est,
            latency_ms=result.total_ms,
            degraded=result.degraded,
            detail={"tools": result.tools_used, "steps": result.steps},
        )
    summary = recorder.finish(answer=result.answer, status=result.status)

    return _build_response(
        mode="agent",
        answer=result.answer,
        refused=result.refused,
        refuse_reason=result.refuse_reason,
        sources=result.sources,
        top_score=0.0,
        retrieved=len(result.sources),
        usage={
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "cost_est": round(result.cost_est, 6),
        },
        retrieval={},
        llm={"degraded": result.degraded, "online": runtime.llm.available},
        meta=result.meta,
        trace_id=summary["run_id"],
        session_id=session_id,
        latency_ms=int(summary["total_ms"]),
        status=result.status,
        steps=result.steps,
        tools_used=result.tools_used,
        tool_calls=result.tool_calls,
        rewritten_question=result.rewritten_question,
        events=[event.to_dict() for event in result.events],
    )


class _ToolCallView:
    """把 tool_calls 记录包装成 TraceRecorder 期望的对象（duck typing）。"""

    def __init__(self, call: Dict[str, Any]) -> None:
        self.ok = bool(call.get("ok", True))
        self.degraded = bool(call.get("retried", False))
        self.error = None if self.ok else str(call.get("error_type") or "tool_error")
        self.error_type = None if self.ok else str(call.get("error_type") or "tool_error")
        self.latency_ms = int(call.get("latency_ms", 0) or 0)
        self.meta: Dict[str, Any] = {}


def _build_response(**kwargs: Any) -> ChatResponse:
    """统一组装响应（避免两处重复的字段拼装）。"""
    sources = [SourceItem(**item) for item in kwargs.pop("sources", [])]
    return ChatResponse(sources=sources, **kwargs)


@router.get("/refusal-message", summary="获取标准拒答话术")
async def refusal_message() -> Dict[str, Any]:
    """返回系统使用的标准拒答话术。

    前端与评测脚本都从这里取常量，避免多处硬编码导致对不上。
    """
    return {"ok": True, "refusal_message": REFUSAL_MESSAGE}


@router.get("/search", summary="仅检索（不生成答案）")
async def search(q: str, k: int = 3) -> Dict[str, Any]:
    """只做混合检索，返回片段与各路分数。

    用途：调试检索质量、构造评测集、给前端做"检索预览"。
    """
    from fastapi.concurrency import run_in_threadpool

    if not q.strip():
        raise HTTPException(status_code=422, detail="查询不能为空")
    engine = get_rag_engine()
    hits = await run_in_threadpool(engine.retriever.retrieve, q, k)
    return {
        "ok": True,
        "query": q,
        "stats": engine.retriever.last_stats,
        "hits": [
            {
                "rank": index,
                "score": hit.score,
                "vec_score": hit.vec_score,
                "bm25_score": hit.bm25_score,
                "retriever": hit.retriever,
                "file_name": hit.chunk.file_name,
                "page": hit.chunk.page,
                "chunk_id": hit.chunk.chunk_id,
                "text": hit.chunk.text[:200],
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }


__all__ = ["router", "ChatRequest", "ChatResponse"]
