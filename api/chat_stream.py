"""SSE 流式问答（B5）。

接口
----
``POST /api/chat/stream``   流式问答：边执行边推送事件（``text/event-stream``）
``POST /api/chat/confirm``  人工确认后恢复被挂起的执行（同样支持流式）

为什么需要流式
--------------
Agent 一次问答可能耗时几秒到几十秒（多步决策 + 工具执行 + 大模型生成）。
如果等全部算完再返回，用户面对的是十几秒白屏；流式则让过程**可见**：
"正在思考 → 决定调用知识库 → 检索到 3 条 → 正在生成回答 → 完成"。

实现要点
--------
* **不阻塞事件循环**：Agent 是同步代码，放到线程池执行；线程通过
  ``asyncio.run_coroutine_threadsafe`` 把事件投递到 ``asyncio.Queue``，
  异步生成器从队列里取事件并按 SSE 协议推送；
* **心跳保活**：长耗时步骤之间插入注释行 ``: ping``，避免中间代理断开空闲连接；
* **事件契约复用**：直接使用 ``EventEmitter`` 的事件类型，与 ``/api/chat`` 返回的
  ``events`` 字段完全一致——前端同一套解析逻辑，既能回放也能实时渲染；
* **收尾落 trace**：流结束后把 run/step 落库，并写入会话消息。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.runtime.events import AgentEvent
from core.runtime.resume import PendingExecution, get_pending_store
from infra import db, metrics
from infra.trace import TraceRecorder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["流式问答"])

# 队列容量：事件很小，256 足够；满了说明客户端读取太慢，宁可丢事件也不阻塞 Agent
QUEUE_MAXSIZE = 256
# 心跳间隔（秒）——用于长耗时步骤之间的保活
HEARTBEAT_INTERVAL_S = 5.0
# SSE 响应头（禁用缓冲，保证逐条推送）
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",   # Nginx 场景下关闭缓冲
}


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class StreamChatRequest(BaseModel):
    """流式问答请求。"""

    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 id；不传则自动生成")
    max_steps: Optional[int] = Field(None, ge=1, le=12, description="Agent 步数预算")
    use_history: bool = Field(True, description="是否读取会话历史（多轮指代消解需要）")


class ConfirmRequest(BaseModel):
    """人工确认请求。"""

    resume_token: str = Field(..., min_length=1, description="挂起时返回的 resume_token")
    approved: bool = Field(True, description="是否同意执行该操作")
    session_id: Optional[str] = Field(None, description="会话 id（可选，用于校验）")


# ---------------------------------------------------------------------------
# SSE 工具
# ---------------------------------------------------------------------------
def sse_frame(event_type: str, payload: Dict[str, Any]) -> str:
    """构造一条 SSE 帧。"""
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n"


async def event_stream(
    queue: "asyncio.Queue[Optional[AgentEvent]]",
    heartbeat_s: float = HEARTBEAT_INTERVAL_S,
) -> AsyncIterator[str]:
    """把 Agent 事件队列转成 SSE 字节流。

    收到 ``None`` 表示执行结束；等待期间按 ``heartbeat_s`` 发送注释行保活。
    """
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
        except asyncio.TimeoutError:
            yield ": ping\n\n"      # SSE 注释行：客户端忽略，仅用于保活
            continue

        if event is None:
            break
        yield sse_frame(event.type, event.to_dict())


# ---------------------------------------------------------------------------
# 内部执行：把同步 Agent 跑在线程里，事件经队列回传
# ---------------------------------------------------------------------------
async def _run_agent_streaming(
    question: str,
    session_id: str,
    history: List[Dict[str, str]],
    max_steps: Optional[int],
    run_id: str,
    resume_context: Optional[PendingExecution] = None,
) -> tuple[StreamingResponse, "asyncio.Queue[Optional[AgentEvent]]"]:
    """启动 Agent 执行并返回（SSE 响应, 事件队列）。"""
    from core.runtime.agent import get_agent_runtime

    runtime = get_agent_runtime()
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Optional[AgentEvent]]" = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    def on_event(event: AgentEvent) -> None:
        """线程中的事件回调：投递到事件循环（队列满时丢弃，避免阻塞 Agent）。"""
        try:
            loop.call_soon_threadsafe(_put_nowait, queue, event)
        except RuntimeError:  # 事件循环已关闭（客户端断开）
            pass

    def worker() -> None:
        """在线程池中执行 Agent，结束后投递结束标记。"""
        try:
            result = runtime.run(
                question,
                session_id=session_id,
                history=history,
                max_steps=max_steps,
                emit=on_event,
                run_id=run_id,
                observations=resume_context.observations if resume_context else None,
                tool_calls=resume_context.tool_calls if resume_context else None,
                tools_used=resume_context.tools_used if resume_context else None,
                rewritten_question=resume_context.rewritten_question if resume_context else None,
                resumed=resume_context is not None,
            )
        except Exception as exc:  # noqa: BLE001 - 兜底：异常也要让前端看到
            logger.exception("流式执行失败：%s", exc)
            loop.call_soon_threadsafe(
                _put_nowait,
                queue,
                AgentEvent(
                    type="error",
                    run_id=run_id,
                    payload={"message": f"{type(exc).__name__}: {exc}", "error_type": "runtime_error"},
                ),
            )
            loop.call_soon_threadsafe(_put_nowait, queue, None)
            return

        # 需要人工确认 → 保存挂起上下文，供 /api/chat/confirm 恢复
        if result.status == "pending_confirmation" and result.pending:
            _save_pending(result, session_id, question, history)

        _finalize_trace(result, session_id, run_id)
        loop.call_soon_threadsafe(_put_nowait, queue, None)

    loop.run_in_executor(None, worker)
    return StreamingResponse(event_stream(queue), media_type="text/event-stream", headers=SSE_HEADERS), queue


def _put_nowait(queue: "asyncio.Queue[Optional[AgentEvent]]", event: Optional[AgentEvent]) -> None:
    """非阻塞投递（队列满时丢弃事件，优先保证 Agent 不被拖慢）。"""
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.warning("SSE 事件队列已满，丢弃一条事件（客户端读取过慢）")


def _save_pending(
    result: Any,
    session_id: str,
    question: str,
    history: List[Dict[str, str]],
) -> None:
    """保存挂起上下文。"""
    pending = result.pending or {}
    execution = PendingExecution(
        token=str(pending.get("resume_token") or ""),
        question=question,
        session_id=session_id,
        tool=str(pending.get("tool") or ""),
        args=dict(pending.get("args") or {}),
        rewritten_question=result.rewritten_question,
        history=list(history),
        observations=list(result.observations),
        tool_calls=list(result.tool_calls),
        tools_used=list(result.tools_used),
    )
    get_pending_store().save(execution)


def _finalize_trace(result: Any, session_id: str, run_id: str) -> None:
    """把流式执行的结果落 trace 与会话消息（与 /api/chat 保持一致）。"""
    try:
        recorder = TraceRecorder(session_id=session_id, question=result.question or "")
        # 复用外部 run_id，保证前端拿到的事件与 trace 表的记录是同一条
        recorder.run_id = run_id
        recorder.start()

        for event in result.events:
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
                str(call.get("tool") or "unknown"),
                dict(call.get("args") or {}),
                _ToolCallView(call),
                latency_ms=int(call.get("latency_ms", 0) or 0),
                detail={"step": call.get("step"), "retried": call.get("retried")},
            )
        if result.refused:
            recorder.refuse(reason=str(result.refuse_reason), top_score=0.0, detail={"mode": "agent_stream"})
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
        recorder.finish(answer=result.answer, status=result.status)

        db.add_message(session_id, "user", result.question or "", run_id=run_id)
        db.add_message(session_id, "assistant", result.answer or "", run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - 观测失败不能影响已经返回给用户的结果
        logger.warning("落 trace 失败（忽略）：%s", exc)


class _ToolCallView:
    """把 tool_calls 记录包装成 TraceRecorder 期望的对象（duck typing）。"""

    def __init__(self, call: Dict[str, Any]) -> None:
        self.ok = bool(call.get("ok", True))
        self.degraded = bool(call.get("retried", False))
        self.error = None if self.ok else str(call.get("error_type") or "tool_error")
        self.error_type = None if self.ok else str(call.get("error_type") or "tool_error")
        self.latency_ms = int(call.get("latency_ms", 0) or 0)
        self.meta: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@router.post("/chat/stream", summary="流式问答（SSE）")
async def chat_stream(request: StreamChatRequest) -> StreamingResponse:
    """边执行边推送事件。

    事件类型与 ``/api/chat`` 的 ``events`` 字段完全一致：``plan`` / ``tool_start`` /
    ``tool_end`` / ``observation`` / ``refuse`` / ``pending_confirmation`` / ``final`` / ``error`` / ``done``。
    前端按 ``event:`` 名分发即可，无需了解 Agent 内部结构。
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")

    session_id = request.session_id or db.new_id("sess_")
    history: List[Dict[str, str]] = []
    if request.use_history:
        history = [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in db.get_history(session_id, limit=20)
        ]

    run_id = db.new_id("run_")
    metrics.incr("chat_requests")
    response, _queue = await _run_agent_streaming(
        question, session_id, history, request.max_steps, run_id
    )
    # 让客户端能拿到 session_id（放在响应头，避免污染事件流）
    response.headers["X-Session-Id"] = session_id
    response.headers["X-Run-Id"] = run_id
    return response


@router.post("/chat/confirm", summary="人工确认后恢复执行（SSE）")
async def chat_confirm(request: ConfirmRequest) -> StreamingResponse:
    """确认或拒绝一个被挂起的危险操作。

    * ``approved=true``：真正执行该工具，并把它的结果并入已有观察，继续完成回答；
    * ``approved=false``：不执行，直接把"用户已拒绝"作为工具结果回灌给 Agent，
      由它给出后续回答（通常是改口提供替代方案）。

    注意：确认后会**删除挂起记录**，同一 token 无法重复执行——
    这是防止"点两次确认按钮导致重复发邮件"的必要保护。
    """
    store = get_pending_store()
    execution = store.load(request.resume_token)
    if execution is None:
        raise HTTPException(
            status_code=404,
            detail="确认请求不存在或已过期（有效期 15 分钟），请重新发起该操作",
        )

    session_id = request.session_id or execution.session_id
    run_id = db.new_id("run_")
    resumed_execution = PendingExecution(
        token=execution.token,
        question=execution.rewritten_question or execution.question,
        session_id=session_id,
        tool=execution.tool,
        args=execution.args,
        rewritten_question=execution.rewritten_question,
        history=execution.history,
        observations=list(execution.observations),
        tool_calls=list(execution.tool_calls),
        tools_used=list(execution.tools_used),
    )

    if request.approved:
        resumed_execution.observations.append(_execute_confirmed_tool(execution))
        resumed_execution.tools_used = list(dict.fromkeys([*execution.tools_used, execution.tool]))
    else:
        resumed_execution.observations.append(
            {
                "tool": execution.tool,
                "args": execution.args,
                "ok": True,
                "error_type": None,
                "error": None,
                "display": f"用户拒绝执行「{execution.tool}」，该操作未生效",
                "text": f"工具 {execution.tool} 被用户拒绝执行，未产生任何副作用。请告知用户操作已取消。",
                "data": {"declined": True},
                "degraded": False,
                "refused": False,
                "latency_ms": 0,
            }
        )

    # 无论同意还是拒绝，都删除挂起记录（防重复执行）
    store.drop(request.resume_token)

    response, _queue = await _run_agent_streaming(
        resumed_execution.question,
        session_id,
        resumed_execution.history,
        None,
        run_id,
        resume_context=resumed_execution,
    )
    response.headers["X-Session-Id"] = session_id
    response.headers["X-Run-Id"] = run_id
    return response


def _execute_confirmed_tool(execution: PendingExecution) -> Dict[str, Any]:
    """执行已确认的工具，返回观察记录。"""
    from core.runtime.executor import get_executor

    executor = get_executor()
    # 记住"这个调用已经被人批准过"：恢复执行后 Planner 若再次提出同一调用，
    # 直接放行而不是二次挂起（否则会形成"确认→挂起→再确认"的死循环）。
    executor.mark_approved(execution.tool, execution.args)
    outcome = executor.execute(
        execution.tool, execution.args, question=execution.question, confirmed=True
    )
    logger.info(
        "人工确认后执行工具：%s（ok=%s，耗时 %sms）",
        execution.tool, outcome.result.ok, outcome.result.latency_ms,
    )
    return outcome.to_observation()


@router.get("/pending", summary="查看待确认操作")
async def list_pending() -> Dict[str, Any]:
    """列出当前挂起等待确认的操作（UI 与调试都用得上）。"""
    store = get_pending_store()
    # 通过公开接口拿到明细：这里简单遍历内部字典（数量很少）
    items: List[Dict[str, Any]] = []
    for token in list(getattr(store, "_items", {}).keys()):  # noqa: SLF001
        execution = store.load(token)
        if execution is not None:
            items.append(execution.to_dict())
    return {"ok": True, "total": len(items), "pending": items}


__all__ = ["router", "sse_frame"]
