"""执行轨迹记录器（可观测性的核心）。

一次提问 = 一个 run，run 内部包含若干 step（plan / rewrite / tool / respond / refuse）。
所有数据落 SQLite，前端可通过 ``/api/traces/{run_id}`` 还原"这次回答是怎么产生的"：

* Planner 在第几步决定调用哪个工具、参数是什么；
* 每次工具调用的耗时、是否降级、失败原因；
* 检索命中几个片段、最高分多少、是否触发拒答；
* LLM 的 token 用量与估算成本。

用法::

    recorder = TraceRecorder(session_id="s1")
    recorder.start(question="年假怎么算？")
    recorder.plan(thought="需要检索知识库", tool="knowledge_search", args={...})
    recorder.tool(tool_name="knowledge_search", args={...}, result=tool_result)
    recorder.finish(answer="...", refused=False)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from infra import db, metrics

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """单步记录的运行期载体（最终会写入 steps 表）。"""

    idx: int
    type: str
    tool_name: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    degraded: bool = False
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_db(self, run_id: str) -> Dict[str, Any]:
        """转换为数据库行。"""
        return {
            "run_id": run_id,
            "idx": self.idx,
            "type": self.type,
            "tool_name": self.tool_name,
            "args": self.args,
            "ok": 1 if self.ok else 0,
            "degraded": 1 if self.degraded else 0,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
            "error_type": self.error_type,
            "detail": self.detail,
        }


class TraceRecorder:
    """一次提问的轨迹记录器。"""

    def __init__(self, session_id: str, question: str = "", enabled: Optional[bool] = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.enabled = settings.trace_enabled if enabled is None else enabled
        self.session_id = session_id
        self.run_id = db.new_id("run_")
        self.question = question
        self.rewritten: Optional[str] = None
        self.answer = ""
        self.refused = False
        self.status = "success"
        self.tools_used: List[str] = []
        self.degraded = False
        self.total_tokens = 0
        self.cost_est = 0.0
        self.steps: List[StepRecord] = []
        self._started = time.perf_counter()
        self._finished = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self, question: Optional[str] = None) -> str:
        """开始记录，返回 run_id。"""
        if question is not None:
            self.question = question
        if self.enabled:
            try:
                db.insert_run(
                    {
                        "run_id": self.run_id,
                        "session_id": self.session_id,
                        "question": self.question,
                        "rewritten": self.rewritten,
                        "answer": "",
                        "refused": 0,
                        "status": "running",
                        "steps": 0,
                        "tools_used": "",
                        "total_ms": 0,
                        "total_tokens": 0,
                        "cost_est": 0.0,
                        "degraded": 0,
                    }
                )
            except Exception as exc:  # 观测失败绝不影响主流程
                logger.warning("写入 run 记录失败（忽略）：%s", exc)
        return self.run_id

    def add_step(self, step: StepRecord) -> None:
        """追加一步并落库。"""
        self.steps.append(step)
        if step.tool_name and step.tool_name not in self.tools_used:
            self.tools_used.append(step.tool_name)
        if step.degraded:
            self.degraded = True
        self.total_tokens += step.prompt_tokens + step.completion_tokens
        if self.enabled:
            try:
                db.insert_step(step.to_db(self.run_id))
            except Exception as exc:
                logger.warning("写入 step 记录失败（忽略）：%s", exc)

    # ------------------------------------------------------------------
    # 便捷方法：按步骤类型记录
    # ------------------------------------------------------------------
    def plan(
        self,
        thought: str,
        action: str,
        tool: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        latency_ms: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        degraded: bool = False,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> StepRecord:
        """记录一次 Planner 决策。"""
        payload = {"thought": thought, "action": action}
        if detail:
            payload.update(detail)
        step = StepRecord(
            idx=len(self.steps),
            type="plan",
            tool_name=tool,
            args=args or {},
            ok=error is None,
            degraded=degraded,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
            error_type=error_type,
            detail=payload,
        )
        self.add_step(step)
        return step

    def rewrite(self, original: str, rewritten: str, reason: str = "", latency_ms: int = 0) -> StepRecord:
        """记录一次问题改写（多轮指代消解）。"""
        self.rewritten = rewritten
        step = StepRecord(
            idx=len(self.steps),
            type="rewrite",
            latency_ms=latency_ms,
            detail={"original": original, "rewritten": rewritten, "reason": reason},
        )
        self.add_step(step)
        return step

    def tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        latency_ms: Optional[int] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> StepRecord:
        """记录一次工具调用。

        Args:
            result: ``ToolResult``（duck typing，避免 infra 层反向依赖 core）。
        """
        ok = bool(getattr(result, "ok", True))
        degraded = bool(getattr(result, "degraded", False))
        latency = latency_ms if latency_ms is not None else int(getattr(result, "latency_ms", 0))
        payload = dict(getattr(result, "meta", {}) or {})
        if detail:
            payload.update(detail)
        step = StepRecord(
            idx=len(self.steps),
            type="tool",
            tool_name=tool_name,
            args=args,
            ok=ok,
            degraded=degraded,
            latency_ms=latency,
            error=getattr(result, "error", None),
            error_type=getattr(result, "error_type", None),
            detail=payload,
        )
        self.add_step(step)
        metrics.observe_tool(tool_name, latency, ok, degraded)
        if getattr(result, "error_type", None) == "tool_timeout":
            metrics.incr("tool_timeouts")
        return step

    def respond(
        self,
        answer: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_est: float = 0.0,
        latency_ms: int = 0,
        degraded: bool = False,
        detail: Optional[Dict[str, Any]] = None,
    ) -> StepRecord:
        """记录回答生成步骤。"""
        self.answer = answer
        self.cost_est += cost_est
        if degraded:
            self.status = "degraded"
        step = StepRecord(
            idx=len(self.steps),
            type="respond",
            ok=True,
            degraded=degraded,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            detail=detail or {},
        )
        self.add_step(step)
        if prompt_tokens or completion_tokens:
            metrics.observe_llm(prompt_tokens + completion_tokens, cost_est, ok=True)
        return step

    def refuse(self, reason: str, top_score: float = 0.0, detail: Optional[Dict[str, Any]] = None) -> StepRecord:
        """记录一次拒答。"""
        self.refused = True
        self.status = "refused"
        payload = {"reason": reason, "top_score": round(top_score, 4)}
        if detail:
            payload.update(detail)
        step = StepRecord(idx=len(self.steps), type="refuse", detail=payload)
        self.add_step(step)
        metrics.incr("refusals")
        return step

    def fail(self, error: str, error_type: str = "runtime_error") -> None:
        """记录一次整体失败（仍会写 run，便于事后排查）。"""
        self.status = "failed"
        self.add_step(
            StepRecord(
                idx=len(self.steps),
                type="respond",
                ok=False,
                error=error,
                error_type=error_type,
            )
        )

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def finish(
        self,
        answer: Optional[str] = None,
        status: Optional[str] = None,
        total_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """结束记录并落库，返回汇总字典。"""
        if answer is not None:
            self.answer = answer
        if status is not None:
            self.status = status
        if self._finished:
            return self.summary()
        self._finished = True

        elapsed = (
            total_ms if total_ms is not None else int((time.perf_counter() - self._started) * 1000)
        )
        metrics.incr("chat_requests")
        metrics.observe_latency(elapsed)
        if self.status == "failed":
            metrics.incr("chat_errors")
        if self.degraded:
            metrics.incr("degraded_runs")

        if self.enabled:
            try:
                db.insert_run(
                    {
                        "run_id": self.run_id,
                        "session_id": self.session_id,
                        "question": self.question,
                        "rewritten": self.rewritten,
                        "answer": self.answer,
                        "refused": 1 if self.refused else 0,
                        "status": self.status,
                        "steps": len(self.steps),
                        "tools_used": ",".join(self.tools_used),
                        "total_ms": elapsed,
                        "total_tokens": self.total_tokens,
                        "cost_est": round(self.cost_est, 6),
                        "degraded": 1 if self.degraded else 0,
                    }
                )
            except Exception as exc:
                logger.warning("写入 run 汇总失败（忽略）：%s", exc)
        return self.summary(total_ms=elapsed)

    def summary(self, total_ms: Optional[int] = None) -> Dict[str, Any]:
        """返回本次执行的汇总信息（也会随 final 事件发给前端）。"""
        return {
            "run_id": self.run_id,
            "trace_id": self.run_id,
            "session_id": self.session_id,
            "question": self.question,
            "rewritten_question": self.rewritten,
            "answer": self.answer,
            "refused": self.refused,
            "status": self.status,
            "steps": len(self.steps),
            "tools_used": self.tools_used,
            "degraded": self.degraded,
            "total_tokens": self.total_tokens,
            "cost_est": round(self.cost_est, 6),
            "total_ms": total_ms if total_ms is not None else int((time.perf_counter() - self._started) * 1000),
        }


# ---------------------------------------------------------------------------
# 查询接口（供 API 层直接调用）
# ---------------------------------------------------------------------------
def get_trace(run_id: str) -> Optional[Dict[str, Any]]:
    """读取一次执行的完整轨迹。"""
    return db.get_run(run_id)


def list_traces(limit: int = 20, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出最近的执行轨迹。"""
    return db.list_runs(limit=limit, session_id=session_id)
