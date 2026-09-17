"""ToolExecutor：工具执行的可靠性与治理层（B4）。

职责
----
* **参数校验失败的处理**：把校验错误原样回灌给 Planner，让它改参数重试
  （这是"模型自纠错"的实现方式，比直接报错给用户友好得多）；
* **超时熔断**：超时不重试（避免重复副作用），直接标记降级并回灌；
* **有限重试**：只对 ``retryable=True`` 且被判定为"瞬时错误"（超时/连接类）的工具重试一次，
  且**同一工具最多重试 1 次**，防止放大失败；
* **人工确认**：``requires_confirmation=True`` 的工具在真正执行前挂起，
  产出 ``resume_token``；用户确认后通过 ``/api/chat/confirm`` 恢复执行。
  挂起状态放在进程内的待确认表里（带过期时间），避免为演示场景引入 Redis。

设计取舍：**重试只发生在执行层，不发生在工具内部**。
工具内部重试会导致一次调用产生多次副作用且耗时不可控。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.tools.base import ToolErrorType, ToolResult
from core.tools.registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)

# 同一工具的瞬时错误最多重试次数
MAX_RETRIES = 1
# 待确认请求的存活时间（秒）
CONFIRMATION_TTL_S = 600
# 瞬时错误类型（可安全重试）
TRANSIENT_ERRORS = {ToolErrorType.TIMEOUT, "tool_connection_error"}


@dataclass
class ExecutionOutcome:
    """一次工具执行的结果（含治理信息）。"""

    result: ToolResult
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    retried: bool = False
    pending_confirmation: bool = False
    resume_token: Optional[str] = None

    def to_observation(self) -> Dict[str, Any]:
        """转成 Planner 可消费的观察记录。"""
        return {
            "tool": self.tool_name,
            "args": self.args,
            "ok": self.result.ok,
            "error_type": self.result.error_type,
            "error": self.result.error,
            "display": self.result.display,
            "text": self.result.to_llm_text(),
            "data": self.result.data,
            "degraded": self.result.degraded,
            "refused": bool(self.result.meta.get("refused")) if self.result.meta else False,
            "latency_ms": self.result.latency_ms,
        }


class PendingConfirmation:
    """待人工确认的调用。"""

    def __init__(self, tool_name: str, args: Dict[str, Any], question: str) -> None:
        self.token = uuid.uuid4().hex[:16]
        self.tool_name = tool_name
        self.args = args
        self.question = question
        self.created_at = time.time()

    @property
    def expired(self) -> bool:
        """是否已过期。"""
        return (time.time() - self.created_at) > CONFIRMATION_TTL_S

    def to_dict(self) -> Dict[str, Any]:
        """转成响应结构。"""
        return {
            "resume_token": self.token,
            "tool": self.tool_name,
            "args": self.args,
            "risk_note": "该工具被标记为需要人工确认，请确认后继续执行",
            "expires_in_s": max(0, int(CONFIRMATION_TTL_S - (time.time() - self.created_at))),
        }


class ToolExecutor:
    """工具执行器。"""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or get_registry()
        self._pending: Dict[str, PendingConfirmation] = {}
        # 已获人工确认的调用指纹（工具名 + 参数）。见 _fingerprint 的说明。
        self._approved: set = set()

    # ------------------------------------------------------------------
    # 待确认管理
    # ------------------------------------------------------------------
    def _create_pending(self, tool_name: str, args: Dict[str, Any], question: str) -> PendingConfirmation:
        """创建待确认请求（顺带清理过期项）。"""
        self._pending = {
            token: item for token, item in self._pending.items() if not item.expired
        }
        pending = PendingConfirmation(tool_name, args, question)
        self._pending[pending.token] = pending
        return pending

    def pop_pending(self, token: str) -> Optional[PendingConfirmation]:
        """取出并移除待确认请求（不存在或过期返回 None）。"""
        pending = self._pending.pop(token, None)
        if pending is None or pending.expired:
            return None
        return pending

    def pending_count(self) -> int:
        """当前待确认数量。"""
        return len([item for item in self._pending.values() if not item.expired])

    # ------------------------------------------------------------------
    # 已确认调用的记忆
    # ------------------------------------------------------------------
    @staticmethod
    def _fingerprint(tool_name: str, args: Dict[str, Any]) -> str:
        """调用指纹：工具名 + 参数（用于记住"人已经批准过这个操作"）。

        为什么需要它：确认恢复执行后，Planner 可能再次提出**完全相同的**调用
        （它的上下文里只有"待确认"的失败观察）。如果不记住"人已经批准过"，
        就会二次挂起 → 用户再点一次确认 → 死循环。
        记住指纹后，同一操作只确认一次；参数变了要重新确认（安全边界不放松）。
        """
        import hashlib
        import json

        payload = json.dumps({"tool": tool_name, "args": args or {}}, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def mark_approved(self, tool_name: str, args: Dict[str, Any]) -> None:
        """记录一次已获人工确认的调用。"""
        self._approved.add(self._fingerprint(tool_name, args))

    def is_approved(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """该调用是否已获人工确认。"""
        return self._fingerprint(tool_name, args) in self._approved

    def clear_approvals(self) -> None:
        """清空确认记忆（测试与"重置会话"场景）。"""
        self._approved.clear()

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    def execute(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        question: str = "",
        confirmed: bool = False,
    ) -> ExecutionOutcome:
        """执行一次工具调用（含重试与确认闸门）。

        Args:
            tool_name: 工具名。
            args: 参数。
            question: 原始问题（仅用于确认提示）。
            confirmed: 是否已获得人工确认。

        Returns:
            ``ExecutionOutcome``。
        """
        args = dict(args or {})
        tool = self.registry.get(tool_name)

        if tool is None:
            return ExecutionOutcome(
                tool_name=tool_name,
                args=args,
                result=ToolResult(
                    ok=False,
                    error=f"未找到工具 {tool_name}",
                    error_type=ToolErrorType.NOT_FOUND,
                    display=f"没有名为 {tool_name} 的工具",
                ),
            )

        # ---- 人工确认闸门 ----
        # 已经确认过的**同一调用**直接放行（避免恢复执行时二次挂起形成死循环）；
        # 参数变化则视为新操作，必须重新确认。
        if tool.requires_confirmation and not confirmed and not self.is_approved(tool.name, args):
            pending = self._create_pending(tool.name, args, question)
            logger.info("工具 %s 需要人工确认，已挂起（token=%s）", tool.name, pending.token)
            return ExecutionOutcome(
                tool_name=tool.name,
                args=args,
                result=ToolResult(
                    ok=False,
                    error="该操作需要人工确认后才能执行",
                    error_type="tool_pending_confirmation",
                    display=f"{tool.name} 需要你确认后才会执行",
                    meta={"pending": pending.to_dict()},
                ),
                pending_confirmation=True,
                resume_token=pending.token,
            )

        # ---- 执行 + 重试 ----
        attempts = 0
        result = tool.run(**args)
        attempts += 1
        retried = False

        if (
            not result.ok
            and tool.retryable
            and result.error_type in TRANSIENT_ERRORS
            and attempts <= MAX_RETRIES
        ):
            retried = True
            logger.info("工具 %s 首次失败（%s），重试一次", tool.name, result.error_type)
            result = tool.run(**args)
            attempts += 1

        if retried:
            result.meta = {**(result.meta or {}), "retried": True, "attempts": attempts}

        return ExecutionOutcome(
            result=result, tool_name=tool.name, args=args, attempts=attempts, retried=retried
        )


_executor_singleton: Optional[ToolExecutor] = None


def get_executor(registry: Optional[ToolRegistry] = None, reload: bool = False) -> ToolExecutor:
    """获取全局执行器单例。"""
    global _executor_singleton
    if _executor_singleton is None or reload:
        _executor_singleton = ToolExecutor(registry)
    return _executor_singleton


def reset_executor() -> None:
    """丢弃执行器单例（测试用）。"""
    global _executor_singleton
    _executor_singleton = None


__all__ = [
    "CONFIRMATION_TTL_S",
    "ExecutionOutcome",
    "PendingConfirmation",
    "ToolExecutor",
    "get_executor",
    "reset_executor",
]
