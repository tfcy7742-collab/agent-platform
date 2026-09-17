"""人工确认的挂起与恢复（B5 的 human-in-the-loop 闭环）。

为什么需要它
------------
``SendEmailTool`` 这类**有副作用**的工具不能直接执行：Agent 必须先挂起、
把"我准备做什么"告诉用户，用户确认后才真正执行。

跨请求恢复需要保存"当时执行到哪了"：
* 用户问题（含改写后的问题）；
* 会话 id；
* 已完成的工具观察（``observations``）——**这是关键**：
  恢复时必须带着它们继续，否则 Agent 会从头重跑一遍（重复调用工具、重复花钱）；
* 还没执行的工具名与参数。

存储取舍
--------
用**进程内字典 + TTL**，不引入 Redis。理由：单机演示/单实例部署下完全够用，
引入 Redis 只会增加部署复杂度；而这块的接口（``save`` / ``load`` / ``drop``）
是清晰的，将来换成 Redis 只需替换实现。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 待确认记录的存活时间（秒）
PENDING_TTL_S = 900


@dataclass
class PendingExecution:
    """一次被挂起的 Agent 执行（等待人工确认）。"""

    token: str
    question: str
    session_id: str
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    rewritten_question: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        """是否已过期。"""
        return (time.time() - self.created_at) > PENDING_TTL_S

    @property
    def expires_in_s(self) -> int:
        """剩余有效时间（秒）。"""
        return max(0, int(PENDING_TTL_S - (time.time() - self.created_at)))

    def to_dict(self) -> Dict[str, Any]:
        """给前端的提示信息。"""
        return {
            "resume_token": self.token,
            "tool": self.tool,
            "args": self.args,
            "question": self.question,
            "expires_in_s": self.expires_in_s,
            "risk_note": (
                f"准备执行「{self.tool}」，这是一个**有副作用**的操作，"
                f"参数：{self.args}。请确认后继续。"
            ),
        }


class PendingStore:
    """挂起记录的进程内存储（线程安全）。"""

    def __init__(self) -> None:
        self._items: Dict[str, PendingExecution] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        """清理过期项（调用方需已持有锁）。"""
        expired = [token for token, item in self._items.items() if item.expired]
        for token in expired:
            self._items.pop(token, None)
        if expired:
            logger.info("清理过期的待确认记录：%s 条", len(expired))

    def save(self, execution: PendingExecution) -> PendingExecution:
        """保存挂起记录。"""
        with self._lock:
            self._purge()
            self._items[execution.token] = execution
        logger.info(
            "已挂起 Agent 执行：token=%s tool=%s（有效期 %ss）",
            execution.token, execution.tool, execution.expires_in_s,
        )
        return execution

    def load(self, token: str) -> Optional[PendingExecution]:
        """读取挂起记录（不删除，允许重复确认时报错更清晰）。"""
        with self._lock:
            self._purge()
            item = self._items.get(token)
        return item

    def drop(self, token: str) -> None:
        """删除挂起记录（执行完成后必须调用，避免重复执行同一操作）。"""
        with self._lock:
            self._items.pop(token, None)

    def count(self) -> int:
        """当前挂起数量。"""
        with self._lock:
            self._purge()
            return len(self._items)


_store_singleton: Optional[PendingStore] = None


def get_pending_store() -> PendingStore:
    """获取全局挂起存储。"""
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = PendingStore()
    return _store_singleton


def reset_pending_store() -> None:
    """清空挂起存储（测试用）。"""
    global _store_singleton
    _store_singleton = None


__all__ = [
    "PENDING_TTL_S",
    "PendingExecution",
    "PendingStore",
    "get_pending_store",
    "reset_pending_store",
]
