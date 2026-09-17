"""Agent 事件模型（SSE 契约，B4 落地，B5 用于流式推送）。

统一信封的好处：前端只需按 ``type`` 分发，未知类型直接忽略，
因此**前后端可以独立升级**（新增事件类型不会打挂老前端）。

事件序列（典型一次问答）::

    plan → tool_start → tool_end → observation → plan(可选，多步) → token* → final → done

特殊情况::

    refuse                 拒答（伴随 final）
    pending_confirmation   危险工具挂起，等待人工确认（B4 支持协议，UI 在 B5 接入）
    error                  整体失败
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class EventType:
    """事件类型常量。"""

    PLAN = "plan"                             # Planner 决策
    REWRITE = "rewrite"                       # 问题改写
    TOOL_START = "tool_start"                 # 工具开始执行
    TOOL_END = "tool_end"                     # 工具执行结束
    OBSERVATION = "observation"               # 工具结果裁剪后进入上下文
    TOKEN = "token"                           # 流式文本增量
    REFUSE = "refuse"                         # 拒答
    PENDING_CONFIRMATION = "pending_confirmation"  # 等待人工确认
    FINAL = "final"                           # 最终答案
    ERROR = "error"                           # 失败
    DONE = "done"                             # 流结束标记


@dataclass
class AgentEvent:
    """一条 Agent 事件。"""

    type: str
    run_id: str = ""
    seq: int = 0
    ts: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的字典（SSE 的 data 字段）。"""
        return {
            "type": self.type,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": round(self.ts, 3),
            "payload": self.payload,
        }

    def to_sse(self) -> str:
        """转成 SSE 帧（B5 使用）。"""
        import json

        return f"event: {self.type}\ndata: {json.dumps(self.to_dict(), ensure_ascii=False)}\n\n"


class EventEmitter:
    """事件收集器：把事件按序缓存，并可选地实时回调（B5 流式用）。"""

    def __init__(self, run_id: str = "", callback: Optional[Any] = None) -> None:
        self.run_id = run_id
        self.callback = callback      # callable(AgentEvent) -> None
        self.events: list = []
        self._seq = 0

    def emit(self, event_type: str, **payload: Any) -> AgentEvent:
        """产生一条事件。"""
        event = AgentEvent(
            type=event_type,
            run_id=self.run_id,
            seq=self._seq,
            payload=payload,
        )
        self._seq += 1
        self.events.append(event)
        if self.callback is not None:
            try:
                self.callback(event)
            except Exception:  # noqa: BLE001 - 回调异常绝不能影响主流程
                import logging

                logging.getLogger(__name__).warning("事件回调失败（忽略）", exc_info=True)
        return event

    def types(self) -> list:
        """已产生的事件类型序列（测试与调试用）。"""
        return [event.type for event in self.events]


__all__ = ["AgentEvent", "EventEmitter", "EventType"]
