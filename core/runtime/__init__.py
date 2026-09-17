"""Agent 运行时：事件、决策、执行与主循环。

========================  ==================================================
模块                       职责
========================  ==================================================
``events``                AgentEvent / EventType / EventEmitter（SSE 契约）
``planner``               决策中枢：LLM 自主路由 + 规则兜底 + 问题改写
``executor``              工具执行：重试、超时熔断、人工确认挂起
``agent``                 主循环：步数预算、观察累积、回答汇总、拒答优先
========================  ==================================================
"""

from .agent import AgentResult, AgentRuntime, get_agent_runtime, reset_agent_runtime  # noqa: F401
from .events import AgentEvent, EventEmitter, EventType  # noqa: F401
from .executor import (  # noqa: F401
    ExecutionOutcome,
    PendingConfirmation,
    ToolExecutor,
    get_executor,
    reset_executor,
)
from .planner import Decision, Planner  # noqa: F401

__all__ = [
    "AgentEvent",
    "AgentResult",
    "AgentRuntime",
    "Decision",
    "EventEmitter",
    "EventType",
    "ExecutionOutcome",
    "PendingConfirmation",
    "Planner",
    "ToolExecutor",
    "get_agent_runtime",
    "get_executor",
    "reset_agent_runtime",
    "reset_executor",
]
