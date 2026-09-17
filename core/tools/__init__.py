"""工具层：统一工具协议与注册表。

======================  ====================================================
模块                     职责
======================  ====================================================
``base``                BaseTool / ToolResult / JSON Schema 参数校验
``registry``            注册、发现、启用禁用、统一调用入口
``knowledge_tool``      knowledge_search：包装 RAG 问答引擎
``trip_tool``           trip_planner：包装多智能体旅行规划子系统
======================  ====================================================
"""

from .base import BaseTool, ToolErrorType, ToolResult, validate_args  # noqa: F401
from .registry import (  # noqa: F401
    TOOL_ALIASES,
    ToolRegistry,
    bootstrap_tools,
    get_registry,
    reset_registry,
)

__all__ = [
    "BaseTool",
    "TOOL_ALIASES",
    "ToolErrorType",
    "ToolRegistry",
    "ToolResult",
    "bootstrap_tools",
    "get_registry",
    "reset_registry",
    "validate_args",
]
