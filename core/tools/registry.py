"""工具注册表（B4）。

职责
----
* 工具的**注册与发现**：Planner 通过 ``catalog()`` 拿到可用工具清单（含说明与 schema），
  因此"新增工具后 Agent 立刻能用"不需要改任何编排代码；
* **启用/禁用**：运行期可以临时关掉某个工具（例如知识库为空时禁掉 knowledge_search）；
* **统一调用入口**：``execute(name, args)`` 是 Executor 与 API 层唯一的调用点；
* **中文工具名映射**：Planner 有时会用中文描述工具（"知识库检索"），
  这里提供别名解析，避免因为叫法不同而路由失败。

注册时机：``bootstrap_tools()`` 在应用启动时调用（见 app.py 的 lifespan），
这样 /api/tools 与 Agent 循环都能看到同一份清单。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# 中文别名 → 工具名（提高 Planner 在中文语境下的路由成功率）
TOOL_ALIASES: Dict[str, str] = {
    "知识库检索": "knowledge_search",
    "知识库搜索": "knowledge_search",
    "文档检索": "knowledge_search",
    "文档问答": "knowledge_search",
    "检索资料": "knowledge_search",
    "查资料": "knowledge_search",
    "旅行规划": "trip_planner",
    "行程规划": "trip_planner",
    "旅游规划": "trip_planner",
    "规划行程": "trip_planner",
    "发邮件": "send_email",
    "发送邮件": "send_email",
    "邮件发送": "send_email",
    "发到邮箱": "send_email",
}


class ToolRegistry:
    """工具注册表（进程内单例，线程安全）。"""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 注册与查找
    # ------------------------------------------------------------------
    def register(self, tool: BaseTool, replace: bool = True) -> None:
        """注册一个工具。"""
        with self._lock:
            if tool.name in self._tools and not replace:
                raise ValueError(f"工具 {tool.name} 已存在")
            self._tools[tool.name] = tool
        logger.info("工具已注册：%s（%s）", tool.name, tool.description[:40])

    def unregister(self, name: str) -> bool:
        """移除工具，返回是否确实移除了。"""
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[BaseTool]:
        """按名称（或中文别名）获取工具。"""
        if not name:
            return None
        key = name.strip()
        with self._lock:
            if key in self._tools:
                return self._tools[key]
            alias = TOOL_ALIASES.get(key)
            if alias and alias in self._tools:
                return self._tools[alias]
            # 宽松匹配：兼容大小写与空格差异
            lowered = key.lower().replace(" ", "_")
            for tool_name, tool in self._tools.items():
                if tool_name.lower() == lowered:
                    return tool
        return None

    def names(self) -> List[str]:
        """全部工具名。"""
        with self._lock:
            return list(self._tools.keys())

    def all(self) -> List[BaseTool]:
        """全部工具实例。"""
        with self._lock:
            return list(self._tools.values())

    def enabled_tools(self) -> List[BaseTool]:
        """当前启用的工具。"""
        return [tool for tool in self.all() if tool.enabled]

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """启用/禁用工具，返回是否找到该工具。"""
        tool = self.get(name)
        if tool is None:
            return False
        tool.enabled = enabled
        logger.info("工具 %s 已%s", tool.name, "启用" if enabled else "禁用")
        return True

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------
    def execute(self, name: str, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolResult:
        """执行指定工具。

        Args:
            name: 工具名（支持中文别名）。
            args: 参数字典。
            **kwargs: 也可直接以关键字传参（与 args 合并，kwargs 优先）。

        Returns:
            ``ToolResult``；工具不存在时返回结构化错误而不是抛异常。
        """
        tool = self.get(name)
        if tool is None:
            available = "、".join(self.names()) or "（无）"
            return ToolResult(
                ok=False,
                error=f"未找到工具 {name!r}，可用工具：{available}",
                error_type="tool_not_found",
                display=f"没有名为 {name} 的工具",
            )

        merged: Dict[str, Any] = dict(args or {})
        merged.update(kwargs)
        return tool.run(**merged)

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def catalog(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """工具目录（供 /api/tools 与 Planner 决策使用）。"""
        tools = self.enabled_tools() if enabled_only else self.all()
        return [tool.spec() for tool in tools]

    def openai_tools(self) -> List[Dict[str, Any]]:
        """OpenAI / DeepSeek function calling 格式的工具清单。"""
        return [tool.to_openai_tool() for tool in self.enabled_tools()]

    def describe_for_prompt(self) -> str:
        """把工具目录渲染成提示词片段（Planner 决策用）。

        刻意写得紧凑：把"能力、成本、耗时、是否需确认"都列出来，
        让 Planner 在等价工具之间做成本/延迟权衡，而不是随机挑一个。
        """
        lines: List[str] = []
        for tool in self.enabled_tools():
            params = (tool.parameters or {}).get("properties", {}) or {}
            param_text = "、".join(params.keys()) if params else "无参数"
            lines.append(
                f"- {tool.name}：{tool.description}\n"
                f"    参数：{param_text}｜成本：{tool.est_cost}｜预期耗时：{tool.est_latency}"
                + ("｜需要人工确认" if tool.requires_confirmation else "")
            )
        return "\n".join(lines) if lines else "（当前没有可用工具）"

    def stats(self) -> List[Dict[str, Any]]:
        """各工具调用统计（供 /api/metrics 与 /health）。"""
        return [tool.stats() for tool in self.all()]


_registry_singleton: Optional[ToolRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ToolRegistry:
    """获取全局工具注册表单例。"""
    global _registry_singleton
    if _registry_singleton is None:
        with _registry_lock:
            if _registry_singleton is None:
                _registry_singleton = ToolRegistry()
    return _registry_singleton


def reset_registry() -> None:
    """清空注册表（测试用）。"""
    global _registry_singleton
    with _registry_lock:
        _registry_singleton = None


def bootstrap_tools(force: bool = False) -> ToolRegistry:
    """注册平台内置工具（幂等）。

    Args:
        force: 为 True 时重建注册表（测试或配置变更后使用）。
    """
    if force:
        reset_registry()
    registry = get_registry()
    if registry.names():
        return registry

    # 延迟导入：避免 tools 子包在导入期就拉起 RAG / 旅行规划子系统
    from core.tools.email_tool import SendEmailTool
    from core.tools.knowledge_tool import KnowledgeSearchTool
    from core.tools.trip_tool import TripPlannerTool

    registry.register(KnowledgeSearchTool())
    registry.register(TripPlannerTool())
    # 需要人工确认的写操作工具（演示 human-in-the-loop）
    registry.register(SendEmailTool())
    logger.info("内置工具注册完成：%s", "、".join(registry.names()))
    return registry


__all__ = [
    "TOOL_ALIASES",
    "ToolRegistry",
    "bootstrap_tools",
    "get_registry",
    "reset_registry",
]
