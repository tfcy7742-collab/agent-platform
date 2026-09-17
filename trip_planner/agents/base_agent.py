"""智能体基类。

本模块把「每个智能体是一个独立 Python 类」这一设计要求落地：

* ``BaseAgent`` 封装了与 **DeepSeek（OpenAI 兼容接口）** 的全部交互细节；
* 子类只需要提供：名称、职责描述、系统提示词、可调用的工具列表、以及
  一个 ``run()`` 方法（接收上下文，返回结构化结果）；
* 内置 **Function Calling** 循环：LLM 决定调用哪个工具 → 本地执行工具 →
  把结果回灌给 LLM → 直到模型给出最终回答（最多 ``max_tool_rounds`` 轮）；
* 内置 **降级策略**：没有配置 API Key、网络异常、模型返回非法 JSON 时，
  自动切换到本地规则实现，保证接口永远可用、演示永远不会失败。

配置读取优先级（见 ``backend/config.py``）：
    DEEPSEEK_API_KEY > OPENAI_API_KEY
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from trip_planner.config import get_settings
from trip_planner.models.schemas import AgentTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON 解析工具
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Optional[Any]:
    """从模型输出中稳健地提取 JSON。

    依次尝试：
    1. 直接 ``json.loads``；
    2. 去掉 ```json ... ``` 代码围栏后解析；
    3. 截取第一个 ``{``/``[`` 到最后一个 ``}``/``]`` 之间的内容解析。

    Returns:
        解析成功返回 Python 对象，失败返回 ``None``（由调用方走降级逻辑）。
    """
    if not text:
        return None

    candidates: List[str] = [text.strip()]

    # 去掉 Markdown 代码围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())

    # 截取最外层花括号 / 方括号
    for left, right in (("{", "}"), ("[", "]")):
        start = text.find(left)
        end = text.rfind(right)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    # 兜底：尝试修复中文全角引号造成的非法 JSON
    try:
        fixed = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        start, end = fixed.find("{"), fixed.rfind("}")
        if start != -1 and end > start:
            return json.loads(fixed[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        pass

    logger.warning("无法从模型输出中解析 JSON：%s", text[:200])
    return None


# ---------------------------------------------------------------------------
# 智能体基类
# ---------------------------------------------------------------------------
class BaseAgent:
    """所有智能体的基类。

    Attributes:
        name: 智能体名称（中文，展示用）。
        role: 智能体职责一句话描述。
        system_prompt: 系统提示词。
        tools: 可调用工具名列表（对应 ``TOOL_REGISTRY`` 中的键）。
        temperature: 采样温度。
        use_json_mode: 是否要求模型直接输出 JSON（DeepSeek 支持 response_format）。
        enable_llm: 是否允许调用大模型；False 时直接走本地规则。
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        tools: Optional[List[str]] = None,
        temperature: float = 0.3,
        use_json_mode: bool = False,
        enable_llm: bool = True,
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.tools: List[str] = tools or []
        self.temperature = temperature
        self.use_json_mode = use_json_mode
        self.enable_llm = enable_llm

        self.settings = get_settings()
        self._client: Optional[Any] = None          # ChatOpenAI 实例（懒加载）
        self._client_error: Optional[str] = None    # 初始化失败原因
        self.last_trace: Optional[AgentTrace] = None  # 最近一次执行轨迹

    # ------------------------------------------------------------------
    # LLM 客户端
    # ------------------------------------------------------------------
    @property
    def llm_available(self) -> bool:
        """当前是否具备调用 DeepSeek 的条件。"""
        return bool(self.enable_llm and self.settings.has_api_key)

    def _get_client(self) -> Optional[Any]:
        """懒加载 ChatOpenAI 客户端（指向 DeepSeek 的 OpenAI 兼容端点）。"""
        if not self.llm_available:
            return None
        if self._client is not None or self._client_error is not None:
            return self._client

        try:
            # 延迟导入：即使未安装 langchain-openai，也能以降级模式运行
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.settings.model,
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                temperature=self.temperature,
                timeout=self.settings.timeout,
                max_retries=self.settings.max_retries,
            )
            logger.info("[%s] 已连接 DeepSeek：%s (%s)", self.name, self.settings.model, self.settings.base_url)
        except Exception as exc:  # pragma: no cover - 依赖缺失或配置异常
            self._client_error = f"{type(exc).__name__}: {exc}"
            logger.warning("[%s] 初始化 DeepSeek 客户端失败，将使用本地降级逻辑：%s", self.name, exc)
        return self._client

    # ------------------------------------------------------------------
    # 基础调用能力
    # ------------------------------------------------------------------
    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> Optional[str]:
        """单轮文本对话。

        Args:
            user_prompt: 用户消息。
            system_prompt: 覆盖默认系统提示词。
            json_mode: 是否强制 JSON 输出；默认取 ``self.use_json_mode``。

        Returns:
            模型文本；失败返回 ``None``（调用方需自行降级）。
        """
        client = self._get_client()
        if client is None:
            return None

        use_json = self.use_json_mode if json_mode is None else json_mode
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if use_json:
                # DeepSeek 兼容 OpenAI 的 response_format={"type": "json_object"}
                response = client.invoke(messages, response_format={"type": "json_object"})
            else:
                response = client.invoke(messages)
            content = getattr(response, "content", None)
            if isinstance(content, list):  # 部分版本会返回分段内容
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                )
            return content or None
        except Exception as exc:
            logger.warning("[%s] 调用 DeepSeek 失败：%s", self.name, exc)
            return None

    def chat_json(self, user_prompt: str, system_prompt: Optional[str] = None) -> Optional[Any]:
        """要求模型返回 JSON 并解析成 Python 对象。"""
        text = self.chat(user_prompt, system_prompt=system_prompt, json_mode=True)
        if not text:
            return None
        return extract_json(text)

    # ------------------------------------------------------------------
    # Function Calling 循环
    # ------------------------------------------------------------------
    def run_with_tools(
        self,
        user_prompt: str,
        tool_impls: Optional[Dict[str, Callable[..., Any]]] = None,
        max_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """带工具调用的对话循环（DeepSeek Function Calling）。

        Args:
            user_prompt: 用户消息。
            tool_impls: 工具名 -> 可调用函数；默认使用 ``TOOL_REGISTRY``。
            max_rounds: 最大工具调用轮数，默认取配置值。

        Returns:
            ``{"content": 最终文本, "tool_calls": [调用记录], "used_llm": bool}``
        """
        from trip_planner.tools.travel_tools import TOOL_REGISTRY, TOOL_SPECS

        tool_impls = tool_impls or TOOL_REGISTRY
        max_rounds = max_rounds or self.settings.max_tool_rounds
        call_log: List[Dict[str, Any]] = []

        client = self._get_client()
        if client is None:
            return {"content": None, "tool_calls": call_log, "used_llm": False}

        # 只绑定本智能体声明可用的工具
        specs = [TOOL_SPECS[name] for name in self.tools if name in TOOL_SPECS]
        bound = client.bind_tools(specs) if specs else client

        messages: List[Any] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            for _ in range(max_rounds):
                ai_msg = bound.invoke(messages)
                messages.append(ai_msg)
                tool_calls = getattr(ai_msg, "tool_calls", None) or []
                if not tool_calls:
                    content = getattr(ai_msg, "content", "") or ""
                    return {"content": content, "tool_calls": call_log, "used_llm": True}

                for call in tool_calls:
                    tool_name = call.get("name")
                    tool_args = call.get("args") or {}
                    impl = tool_impls.get(tool_name)
                    if impl is None:
                        result: Any = {"error": f"未知工具：{tool_name}"}
                    else:
                        try:
                            result = impl(**tool_args)
                        except Exception as exc:  # 工具自身异常也要回灌给模型
                            result = {"error": f"{type(exc).__name__}: {exc}"}
                    call_log.append({"tool": tool_name, "args": tool_args})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", tool_name),
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            # 超出轮数上限：要求模型直接总结
            messages.append({"role": "user", "content": "请基于以上工具结果直接给出最终答案。"})
            final = bound.invoke(messages)
            return {
                "content": getattr(final, "content", "") or "",
                "tool_calls": call_log,
                "used_llm": True,
            }
        except Exception as exc:
            logger.warning("[%s] Function Calling 循环失败：%s", self.name, exc)
            return {"content": None, "tool_calls": call_log, "used_llm": False}

    # ------------------------------------------------------------------
    # 轨迹记录
    # ------------------------------------------------------------------
    def make_trace(
        self,
        status: str,
        duration_ms: int,
        tools: Optional[List[str]] = None,
        summary: str = "",
        error: Optional[str] = None,
    ) -> AgentTrace:
        """构造并保存一条执行轨迹。"""
        trace = AgentTrace(
            agent=self.name,
            role=self.role,
            status=status,
            duration_ms=duration_ms,
            tools=tools or [],
            summary=summary,
            error=error,
        )
        self.last_trace = trace
        return trace

    # ------------------------------------------------------------------
    # 子类必须实现的接口
    # ------------------------------------------------------------------
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """执行智能体任务（由子类实现）。"""
        raise NotImplementedError("子类必须实现 run() 方法")
