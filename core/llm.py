"""LLM 工厂。

职责
----
1. 按配置创建指向 **DeepSeek**（或阿里云百炼）的 ``ChatOpenAI`` 客户端；
2. 统一超时、重试、温度等参数，避免各处重复配置；
3. 提供 ``LLMResult`` 统一返回：文本、token 用量、耗时、是否降级、错误分类，
   让上层（Planner / Responder）不必关心底层异常细节；
4. 支持 **离线模式**：``settings.llm_online`` 为 False 时不创建客户端，
   调用方拿到 ``ok=False, error_type="llm_offline"`` 并走各自的规则降级路径。

错误分类（与架构文档第 13 节一致）
----------------------------------
``llm_timeout`` / ``llm_auth`` / ``llm_rate_limit`` / ``llm_bad_output`` / ``llm_offline`` / ``llm_error``
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# 粗略成本估算（元 / 千 token），仅用于成本看板展示，非精确计费
COST_PER_1K_PROMPT = 0.001
COST_PER_1K_COMPLETION = 0.002


# ---------------------------------------------------------------------------
# 统一返回结构
# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    """一次 LLM 调用的结果。"""

    ok: bool
    text: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_est: float = 0.0
    raw: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        """是否属于降级结果（调用方据此在 trace 上打标）。"""
        return not self.ok


def classify_error(exc: BaseException) -> Tuple[str, str]:
    """把底层异常分类成稳定的 ``error_type``，便于上层做差异化降级。

    Returns:
        ``(error_type, 可读错误信息)``
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()

    if "timeout" in lowered or "timed out" in lowered:
        return "llm_timeout", "大模型请求超时"
    if "401" in text or "authentication" in lowered or "invalid_api_key" in lowered or "incorrect api key" in lowered:
        return "llm_auth", "大模型鉴权失败（API Key 无效或欠费）"
    if "429" in text or "rate limit" in lowered or "quota" in lowered:
        return "llm_rate_limit", "大模型限流或额度不足"
    if "connection" in lowered or "connect" in lowered or "dns" in lowered or "ssl" in lowered:
        return "llm_error", "无法连接大模型服务（检查网络或 BASE_URL）"
    if "json" in lowered:
        return "llm_bad_output", "大模型输出不是合法 JSON"
    return "llm_error", f"大模型调用异常：{text[:200]}"


def extract_json(text: str) -> Optional[Any]:
    """从模型输出中稳健地提取 JSON 对象。

    依次尝试：直接解析 → 去掉 ```json 围栏 → 截取最外层花括号 → 修复中文全角引号后重试。
    这些容错在真实项目里非常必要：模型经常在 JSON 前后加一句"好的，以下是结果"。
    """
    import json

    if not text:
        return None

    candidates: List[str] = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    for left, right in (("{", "}"), ("[", "]")):
        start, end = text.find(left), text.rfind(right)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    # 兜底：修复中文全角标点造成的非法 JSON。
    # 这一步很有必要——模型（尤其中文模型）经常输出 {"问题"："答案"，"分数"：0.8}
    # 这类"看起来像 JSON 但标点是全角"的文本，标准 json.loads 必然失败。
    # 注意：必须先替换引号（否则引号与冒号交替替换会互相破坏），再替换冒号与逗号。
    fixed = text
    for source, target in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("：", ":"), ("，", ",")):
        fixed = fixed.replace(source, target)
    start, end = fixed.find("{"), fixed.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(fixed[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# LLM 客户端封装
# ---------------------------------------------------------------------------
class LLMClient:
    """LLM 客户端（进程内单例使用，见 ``get_llm_client``）。

    Args:
        settings: 配置；默认取全局配置。
        enable: 是否强制禁用（用于测试时完全不触碰网络）。
    """

    def __init__(self, settings: Optional[Settings] = None, enable: bool = True) -> None:
        self.settings = settings or get_settings()
        self.enable = enable and self.settings.llm_online
        self._client: Any = None
        self._init_error: Optional[str] = None
        self._init_error_type: Optional[str] = None
        # 累计用量，供 /health 与成本看板读取
        self.usage: Dict[str, Any] = {
            "calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_est": 0.0,
            "total_latency_ms": 0,
        }

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """当前是否具备真实调用能力（配置齐全且模型可用）。"""
        return bool(self.enable and self.settings.has_api_key) and self._init_error is None

    def describe(self) -> Dict[str, Any]:
        """给 /health 用的描述信息。"""
        return {
            "provider": self.settings.llm_provider,
            "model": self.settings.provider_model,
            "base_url": self.settings.provider_base_url,
            "api_key": self.settings.masked_key,
            "mode": self.settings.llm_mode,
            "available": self.available,
            "init_error": self._init_error,
            "usage": dict(self.usage),
        }

    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        """懒加载 ``ChatOpenAI``（指向 DeepSeek 的 OpenAI 兼容端点）。"""
        if not self.available or self._client is not None:
            return self._client
        try:
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.settings.provider_model,
                api_key=self.settings.provider_api_key,
                base_url=self.settings.provider_base_url,
                temperature=self.settings.llm_temperature,
                timeout=self.settings.llm_timeout,
                max_retries=self.settings.llm_max_retries,
                max_tokens=self.settings.llm_max_tokens,
            )
            logger.info(
                "LLM 客户端就绪：provider=%s model=%s base_url=%s key=%s",
                self.settings.llm_provider,
                self.settings.provider_model,
                self.settings.provider_base_url,
                self.settings.masked_key,
            )
        except Exception as exc:  # pragma: no cover - 依赖缺失等异常环境
            self._init_error_type, self._init_error = classify_error(exc)
            logger.warning("LLM 客户端初始化失败，将走离线降级：%s", exc)
        return self._client

    # ------------------------------------------------------------------
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
    ) -> LLMResult:
        """执行一次对话调用。

        Args:
            system_prompt: 系统提示词。
            user_prompt: 用户消息。
            json_mode: 是否要求模型直接输出 JSON（DeepSeek 支持 response_format）。
            temperature: 覆盖默认温度。

        Returns:
            ``LLMResult``；任何失败都不抛异常，由调用方按 ``error_type`` 降级。
        """
        started = time.perf_counter()

        if not self.enable:
            return LLMResult(
                ok=False,
                error="离线模式：未启用大模型",
                error_type="llm_offline",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        client = self._get_client()
        if client is None:
            return LLMResult(
                ok=False,
                error=self._init_error or "大模型不可用（未配置 API Key）",
                error_type=self._init_error_type or "llm_auth",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: Dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            response = client.invoke(messages, **kwargs)
            text = self._extract_text(response)
            prompt_tokens, completion_tokens = self._extract_usage(response)
            latency_ms = int((time.perf_counter() - started) * 1000)

            cost = (
                prompt_tokens / 1000 * COST_PER_1K_PROMPT
                + completion_tokens / 1000 * COST_PER_1K_COMPLETION
            )
            self._accumulate(prompt_tokens, completion_tokens, cost, latency_ms)

            return LLMResult(
                ok=True,
                text=text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_est=round(cost, 6),
                raw=response,
            )
        except Exception as exc:
            error_type, message = classify_error(exc)
            latency_ms = int((time.perf_counter() - started) * 1000)
            self.usage["failed_calls"] += 1
            logger.warning("LLM 调用失败（%s）：%s", error_type, exc)
            return LLMResult(
                ok=False, error=message, error_type=error_type, latency_ms=latency_ms
            )

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> Tuple[Optional[Any], LLMResult]:
        """调用并解析 JSON，返回 ``(解析结果, LLMResult)``。"""
        result = self.chat(system_prompt, user_prompt, json_mode=True, temperature=temperature)
        if not result.ok:
            return None, result
        parsed = extract_json(result.text)
        if parsed is None:
            result.ok = False
            result.error_type = "llm_bad_output"
            result.error = "大模型输出无法解析为 JSON"
            return None, result
        return parsed, result

    # ------------------------------------------------------------------
    def _accumulate(self, prompt_tokens: int, completion_tokens: int, cost: float, latency_ms: int) -> None:
        """累计用量统计。"""
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens
        self.usage["cost_est"] = round(self.usage["cost_est"] + cost, 6)
        self.usage["total_latency_ms"] += latency_ms

    @staticmethod
    def _extract_text(response: Any) -> str:
        """从 AIMessage 中取出文本（兼容分片内容）。"""
        content = getattr(response, "content", None)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return content or ""

    @staticmethod
    def _extract_usage(response: Any) -> Tuple[int, int]:
        """从响应中提取 token 用量（不同版本字段位置不一致，做多重兜底）。"""
        # 1) LangChain 标准位置
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            prompt = int(usage.get("input_tokens") or 0)
            completion = int(usage.get("output_tokens") or 0)
            if prompt or completion:
                return prompt, completion

        # 2) OpenAI 原始字段
        meta = getattr(response, "response_metadata", None) or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(token_usage, dict):
            prompt = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
            completion = int(
                token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            )
            if prompt or completion:
                return prompt, completion

        # 3) 兜底：按字符数粗估（中文约 1.5 字符/token）
        text = LLMClient._extract_text(response)
        return 0, int(len(text) / 1.5)


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_client_singleton: Optional[LLMClient] = None


def get_llm_client(reload: bool = False) -> LLMClient:
    """获取全局 LLM 客户端单例。

    Args:
        reload: 为 True 时重建客户端（配置变更后调用，例如前端切换 provider）。
    """
    global _client_singleton
    if _client_singleton is None or reload:
        _client_singleton = LLMClient()
    return _client_singleton
