"""工具协议（B4 核心之一）。

本模块把「Agent 能用的能力」抽象成统一接口，是整个平台可插拔性的基础：

* 新增一个工具 = 继承 ``BaseTool`` + 实现 ``_run()`` + 声明 JSON Schema + 注册；
* 超时、重试、参数校验、人工确认、耗时统计等治理策略**统一在基类里**实现，
  业务代码只关心自己的能力逻辑，不会各写一套。

一个工具需要声明
----------------
======================  ==========================================================
字段                     作用
======================  ==========================================================
``name``                唯一标识（Planner 决策与 trace 都用它）
``description``         能力说明——**这是路由准确率的关键**，要写清"什么时候该用它"
``parameters``          JSON Schema（会原样喂给大模型做 function calling）
``timeout_s``           超时上限，超时即熔断并标记降级
``est_cost``            成本档位 low/medium/high，让 Planner 在等价工具间做选择
``est_latency``         预期耗时档位，同上
``retryable``           失败是否可重试（有副作用的工具必须为 False）
``requires_confirmation`` 危险操作标记：执行前挂起，等人工确认
======================  ==========================================================

统一返回 ``ToolResult``：区分「给大模型看的 data」「给前端看的 display」「给人看的 error」，
这样上层不必猜测工具返回结构，也避免了把大段 JSON 塞进给用户的回答里。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 错误类型（与架构文档第 13 节的降级矩阵对应）
# ---------------------------------------------------------------------------
class ToolErrorType:
    """工具错误分类常量。"""

    INVALID_ARGS = "tool_invalid_args"
    TIMEOUT = "tool_timeout"
    NOT_FOUND = "tool_not_found"
    DISABLED = "tool_disabled"
    DECLINED = "tool_declined"
    INTERNAL = "tool_error"
    DEPENDENCY = "tool_dependency_error"


# ---------------------------------------------------------------------------
# 统一返回结构
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    """工具执行结果。"""

    ok: bool
    data: Any = None
    display: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    degraded: bool = False
    latency_ms: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的字典。"""
        return {
            "ok": self.ok,
            "data": self.data,
            "display": self.display,
            "error": self.error,
            "error_type": self.error_type,
            "degraded": self.degraded,
            "latency_ms": self.latency_ms,
            "meta": self.meta,
        }

    def to_llm_text(self) -> str:
        """转成喂给大模型的紧凑文本（控制上下文长度，避免把完整 JSON 塞进去）。"""
        if not self.ok:
            return f"工具执行失败（{self.error_type or 'unknown'}）：{self.error}"
        if self.display:
            return self.display
        return str(self.data)[:2000]


# ---------------------------------------------------------------------------
# 轻量 JSON Schema 校验
# ---------------------------------------------------------------------------
def validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """按 JSON Schema 校验并补全参数（支持 required / type / default / enum）。

    为什么自己写而不引 jsonschema：本项目的工具参数结构简单（扁平字段 + 枚举），
    自实现约 40 行、零依赖、报错信息是中文且能直接回灌给模型改进参数；
    引入 jsonschema 会多一个依赖，错误信息还需要翻译一遍。

    不支持嵌套对象/数组的深层校验——工具的 ``_run`` 里会再兜一层。

    Returns:
        补全默认值后的参数字典。

    Raises:
        ValueError: 缺少必填参数、类型不符或枚举取值非法。
    """
    schema = schema or {}
    properties: Dict[str, Any] = schema.get("properties", {}) or {}
    required: List[str] = list(schema.get("required", []) or [])
    merged: Dict[str, Any] = dict(args or {})

    # 1) 补默认值
    for key, spec in properties.items():
        if key not in merged and "default" in spec:
            merged[key] = spec["default"]

    # 2) 必填校验（None 与空字符串都视为未提供）
    missing = [
        key for key in required
        if merged.get(key) is None or (isinstance(merged.get(key), str) and not merged[key].strip())
    ]
    if missing:
        raise ValueError(f"缺少必填参数：{', '.join(missing)}")

    # 3) 类型与枚举校验
    type_map = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
    }
    for key, value in list(merged.items()):
        spec = properties.get(key)
        if spec is None or value is None:
            continue

        expected = spec.get("type")
        if expected in type_map:
            allowed = type_map[expected]
            # bool 是 int 的子类，需要单独排除，否则 True 会被当成合法整数
            if expected in ("integer", "number") and isinstance(value, bool):
                raise ValueError(f"参数 {key} 期望 {expected}，收到 boolean")
            if not isinstance(value, allowed):
                # 允许"数字字符串 → 数字"的宽松转换（模型经常把 3 写成 "3"）
                if expected in ("integer", "number") and isinstance(value, str):
                    try:
                        merged[key] = int(value) if expected == "integer" else float(value)
                        value = merged[key]
                    except ValueError as exc:
                        raise ValueError(f"参数 {key} 期望 {expected}，无法从 {value!r} 转换") from exc
                else:
                    raise ValueError(f"参数 {key} 期望 {expected}，收到 {type(value).__name__}")

        enum_values = spec.get("enum")
        if enum_values and value not in enum_values:
            raise ValueError(f"参数 {key} 必须是 {enum_values} 之一，收到 {value!r}")

        maximum = spec.get("maximum")
        if maximum is not None and isinstance(value, (int, float)) and value > maximum:
            raise ValueError(f"参数 {key} 不能大于 {maximum}，收到 {value}")
        minimum = spec.get("minimum")
        if minimum is not None and isinstance(value, (int, float)) and value < minimum:
            raise ValueError(f"参数 {key} 不能小于 {minimum}，收到 {value}")

    return merged


# ---------------------------------------------------------------------------
# 工具基类
# ---------------------------------------------------------------------------
class BaseTool(ABC):
    """所有工具的基类（模板方法模式）。"""

    name: str = "base_tool"
    description: str = "基础工具"
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    timeout_s: float = 30.0
    est_cost: str = "medium"          # low | medium | high
    est_latency: str = "medium"       # low | medium | high
    retryable: bool = True
    requires_confirmation: bool = False

    def __init__(self) -> None:
        self.enabled = True
        self.call_count = 0
        self.error_count = 0
        self.total_latency_ms = 0

    # ------------------------------------------------------------------
    # 对外接口（模板方法：计时 + 异常包装 + 超时熔断）
    # ------------------------------------------------------------------
    def run(self, **kwargs: Any) -> ToolResult:
        """执行工具（带参数校验、超时熔断与异常包装）。

        任何情况下都返回 ``ToolResult``，不向上抛异常——这样 Planner
        永远能拿到可解释的结果并决定下一步，而不是让整个请求 500。
        """
        if not self.enabled:
            return ToolResult(
                ok=False, error=f"工具 {self.name} 当前已禁用",
                error_type=ToolErrorType.DISABLED,
            )

        started = time.perf_counter()

        # ---- 参数校验（失败可回灌给模型改参数）----
        try:
            validated = validate_args(self.parameters, kwargs)
        except ValueError as exc:
            return self._finish(
                ToolResult(
                    ok=False, error=str(exc), error_type=ToolErrorType.INVALID_ARGS,
                    display=f"参数不合法：{exc}",
                ),
                started,
            )

        # ---- 超时熔断：把实际执行丢到线程里，主线程只等 timeout_s ----
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{self.name}") as pool:
                future = pool.submit(self._run, **validated)
                result = future.result(timeout=self.timeout_s)
        except FutureTimeoutError:
            logger.warning("工具 %s 执行超时（>%ss）", self.name, self.timeout_s)
            return self._finish(
                ToolResult(
                    ok=False,
                    error=f"工具执行超时（超过 {self.timeout_s} 秒）",
                    error_type=ToolErrorType.TIMEOUT,
                    degraded=True,
                    display=f"{self.name} 执行超时，已放弃该工具",
                ),
                started,
            )
        except Exception as exc:  # noqa: BLE001 - 工具内部任何异常都要被包装
            logger.exception("工具 %s 执行异常：%s", self.name, exc)
            return self._finish(
                ToolResult(
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    error_type=ToolErrorType.INTERNAL,
                    display=f"{self.name} 执行失败：{exc}",
                ),
                started,
            )

        if not isinstance(result, ToolResult):
            result = ToolResult(ok=True, data=result, display=str(result)[:500])
        return self._finish(result, started)

    def _finish(self, result: ToolResult, started: float) -> ToolResult:
        """补齐耗时并累计统计。"""
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        self.call_count += 1
        self.total_latency_ms += result.latency_ms
        if not result.ok:
            self.error_count += 1
        return result

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------
    @abstractmethod
    def _run(self, **kwargs: Any) -> ToolResult:
        """工具的真实逻辑（参数已完成校验与默认值补全）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def spec(self) -> Dict[str, Any]:
        """工具说明（供 /api/tools 与 Planner 的工具目录使用）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "timeout_s": self.timeout_s,
            "est_cost": self.est_cost,
            "est_latency": self.est_latency,
            "retryable": self.retryable,
            "requires_confirmation": self.requires_confirmation,
            "enabled": self.enabled,
        }

    def to_openai_tool(self) -> Dict[str, Any]:
        """转成 OpenAI / DeepSeek 的 function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def stats(self) -> Dict[str, Any]:
        """调用统计。"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "calls": self.call_count,
            "errors": self.error_count,
            "avg_latency_ms": round(self.total_latency_ms / self.call_count, 1) if self.call_count else 0.0,
        }


__all__ = ["BaseTool", "ToolErrorType", "ToolResult", "validate_args"]
