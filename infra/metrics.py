"""指标聚合。

把散落在各处的计数（LLM 用量、工具调用耗时、拒答次数、请求耗时）收敛到一处，
供 ``/health``、``/api/metrics`` 与评测报告读取。

设计取舍：**只用进程内计数器，不引入 Prometheus**。
单机演示与校园/个人项目场景下，暴露一个 JSON 汇总接口比接入监控栈更实用；
真要接 Prometheus，把 ``snapshot()`` 换成 Gauge 注册即可，接口不变。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

_lock = threading.Lock()

# 进程内累计指标
_counters: Dict[str, float] = {
    "chat_requests": 0,
    "chat_errors": 0,
    "refusals": 0,
    "tool_calls": 0,
    "tool_errors": 0,
    "tool_timeouts": 0,
    "llm_calls": 0,
    "llm_failures": 0,
    "degraded_runs": 0,
    "tokens_total": 0,
    "cost_total": 0.0,
    "latency_ms_total": 0,
}

# 各工具调用次数与耗时（用于发现"哪个工具最慢"）
_tool_stats: Dict[str, Dict[str, float]] = {}

# 最近的请求延迟样本（用于算 P50/P95，只保留最近 500 条）
_latency_samples: List[int] = []
_MAX_SAMPLES = 500


def incr(name: str, value: float = 1) -> None:
    """累加一个计数器。"""
    with _lock:
        _counters[name] = _counters.get(name, 0) + value


def observe_tool(tool_name: str, latency_ms: int, ok: bool, degraded: bool = False) -> None:
    """记录一次工具调用。"""
    with _lock:
        _counters["tool_calls"] += 1
        if not ok:
            _counters["tool_errors"] += 1
        if degraded:
            _counters["degraded_runs"] += 1
        stat = _tool_stats.setdefault(
            tool_name, {"calls": 0, "errors": 0, "latency_ms_total": 0, "max_ms": 0}
        )
        stat["calls"] += 1
        stat["latency_ms_total"] += latency_ms
        stat["max_ms"] = max(stat["max_ms"], latency_ms)
        if not ok:
            stat["errors"] += 1


def observe_latency(latency_ms: int) -> None:
    """记录一次端到端请求延迟样本。"""
    with _lock:
        _counters["latency_ms_total"] += latency_ms
        _latency_samples.append(latency_ms)
        if len(_latency_samples) > _MAX_SAMPLES:
            del _latency_samples[: len(_latency_samples) - _MAX_SAMPLES]


def observe_llm(total_tokens: int, cost: float, ok: bool = True) -> None:
    """记录一次大模型调用。"""
    with _lock:
        _counters["llm_calls"] += 1
        if not ok:
            _counters["llm_failures"] += 1
        _counters["tokens_total"] += total_tokens
        _counters["cost_total"] = round(_counters["cost_total"] + cost, 6)


def percentile(samples: List[int], ratio: float) -> float:
    """计算分位数（线性插值，样本为空返回 0）。"""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 1)


def snapshot() -> Dict[str, Any]:
    """导出当前指标快照。"""
    with _lock:
        counters = dict(_counters)
        tools = {
            name: {
                "calls": int(stat["calls"]),
                "errors": int(stat["errors"]),
                "avg_ms": round(stat["latency_ms_total"] / stat["calls"], 1) if stat["calls"] else 0.0,
                "max_ms": int(stat["max_ms"]),
            }
            for name, stat in _tool_stats.items()
        }
        samples = list(_latency_samples)

    requests = counters.get("chat_requests", 0) or 0
    return {
        "counters": counters,
        "tools": tools,
        "latency_ms": {
            "p50": percentile(samples, 0.50),
            "p95": percentile(samples, 0.95),
            "avg": round(counters["latency_ms_total"] / requests, 1) if requests else 0.0,
            "samples": len(samples),
        },
    }


def reset() -> None:
    """重置所有指标（测试用）。"""
    with _lock:
        for key in _counters:
            _counters[key] = 0.0
        _tool_stats.clear()
        _latency_samples.clear()
