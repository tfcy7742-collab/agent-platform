"""工具目录与执行轨迹接口（B4 交付）。

接口
----
``GET  /api/tools``              工具目录（名称 / 说明 / JSON Schema / 成本 / 是否需确认）
``GET  /api/tools/stats``        各工具调用统计
``POST /api/tools/{name}/toggle`` 启用/禁用工具（演示"运行期治理"）
``POST /api/tools/{name}/invoke`` 直接调用某个工具（调试用，绕过 Planner）
``GET  /api/traces``             最近的执行轨迹列表
``GET  /api/traces/{run_id}``    单次执行的分步明细

``/api/traces`` 是"可观测性"最直接的入口：能看到 Planner 每步为什么这么选、
工具耗时多少、检索命中几条、是否降级。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.runtime.executor import get_executor
from core.tools.registry import get_registry
from infra import db, trace as trace_module

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["工具与轨迹"])


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------
class ToolItem(BaseModel):
    """工具描述。"""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 30.0
    est_cost: str = "medium"
    est_latency: str = "medium"
    retryable: bool = True
    requires_confirmation: bool = False
    enabled: bool = True
    calls: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0


class InvokeRequest(BaseModel):
    """直接调用工具的请求。"""

    args: Dict[str, Any] = Field(default_factory=dict, description="工具参数")


# ---------------------------------------------------------------------------
# 工具相关接口
# ---------------------------------------------------------------------------
@router.get("/tools", summary="工具目录")
async def list_tools(enabled_only: bool = Query(False, description="只返回启用的工具")) -> Dict[str, Any]:
    """返回所有可用工具及其 JSON Schema。

    Schema 会原样用于两处：喂给 Planner 做路由决策、以及前端展示"这个 Agent 能做什么"。
    """
    registry = get_registry()
    executor = get_executor(registry)
    items = []
    for spec in registry.catalog(enabled_only=enabled_only):
        tool = registry.get(spec["name"])
        stats = tool.stats() if tool is not None else {}
        items.append(
            ToolItem(
                **{key: spec[key] for key in (
                    "name", "description", "parameters", "timeout_s", "est_cost",
                    "est_latency", "retryable", "requires_confirmation", "enabled",
                )},
                calls=int(stats.get("calls", 0)),
                errors=int(stats.get("errors", 0)),
                avg_latency_ms=float(stats.get("avg_latency_ms", 0.0)),
            )
        )
    return {
        "ok": True,
        "total": len(items),
        "executor": {"pending_confirmations": executor.pending_count()},
        "tools": [item.model_dump() for item in items],
    }


@router.get("/tools/stats", summary="工具调用统计")
async def tool_stats() -> Dict[str, Any]:
    """各工具的调用次数、错误数与平均耗时。"""
    registry = get_registry()
    return {
        "ok": True,
        "stats": registry.stats(),
        "pending_confirmations": get_executor(registry).pending_count(),
    }


@router.post("/tools/{name}/toggle", summary="启用/禁用工具")
async def toggle_tool(name: str, enabled: bool = Query(..., description="true=启用 false=禁用")) -> Dict[str, Any]:
    """运行期启用或禁用某个工具。

    用途：知识库为空时可以临时禁用 ``knowledge_search``，
    让 Planner 不再把问题路由到没有数据的工具上。
    """
    registry = get_registry()
    if not registry.set_enabled(name, enabled):
        raise HTTPException(status_code=404, detail=f"工具不存在：{name}")
    tool = registry.get(name)
    return {"ok": True, "name": tool.name if tool else name, "enabled": enabled}


@router.post("/tools/{name}/invoke", summary="直接调用工具（调试）")
async def invoke_tool(name: str, request: InvokeRequest) -> Dict[str, Any]:
    """绕过 Planner 直接执行一个工具，便于调试参数与观察返回结构。

    注意：需要人工确认的工具在这里也会走确认闸门。
    """
    from fastapi.concurrency import run_in_threadpool

    executor = get_executor(get_registry())
    outcome = await run_in_threadpool(executor.execute, name, request.args, "", False)
    return {
        "ok": outcome.result.ok,
        "tool": outcome.tool_name,
        "args": outcome.args,
        "attempts": outcome.attempts,
        "retried": outcome.retried,
        "pending_confirmation": outcome.pending_confirmation,
        "resume_token": outcome.resume_token,
        "result": outcome.result.to_dict(),
    }


# ---------------------------------------------------------------------------
# 轨迹接口
# ---------------------------------------------------------------------------
@router.get("/traces", summary="执行轨迹列表")
async def list_traces(
    limit: int = Query(20, ge=1, le=200, description="返回条数"),
    session_id: Optional[str] = Query(None, description="按会话过滤"),
) -> Dict[str, Any]:
    """列出最近的执行轨迹（新的在前）。"""
    items = trace_module.list_traces(limit=limit, session_id=session_id)
    return {
        "ok": True,
        "total": len(items),
        "stats": db.trace_stats(),
        "traces": items,
    }


@router.get("/traces/{run_id}", summary="执行轨迹详情")
async def get_trace(run_id: str) -> Dict[str, Any]:
    """返回单次执行的分步明细：Planner 决策、工具调用、耗时、检索分数、降级原因。"""
    detail = trace_module.get_trace(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"轨迹不存在：{run_id}")
    return {"ok": True, **detail}


__all__ = ["router"]
