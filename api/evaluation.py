"""评测接口（B6）。

**模块名为什么叫 ``evaluation`` 而不是 ``eval``**：项目根目录下有一个顶层包
``eval/``（评测框架）。Python 里 ``api.eval`` 与顶层 ``eval`` 同名会互相干扰——
子模块注册为 ``sys.modules["api.eval"]``，而顶层包是 ``sys.modules["eval"]``，
混用时导入可能解析到错误的模块，表现为评测接口**静默返回 0 条数据**（实际踩过）。
换个名字是最简单可靠的解法。

接口
----
``GET /api/eval/datasets``  评测集概况（规模、题型分布、能对上哪些样例文档）
``GET /api/eval/reports``   历史评测报告列表（含核心指标，便于做趋势对比）
``GET /api/eval/reports/{name}``  读取某份报告的完整 markdown

**为什么评测用 CLI 而不是 HTTP 接口触发**：一次完整评测要跑几十次检索与
（可能的）大模型调用，耗时几分钟且会占用大量资源，放在 HTTP 请求里做同步等待
很容易超时、也不便于查看过程日志。因此评测入口是
``python -m eval.run_eval``（可在 CI 里跑），HTTP 侧只负责**读取结果**。
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

# 项目根目录：本文件位于 <root>/api/evaluation.py，因此 parents[1] 是 <root>。
# 注意：``Path(__file__).parents`` 是从**文件所在目录**往上数，
# 不是从包目录往上数——这里踩过一次（写成 parents[2] 会指到项目外面去，
# 表现为接口正常返回但数据集规模恒为 0）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "eval" / "datasets"
REPORT_DIR = PROJECT_ROOT / "eval" / "reports"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["评测"])


def _load_eval_module(name: str) -> Any:
    """按绝对路径导入顶层 ``eval`` 包里的模块。

    为什么不用 ``from eval.dataset import ...``：本模块位于 ``api`` 包内，
    直接写 ``eval`` 有被解析成 ``api.eval`` 的风险（命名冲突）。
    用 ``importlib.import_module`` 并保证项目根在 ``sys.path`` 上，语义最明确。
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return importlib.import_module(name)


@router.get("/datasets", summary="评测集概况")
async def dataset_overview() -> Dict[str, Any]:
    """返回评测集规模与题型分布。"""
    try:
        dataset_module = _load_eval_module("eval.dataset")
    except Exception as exc:  # noqa: BLE001 - 评测包缺失不应让接口 500
        logger.warning("加载评测模块失败：%s", exc)
        return {"ok": False, "error": f"评测模块不可用：{exc}", "qa": {}, "routing": {}}

    load_qa_cases = dataset_module.load_qa_cases
    load_routing_cases = dataset_module.load_routing_cases

    qa_path = DATASET_DIR / "rag_qa.jsonl"
    routing_path = DATASET_DIR / "routing.jsonl"
    qa_cases = load_qa_cases(qa_path) if qa_path.exists() else []
    routing_cases = load_routing_cases(routing_path) if routing_path.exists() else []

    answerable = [case for case in qa_cases if not case.is_unanswerable]
    unanswerable = [case for case in qa_cases if case.is_unanswerable]
    tools: Dict[str, int] = {}
    for case in routing_cases:
        tools[case.expected_tool] = tools.get(case.expected_tool, 0) + 1

    return {
        "ok": True,
        "qa": {
            "total": len(qa_cases),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "sources": sorted({gold for case in qa_cases for gold in case.gold_sources}),
        },
        "routing": {"total": len(routing_cases), "by_tool": tools},
    }


@router.get("/reports", summary="历史评测报告")
async def list_reports(limit: int = 20) -> Dict[str, Any]:
    """列出历史评测报告（按时间倒序），附核心指标。"""
    if not REPORT_DIR.exists():
        return {"ok": True, "total": 0, "reports": []}

    reports: List[Dict[str, Any]] = []
    for path in sorted(REPORT_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("读取报告失败 %s：%s", path.name, exc)
            continue
        qa_metrics = payload.get("qa") or {}
        routing_metrics = payload.get("routing") or {}
        reports.append(
            {
                "name": path.stem,
                "config": payload.get("config"),
                "description": payload.get("description"),
                "generated_at": payload.get("generated_at"),
                "duration_s": payload.get("duration_s"),
                "top_k": payload.get("top_k"),
                "metrics": {
                    "recall": qa_metrics.get("recall"),
                    "mrr": qa_metrics.get("mrr"),
                    "refusal_accuracy": qa_metrics.get("refusal_accuracy"),
                    "keyword_coverage": qa_metrics.get("keyword_coverage"),
                    "llm_judge_accuracy": qa_metrics.get("llm_judge_accuracy"),
                    "route_accuracy": routing_metrics.get("route_accuracy"),
                    "p50_ms": (qa_metrics.get("latency_ms") or {}).get("p50"),
                    "p95_ms": (qa_metrics.get("latency_ms") or {}).get("p95"),
                    "avg_tokens": (qa_metrics.get("tokens") or {}).get("avg_per_case"),
                },
            }
        )
    return {"ok": True, "total": len(reports), "reports": reports}


@router.get("/reports/{name}", summary="读取报告全文")
async def get_report(name: str) -> Dict[str, Any]:
    """返回某份报告的 markdown 全文（name 为文件名去掉扩展名）。"""
    safe_name = Path(name).name          # 防止路径穿越
    markdown_path = REPORT_DIR / f"{safe_name}.md"
    json_path = REPORT_DIR / f"{safe_name}.json"
    if not markdown_path.exists():
        raise HTTPException(status_code=404, detail=f"报告不存在：{name}")

    payload: Dict[str, Any] = {}
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    return {
        "ok": True,
        "name": safe_name,
        "markdown": markdown_path.read_text(encoding="utf-8"),
        "metrics": {"qa": payload.get("qa"), "routing": payload.get("routing")},
    }


__all__ = ["router"]
