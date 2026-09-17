"""评测数据集加载与指标计算。

数据集格式（JSONL，每行一个 JSON 对象）
---------------------------------------
问答集 ``eval/datasets/rag_qa.jsonl``::

    {"id": "qa001", "type": "answerable",
     "question": "年假有几天？",
     "expected_answer": "满一年不满十年 5 天，满十年不满二十年 10 天，满二十年 15 天",
     "keywords": ["年假", "5", "10", "15"],
     "gold_sources": ["员工手册"]}

* ``type``：``answerable``（应能回答并命中来源）/ ``unanswerable``（应拒答）
* ``gold_sources``：期望命中的文件名关键字（做子串匹配，避免因扩展名差异误判）
* ``keywords``：离线判分用的关键字（在线模式下由 LLM judge 判分）

路由集 ``eval/datasets/routing.jsonl``::

    {"id": "rt001", "question": "帮我规划北京三日游", "expected_tool": "trip_planner"}

指标
----
==========================  ==========================================================
指标                         含义
==========================  ==========================================================
``recall@k``                 金标来源是否出现在前 k 条检索结果里
``mrr``                      第一个命中金标来源的倒数排名（衡量排序质量）
``refusal_accuracy``         文档外问题是否正确拒答（必须逐字匹配标准话术）
``route_accuracy``           Planner 是否选对工具
``answer_keyword_score``     答案对关键信息的覆盖度（离线判分，0~1）
``citation_validity``        回答里的引用编号是否都能对应到片段
``p50_ms`` / ``p95_ms``      端到端与各阶段延迟
``avg_tokens`` / ``cost``    平均 token 消耗与估算成本
==========================  ==========================================================

为什么用"关键字覆盖 + 可选 LLM judge"两套判分
---------------------------------------------
LLM judge 更准，但要消耗 token 且需要 Key；关键字覆盖虽然粗，但**免费、离线、可复现**，
适合做 CI 回归。评测脚本默认两套都跑，报告里并列展示，并在方法说明里写清差异。
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class QaCase:
    """一条问答评测用例。"""

    id: str
    question: str
    expected_answer: str = ""
    keywords: List[str] = field(default_factory=list)
    gold_sources: List[str] = field(default_factory=list)
    type: str = "answerable"          # answerable | unanswerable
    note: str = ""

    @property
    def is_unanswerable(self) -> bool:
        """是否属于"资料中无相关内容"的用例。"""
        return self.type == "unanswerable"


@dataclass
class RoutingCase:
    """一条路由评测用例。"""

    id: str
    question: str
    expected_tool: str
    note: str = ""


@dataclass
class CaseResult:
    """单条用例的执行结果与判分。"""

    case_id: str
    question: str
    # 检索侧
    hits: List[Dict[str, Any]] = field(default_factory=list)
    recall_hit: bool = False
    reciprocal_rank: float = 0.0
    top_score: float = 0.0
    # 回答侧
    answer: str = ""
    refused: bool = False
    refusal_correct: bool = False
    answer_keyword_score: float = 0.0
    citation_valid: bool = True
    llm_judge: Optional[bool] = None
    llm_judge_reason: str = ""
    # 路由侧
    tools_used: List[str] = field(default_factory=list)
    route_correct: Optional[bool] = None
    # 该用例的金标来源与题型（由 attach_gold 回填）
    # 题型用于区分"可答"与"应拒答"——**拒答准确率只能在应拒答用例上算**，
    # 否则一条正常的可答用例会把分母撑大、把指标稀释。
    gold_sources: List[str] = field(default_factory=list)
    case_type: str = "answerable"          # answerable | unanswerable | routing
    # 性能与成本
    total_ms: int = 0
    retrieval_ms: int = 0
    llm_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_est: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转成可序列化字典（报告里逐条展示）。"""
        return {
            "id": self.case_id,
            "question": self.question,
            "hits": [hit.get("file_name") for hit in self.hits],
            "top_score": round(self.top_score, 4),
            "recall_hit": self.recall_hit,
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "answer": self.answer[:200],
            "refused": self.refused,
            "refusal_correct": self.refusal_correct,
            "answer_keyword_score": round(self.answer_keyword_score, 3),
            "citation_valid": self.citation_valid,
            "llm_judge": self.llm_judge,
            "llm_judge_reason": self.llm_judge_reason[:120],
            "tools_used": self.tools_used,
            "route_correct": self.route_correct,
            "total_ms": self.total_ms,
            "retrieval_ms": self.retrieval_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_est": round(self.cost_est, 6),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件（忽略空行与以 # 开头的注释行）。"""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {line_number} 行不是合法 JSON：{exc}") from exc
    return records


def load_qa_cases(path: Path) -> List[QaCase]:
    """加载问答评测集。"""
    cases: List[QaCase] = []
    for index, record in enumerate(load_jsonl(path), start=1):
        cases.append(
            QaCase(
                id=str(record.get("id") or f"qa{index:03d}"),
                question=str(record["question"]),
                expected_answer=str(record.get("expected_answer") or ""),
                keywords=[str(item) for item in record.get("keywords") or []],
                gold_sources=[str(item) for item in record.get("gold_sources") or []],
                type=str(record.get("type") or "answerable"),
                note=str(record.get("note") or ""),
            )
        )
    return cases


def load_routing_cases(path: Path) -> List[RoutingCase]:
    """加载路由评测集。"""
    cases: List[RoutingCase] = []
    for index, record in enumerate(load_jsonl(path), start=1):
        cases.append(
            RoutingCase(
                id=str(record.get("id") or f"rt{index:03d}"),
                question=str(record["question"]),
                expected_tool=str(record["expected_tool"]),
                note=str(record.get("note") or ""),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------
def source_matches(file_name: str, gold_sources: Iterable[str]) -> bool:
    """文件名是否命中任一金标来源（子串匹配，兼容扩展名差异）。"""
    if not file_name:
        return False
    return any(gold in file_name for gold in gold_sources if gold)


def compute_recall_and_rr(
    hits: List[Dict[str, Any]], gold_sources: List[str], k: int
) -> tuple:
    """计算 Recall@k 与 MRR。

    Returns:
        ``(是否命中, 倒数排名)``；未命中时倒数排名为 0。
    """
    if not gold_sources:
        return False, 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if source_matches(str(hit.get("file_name") or ""), gold_sources):
            return True, 1.0 / rank
    return False, 0.0


def keyword_coverage(answer: str, keywords: List[str]) -> float:
    """答案对关键字的覆盖度（0~1）。

    用于离线判分：不追求语义等价，只检查"该出现的关键信息有没有出现"。
    数字类关键字（如 "5"、"600"）能有效区分"答对"与"胡答"。
    """
    if not keywords:
        return 1.0
    text = (answer or "").lower()
    hit = sum(1 for keyword in keywords if str(keyword).lower() in text)
    return round(hit / len(keywords), 4)


# ---------------------------------------------------------------------------
# 聚合指标
# ---------------------------------------------------------------------------
def summarize(results: List[CaseResult], k: int) -> Dict[str, Any]:
    """把逐条结果聚合成报告用的指标。

    **指标口径（很重要，写错会让报告失去意义）**：

    * ``recall`` / ``mrr``：只在**有金标来源**的用例上计算
      （也就是"可回答"的那些）。如果把"应拒答"用例也算进分母，
      它们的 recall 恒为 0，会把整体数值稀释成一个没有意义的数字——
      这是第一版报告踩过的坑；
    * ``refusal_accuracy``：只在**应拒答**的用例上计算；
    * ``answer_accuracy``：两类合起来看——可答案例"答了且关键字覆盖达标"，
      应拒答案例"正确拒答"，各按自身标准判定，避免口径混用。

    Args:
        results: 全部用例结果。
        k: Recall@k 的 k（与检索返回条数一致）。
    """
    if not results:
        return {}

    # 有金标来源 = 可回答用例；题型标为 unanswerable = 应拒答用例
    with_gold = [item for item in results if item.case_type == "answerable"]
    unanswerable = [item for item in results if item.case_type == "unanswerable"]

    recalls = [1.0 if item.recall_hit else 0.0 for item in with_gold]
    rrs = [item.reciprocal_rank for item in with_gold]
    # 拒答准确率只在应拒答用例上算
    refusal_cases = unanswerable
    latencies = [item.total_ms for item in results if item.total_ms > 0]

    routed = [item for item in results if item.route_correct is not None]
    judged = [item for item in results if item.llm_judge is not None]

    def percentile(values: List[int], ratio: float) -> float:
        """线性插值分位数。"""
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        position = (len(ordered) - 1) * ratio
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 1)

    total_tokens = sum(item.prompt_tokens + item.completion_tokens for item in results)
    return {
        "cases": len(results),
        "cases_with_gold": len(with_gold),
        "cases_unanswerable": len(unanswerable),
        "recall": round(statistics.mean(recalls), 4) if recalls else None,
        "mrr": round(statistics.mean(rrs), 4) if rrs else None,
        "refusal_accuracy": (
            round(sum(1 for item in refusal_cases if item.refusal_correct) / len(refusal_cases), 4)
            if refusal_cases
            else None
        ),
        "route_accuracy": (
            round(sum(1 for item in routed if item.route_correct) / len(routed), 4) if routed else None
        ),
        "keyword_coverage": (
            round(
                statistics.mean([item.answer_keyword_score for item in with_gold if item.answer]),
                4,
            )
            if any(item.answer for item in with_gold)
            else None
        ),
        "llm_judge_accuracy": (
            round(sum(1 for item in judged if item.llm_judge) / len(judged), 4) if judged else None
        ),
        "citation_validity": (
            round(sum(1 for item in results if item.citation_valid) / len(results), 4) if results else 1.0
        ),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "avg": round(statistics.mean(latencies), 1) if latencies else 0.0,
        },
        "tokens": {
            "total": total_tokens,
            "avg_per_case": round(total_tokens / len(results), 1) if results else 0.0,
        },
        "cost_est": round(sum(item.cost_est for item in results), 6),
        "errors": sum(1 for item in results if item.error),
        "recall_k": k,
    }


def _has_gold(item: "CaseResult") -> bool:
    """该用例是否标了金标来源（标了 = 资料中应有答案）。

    用 ``gold_sources`` 而不是 ``recall_hit``：后者是执行结果，
    命中与否不能用来判断"这条用例本身该不该命中"。
    """
    return bool(getattr(item, "gold_sources", None))


def attach_gold(results: List["CaseResult"], cases: List[Any]) -> None:
    """把用例的金标来源与题型回填到结果里（summarize 依赖它们区分两类用例）。"""
    by_id = {
        case.id: (list(getattr(case, "gold_sources", []) or []), str(getattr(case, "type", "answerable")))
        for case in cases
    }
    for item in results:
        gold, case_type = by_id.get(item.case_id, ([], "answerable"))
        item.gold_sources = gold
        item.case_type = case_type


def format_metric(value: Any, percent: bool = True) -> str:
    """把指标格式化成报告里好读的字符串。"""
    if value is None:
        return "—"
    if isinstance(value, float) and percent:
        return f"{value * 100:.1f}%"
    return str(value)


__all__ = [
    "CaseResult",
    "QaCase",
    "RoutingCase",
    "attach_gold",
    "compute_recall_and_rr",
    "format_metric",
    "keyword_coverage",
    "load_jsonl",
    "load_qa_cases",
    "load_routing_cases",
    "source_matches",
    "summarize",
]
