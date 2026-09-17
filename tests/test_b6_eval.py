"""B6 测试：评测集、判分与指标计算。

覆盖点（评测框架本身也必须被测，否则"报告不可信"没人知道）：
1. 数据集能正确加载，字段与格式符合约定；
2. **指标口径**：Recall 只在有金标来源的用例上算，拒答准确率只在应拒答用例上算；
3. 判分函数：关键字覆盖、引用校验、Recall/MRR 的边界情况；
4. 报告渲染：markdown 包含核心指标与逐条结果；
5. 数据集质量自检：ID 唯一、题型分布合理、金标来源与样例文档对得上。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

DATASET_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets"
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
REFUSAL = "根据现有资料，我无法回答这个问题"


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------
def test_qa_dataset_loads() -> None:
    """问答集能加载，且规模与题型分布符合预期。"""
    from eval.dataset import load_qa_cases

    cases = load_qa_cases(DATASET_DIR / "rag_qa.jsonl")
    assert len(cases) >= 30, "评测集应当有 30 条以上，否则指标没有统计意义"

    answerable = [case for case in cases if not case.is_unanswerable]
    unanswerable = [case for case in cases if case.is_unanswerable]
    assert len(answerable) >= 20, "可答用例不少于 20 条"
    assert len(unanswerable) >= 10, "应拒答用例不少于 10 条（拒答是核心验收项）"

    for case in answerable:
        assert case.gold_sources, f"{case.id} 缺少金标来源"
        assert case.keywords, f"{case.id} 缺少关键字（离线判分需要）"
    for case in unanswerable:
        assert case.gold_sources == [], f"{case.id} 是应拒答用例，不应有金标来源"
        assert case.expected_answer == REFUSAL, f"{case.id} 的参考答案应是标准拒答话术"


def test_routing_dataset_loads() -> None:
    """路由集能加载，且三个工具 + 能力边界题都有覆盖。"""
    from eval.dataset import load_routing_cases

    cases = load_routing_cases(DATASET_DIR / "routing.jsonl")
    assert len(cases) >= 20

    tools = {case.expected_tool for case in cases}
    assert {"knowledge_search", "trip_planner", "send_email"} <= tools, "三个工具都要有评测用例"
    assert "decline" in tools, "必须有'能力边界'用例（期望不调工具）"
    for case in cases:
        assert case.question.strip(), f"{case.id} 问题为空"


def test_dataset_ids_unique() -> None:
    """ID 必须唯一（报告里用 ID 关联用例与结果）。"""
    from eval.dataset import load_qa_cases, load_routing_cases

    qa_ids = [case.id for case in load_qa_cases(DATASET_DIR / "rag_qa.jsonl")]
    routing_ids = [case.id for case in load_routing_cases(DATASET_DIR / "routing.jsonl")]
    assert len(qa_ids) == len(set(qa_ids)), "问答集存在重复 ID"
    assert len(routing_ids) == len(set(routing_ids)), "路由集存在重复 ID"


def test_dataset_gold_sources_match_sample_docs() -> None:
    """金标来源必须能在样例文档里找到（拼错了报告就没意义）。"""
    from eval.dataset import load_qa_cases

    if not SAMPLE_DIR.exists():
        pytest.skip("样例文档不存在，跳过")

    file_names = [path.name for path in SAMPLE_DIR.iterdir() if path.is_file()]
    cases = load_qa_cases(DATASET_DIR / "rag_qa.jsonl")
    for case in cases:
        for gold in case.gold_sources:
            assert any(gold in name for name in file_names), (
                f"{case.id} 的金标来源 {gold!r} 在样例文档中找不到（现有：{file_names}）"
            )


def test_jsonl_tolerates_comments_and_blank_lines(tmp_path: Path) -> None:
    """JSONL 允许注释行与空行（人工维护的数据集需要这个）。"""
    from eval.dataset import load_jsonl

    path = tmp_path / "sample.jsonl"
    path.write_text(
        '# 这是注释\n\n{"id": "a", "question": "q"}\n\n# 又一条注释\n{"id": "b", "question": "q2"}\n',
        encoding="utf-8",
    )
    records = load_jsonl(path)
    assert len(records) == 2
    assert records[0]["id"] == "a"


def test_jsonl_reports_bad_line(tmp_path: Path) -> None:
    """非法 JSON 要报出具体行号（人工排查时最有用）。"""
    from eval.dataset import load_jsonl

    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a"}\n{不是 JSON}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        load_jsonl(path)


# ---------------------------------------------------------------------------
# 判分函数
# ---------------------------------------------------------------------------
def test_source_matches() -> None:
    """来源匹配是子串匹配（兼容扩展名差异）。"""
    from eval.dataset import source_matches

    assert source_matches("员工手册.pdf", ["员工手册"]) is True
    assert source_matches("云笔记产品需求文档.docx", ["云笔记"]) is True
    assert source_matches("运维技术FAQ.md", ["FAQ"]) is True
    assert source_matches("员工手册.pdf", ["云笔记"]) is False
    assert source_matches("", ["员工手册"]) is False
    assert source_matches("任何文件", []) is False


@pytest.mark.parametrize(
    "hits,gold,expected_hit,expected_rr",
    [
        ([{"file_name": "员工手册.pdf"}], ["员工手册"], True, 1.0),
        ([{"file_name": "其他.pdf"}, {"file_name": "员工手册.pdf"}], ["员工手册"], True, 0.5),
        ([{"file_name": "其他.pdf"}, {"file_name": "另一个.pdf"}, {"file_name": "员工手册.pdf"}], ["员工手册"], True, 1 / 3),
        ([{"file_name": "其他.pdf"}], ["员工手册"], False, 0.0),
        ([], ["员工手册"], False, 0.0),
        ([{"file_name": "员工手册.pdf"}], [], False, 0.0),
    ],
)
def test_recall_and_rr(hits: List[Dict[str, Any]], gold: List[str], expected_hit: bool, expected_rr: float) -> None:
    """Recall@k 与 MRR 的计算（含只在前 k 条里找的约束）。"""
    from eval.dataset import compute_recall_and_rr

    hit, rr = compute_recall_and_rr(hits, gold, k=3)
    assert hit is expected_hit
    assert rr == pytest.approx(expected_rr)


def test_recall_respects_k() -> None:
    """金标来源出现在第 4 位时，Recall@3 应当判为未命中。"""
    from eval.dataset import compute_recall_and_rr

    hits = [{"file_name": f"doc{i}.pdf"} for i in range(3)] + [{"file_name": "员工手册.pdf"}]
    hit, rr = compute_recall_and_rr(hits, ["员工手册"], k=3)
    assert hit is False and rr == 0.0

    hit4, rr4 = compute_recall_and_rr(hits, ["员工手册"], k=4)
    assert hit4 is True and rr4 == pytest.approx(0.25)


@pytest.mark.parametrize(
    "answer,keywords,expected",
    [
        ("年假为 5 天、10 天、15 天", ["5", "10", "15"], 1.0),
        ("年假为 5 天", ["5", "10", "15"], 1 / 3),
        ("我不知道", ["5"], 0.0),
        ("任意回答", [], 1.0),
    ],
)
def test_keyword_coverage(answer: str, keywords: List[str], expected: float) -> None:
    """关键字覆盖率（离线判分的核心）。"""
    from eval.dataset import keyword_coverage

    assert keyword_coverage(answer, keywords) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# 指标口径（重点）
# ---------------------------------------------------------------------------
def _make_result(case_id: str, gold: List[str], recall_hit: bool, refused: bool, **kwargs: Any):
    """构造一条用例结果。"""
    from eval.dataset import CaseResult

    item = CaseResult(case_id=case_id, question=f"问题 {case_id}")
    item.gold_sources = gold
    item.case_type = "answerable" if gold else "unanswerable"
    item.recall_hit = recall_hit
    item.reciprocal_rank = 1.0 if recall_hit else 0.0
    item.refused = refused
    item.answer = "根据现有资料，我无法回答这个问题" if refused else "年假为 5 天"
    item.refusal_correct = refused if not gold else (not refused)
    item.hits = [{"file_name": "员工手册.pdf"}]
    for key, value in kwargs.items():
        setattr(item, key, value)
    return item


def test_summarize_recall_excludes_unanswerable() -> None:
    """**核心口径**：Recall 只在有金标来源的用例上算。

    2 条可答（都命中）+ 2 条应拒答（无金标）→ Recall 必须是 100%，
    而不是被应拒答用例稀释成 50%。这是第一版报告踩过的坑。
    """
    from eval.dataset import summarize

    results = [
        _make_result("a1", ["员工手册"], True, False),
        _make_result("a2", ["FAQ"], True, False),
        _make_result("u1", [], False, True),
        _make_result("u2", [], False, True),
    ]
    metrics = summarize(results, k=3)
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["cases_with_gold"] == 2
    assert metrics["cases_unanswerable"] == 2
    assert metrics["refusal_accuracy"] == 1.0


def test_summarize_refusal_accuracy_only_on_unanswerable() -> None:
    """拒答准确率只在应拒答用例上算：一条漏拒 → 50%。"""
    from eval.dataset import summarize

    results = [
        _make_result("u1", [], False, True),                       # 正确拒答
        _make_result("u2", [], False, False),                      # 漏拒（错误地答了）
        _make_result("a1", ["员工手册"], True, False),              # 可答且答了
    ]
    results[1].refusal_correct = False
    metrics = summarize(results, k=3)
    assert metrics["refusal_accuracy"] == 0.5
    assert metrics["recall"] == 1.0


def test_summarize_handles_empty() -> None:
    """空结果集返回空字典（报告里显示"—"而不是崩掉）。"""
    from eval.dataset import summarize

    assert summarize([], k=3) == {}


def test_summarize_route_accuracy() -> None:
    """路由准确率只在跑过路由的用例上算。"""
    from eval.dataset import summarize

    results = [
        _make_result("r1", [], False, False),
        _make_result("r2", [], False, False),
    ]
    results[0].route_correct = True
    results[1].route_correct = False
    metrics = summarize(results, k=3)
    assert metrics["route_accuracy"] == 0.5


def test_format_metric() -> None:
    """指标格式化：None 显示为破折号，浮点按百分比展示。"""
    from eval.dataset import format_metric

    assert format_metric(None) == "—"
    assert format_metric(0.6471) == "64.7%"
    assert format_metric(0.5, percent=False) == "0.5"


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def test_report_rendering_contains_metrics() -> None:
    """报告必须包含核心指标、逐条结果与口径说明。"""
    from eval.dataset import attach_gold, load_qa_cases, load_routing_cases
    from eval.run_eval import EvalConfig, render_report

    qa_cases = load_qa_cases(DATASET_DIR / "rag_qa.jsonl")[:4]
    routing_cases = load_routing_cases(DATASET_DIR / "routing.jsonl")[:2]

    qa_results = [
        _make_result(case.id, case.gold_sources, True, case.is_unanswerable)
        for case in qa_cases
    ]
    attach_gold(qa_results, qa_cases)

    routing_results = [_make_result(case.id, [], False, False) for case in routing_cases]
    for index, item in enumerate(routing_results):
        item.route_correct = index == 0
        item.tools_used = [routing_cases[index].expected_tool]

    config = EvalConfig(name="test", description="单元测试配置")
    report = render_report(
        config, qa_cases, routing_cases, qa_results, routing_results,
        k=3, store_stats={"chunks": 10, "embedding_backend": "sentence_transformers"},
        duration_s=1.5, use_judge=False,
    )

    assert "# 评测报告：test" in report
    assert "Recall@3" in report
    assert "拒答准确率" in report
    assert "指标口径说明" in report
    assert "路由逐条结果" in report
    # 未通过用例要单独列出（逐条分析用）
    assert "未通过用例" in report


def test_comparison_rendering() -> None:
    """对比报告要给出各配置的指标与相对差异。"""
    from eval.run_eval import render_comparison

    reports = {
        "baseline": {"qa": {"recall": 0.8, "mrr": 0.8, "latency_ms": {"p50": 10, "p95": 20}, "tokens": {"avg_per_case": 100}}, "routing": {"route_accuracy": 0.9, "latency_ms": {"p50": 5}, "tokens": {"avg_per_case": 50}}},
        "hybrid": {"qa": {"recall": 0.95, "mrr": 0.9, "latency_ms": {"p50": 12, "p95": 25}, "tokens": {"avg_per_case": 120}}, "routing": {"route_accuracy": 0.9, "latency_ms": {"p50": 6}, "tokens": {"avg_per_case": 60}}},
    }
    text = render_comparison(reports, k=3)
    assert "# 配置对比报告" in text
    assert "baseline" in text and "hybrid" in text
    assert "Recall@k" in text
    assert "结论要点" in text
    assert "+15.0 个百分点" in text      # (0.95-0.8)*100


def test_eval_config_registry() -> None:
    """预置配置齐全，且支持动态阈值配置。"""
    from eval.run_eval import CONFIGS, resolve_config

    assert {"baseline", "hybrid", "no_prefix", "hybrid_rerank"} <= set(CONFIGS)
    assert CONFIGS["baseline"].enable_bm25 is False
    assert CONFIGS["hybrid"].enable_bm25 is True
    assert CONFIGS["no_prefix"].query_prefix == ""
    assert CONFIGS["hybrid_rerank"].enable_rerank is True

    dynamic = resolve_config("threshold_0.5")
    assert dynamic.threshold == 0.5
    with pytest.raises(SystemExit):
        resolve_config("不存在")


def test_judge_helper_offline() -> None:
    """未配置大模型时判分返回 None（不计入准确率），而不是判为错误。"""
    from core.llm import LLMClient
    from eval.judges import judge_answer, judge_summary

    client = LLMClient(enable=False)
    correct, reason = judge_answer(client, "年假有几天", "五天", "五天一")
    assert correct is None
    assert "跳过" in reason or "失败" in reason
    assert judge_summary([]) == {"judged": 0, "correct": 0, "accuracy": None}


# ---------------------------------------------------------------------------
# 评测接口（守住"报告能被读到"这件事）
# ---------------------------------------------------------------------------
def test_eval_datasets_endpoint(client) -> None:
    """评测集概况接口必须返回真实规模。

    这条用例是为了守住一个真实 bug：接口模块曾命名为 ``api/eval.py``，
    与顶层 ``eval`` 包**命名冲突**，导致导入解析到错误模块、接口静默返回 0 条。
    HTTP 层不测就发现不了——CLI 跑得再对也没用。
    """
    body = client.get("/api/eval/datasets").json()
    assert body["ok"] is True, body.get("error")
    assert body["qa"]["total"] >= 30, "问答集规模应当被正确读到"
    assert body["qa"]["answerable"] >= 20
    assert body["qa"]["unanswerable"] >= 10
    assert body["routing"]["total"] >= 20
    assert "knowledge_search" in body["routing"]["by_tool"]
    assert body["qa"]["sources"], "应当能列出金标来源"


def test_eval_reports_endpoint(client) -> None:
    """历史报告接口可用（没有报告时也要正常返回，而不是报错）。"""
    body = client.get("/api/eval/reports").json()
    assert body["ok"] is True
    assert isinstance(body["reports"], list)
    for report in body["reports"]:
        assert "config" in report
        assert "metrics" in report


def test_eval_report_detail_404(client) -> None:
    """读取不存在的报告 → 404。"""
    assert client.get("/api/eval/reports/不存在的报告").status_code == 404


def test_eval_report_detail_blocks_path_traversal(client) -> None:
    """报告名要做路径穿越防护（只取 basename）。"""
    response = client.get("/api/eval/reports/..%2F..%2FREADME")
    assert response.status_code == 404


def test_api_eval_module_not_named_eval() -> None:
    """守住命名冲突：API 侧模块名不能叫 eval。"""
    from pathlib import Path

    api_dir = Path(__file__).resolve().parents[1] / "api"
    assert not (api_dir / "eval.py").exists(), "api/eval.py 会与顶层 eval 包命名冲突"
