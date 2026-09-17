"""评测 CLI：跑问答集与路由集，输出可对比的 markdown 报告。

用法
----
::

    # 单次评测（默认配置，含真实回答生成）
    python -m eval.run_eval

    # 指定配置名（报告文件名会带配置名，便于对比）
    python -m eval.run_eval --config baseline
    python -m eval.run_eval --config hybrid
    python -m eval.run_eval --config hybrid_rerank

    # 批量对比多组配置（同一进程内依次跑，报告里给出对比表）
    python -m eval.run_eval --compare baseline hybrid
    python -m eval.run_eval --compare baseline hybrid no_prefix

    # 只跑检索指标（更快、不消耗大模型 token）
    python -m eval.run_eval --retrieval-only

    # 控制规模与判分方式
    python -m eval.run_eval --limit 10          # 只跑前 10 条
    python -m eval.run_eval --no-judge          # 只用关键字判分（免费、离线）

配置项含义
----------
==============  ==================================================================
配置名           含义
==============  ==================================================================
``baseline``    纯向量检索 + bge 检索前缀 + 默认阈值（对照组）
``hybrid``      向量 + BM25 + RRF 融合
``no_prefix``   hybrid，但**去掉 bge 查询前缀**（验证前缀是否真的有用）
``hybrid_rerank`` hybrid + CrossEncoder 重排（需先下载 rerank 模型，约 1GB）
``threshold_*`` hybrid，但指定拒答阈值（用于阈值扫描）
==============  ==================================================================

设计说明
--------
* 每次评测在**独立的临时目录**里重建向量库，不污染正式的 ``chroma_db``，
  也避免上一次配置的向量被复用；
* 同进程内跑多组配置：Embedding 模型只加载一次（省十几秒），
  但向量库与检索器按配置重建；
* 报告同时落 markdown（给人看）与 json（给程序读/做回归对比）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

from eval.dataset import (  # noqa: E402
    CaseResult,
    QaCase,
    RoutingCase,
    attach_gold,
    compute_recall_and_rr,
    format_metric,
    keyword_coverage,
    load_qa_cases,
    load_routing_cases,
    summarize,
)

DATASET_DIR = PROJECT_ROOT / "eval" / "datasets"
REPORT_DIR = PROJECT_ROOT / "eval" / "reports"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_docs"
WORK_ROOT = PROJECT_ROOT / ".eval_work"


# ---------------------------------------------------------------------------
# 配置定义
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """一组评测配置（不同检索/拒答策略）。"""

    name: str
    description: str
    enable_bm25: bool = True
    enable_rerank: bool = False
    query_prefix: str = "为这个句子生成表示以用于检索相关文章："
    threshold: Optional[float] = None      # None 表示用系统默认


CONFIGS: Dict[str, EvalConfig] = {
    "baseline": EvalConfig(
        name="baseline",
        description="纯向量检索（对照组）",
        enable_bm25=False,
    ),
    "hybrid": EvalConfig(
        name="hybrid",
        description="向量 + BM25 + RRF 融合",
        enable_bm25=True,
    ),
    "no_prefix": EvalConfig(
        name="no_prefix",
        description="混合检索，但去掉 bge 查询前缀",
        enable_bm25=True,
        query_prefix="",
    ),
    "hybrid_rerank": EvalConfig(
        name="hybrid_rerank",
        description="混合检索 + CrossEncoder 重排",
        enable_bm25=True,
        enable_rerank=True,
    ),
}


def resolve_config(name: str) -> EvalConfig:
    """解析配置名，支持 ``threshold_0.5`` 这种动态阈值配置。"""
    if name in CONFIGS:
        return CONFIGS[name]
    if name.startswith("threshold_"):
        try:
            value = float(name.split("_", 1)[1])
        except ValueError as exc:
            raise SystemExit(f"无法解析阈值配置名：{name}") from exc
        return EvalConfig(
            name=name,
            description=f"混合检索 + 拒答阈值 {value}",
            enable_bm25=True,
            threshold=value,
        )
    raise SystemExit(f"未知配置：{name}（可选：{', '.join(CONFIGS)} 或 threshold_0.5）")


# ---------------------------------------------------------------------------
# 评测环境构建
# ---------------------------------------------------------------------------
def build_engine(config: EvalConfig, work_dir: Path) -> Dict[str, Any]:
    """按配置构建一套独立的检索环境（向量库 + 检索器 + 问答引擎）。

    关键：每个配置用自己的临时 chroma 目录，且开启 ``CHROMA_FRESH_CLIENT``，
    避免 chromadb 的"按路径全局缓存客户端"导致读到上一组配置的向量。
    """
    from config import settings as settings_module

    os.environ["CHROMA_DIR"] = str(work_dir / "chroma_db")
    os.environ["UPLOAD_DIR"] = str(work_dir / "uploads")
    os.environ["ENABLE_BM25"] = "true" if config.enable_bm25 else "false"
    os.environ["ENABLE_RERANK"] = "true" if config.enable_rerank else "false"
    os.environ["EMBEDDING_QUERY_PREFIX"] = config.query_prefix
    os.environ["CHROMA_FRESH_CLIENT"] = "true"
    if config.threshold is not None:
        os.environ["REFUSE_THRESHOLD"] = str(config.threshold)

    settings_module.get_settings.cache_clear()

    from config.settings import get_settings
    from core.llm import get_llm_client
    from rag.answer import RagEngine
    from rag.embeddings import get_embedding_service
    from rag.pipeline import IngestPipeline
    from rag.retriever import HybridRetriever
    from rag.store import VectorStore

    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.upload_path.mkdir(parents=True, exist_ok=True)

    embedding_service = get_embedding_service(reload=True)
    store = VectorStore(settings=settings, embedding_service=embedding_service)
    pipeline = IngestPipeline(settings=settings, embedding_service=embedding_service, vector_store=store)
    retriever = HybridRetriever(settings=settings, vector_store=store, embedding_service=embedding_service)
    engine = RagEngine(
        settings=settings,
        retriever=retriever,
        llm=get_llm_client(),
        vector_store=store,
    )
    return {
        "settings": settings,
        "embedding": embedding_service,
        "store": store,
        "pipeline": pipeline,
        "retriever": retriever,
        "engine": engine,
    }


def ingest_samples(pipeline: Any) -> Dict[str, Any]:
    """把样例文档灌入当前环境的向量库。"""
    files = sorted(path for path in SAMPLE_DIR.iterdir() if path.is_file())
    report = pipeline.ingest_uploads([(path.name, path.read_bytes()) for path in files])
    return report.store_stats


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """你是一个严格的答案评审员。请判断【模型回答】是否**正确回答了**【问题】。

判定标准：
1. 与【参考答案】相比，关键事实（数字、条件、结论）正确即可，不要求措辞一致；
2. 如果问题本身在资料中无答案（参考答案是"根据现有资料，我无法回答这个问题"），
   那么模型必须也拒绝回答才算正确；
3. 只要出现编造、与参考答案矛盾的关键事实，一律判为不正确。

只输出 JSON：{"correct": true/false, "reason": "10 字以内的理由"}
"""

JUDGE_USER_TEMPLATE = """【问题】
{question}

【参考答案】
{expected}

【模型回答】
{answer}
"""


def judge_answer(llm: Any, case: QaCase, answer: str) -> tuple:
    """用大模型判分。返回 ``(是否正确, 理由)``。"""
    if not llm.available:
        return None, "（未配置大模型，跳过 LLM 判分）"
    parsed, result = llm.chat_json(
        JUDGE_SYSTEM_PROMPT,
        JUDGE_USER_TEMPLATE.format(
            question=case.question,
            expected=case.expected_answer or "（资料中无答案，应拒答）",
            answer=answer[:800],
        ),
    )
    if not result.ok or not isinstance(parsed, dict):
        return None, f"判分失败：{result.error_type}"
    return bool(parsed.get("correct")), str(parsed.get("reason") or "")


# ---------------------------------------------------------------------------
# 评测执行
# ---------------------------------------------------------------------------
def run_qa_suite(
    config: EvalConfig,
    env: Dict[str, Any],
    cases: List[QaCase],
    k: int,
    generate_answer: bool,
    use_judge: bool,
    limit: Optional[int],
) -> List[CaseResult]:
    """跑问答评测集。"""
    from core.prompts import REFUSAL_MESSAGE

    retriever = env["retriever"]
    engine = env["engine"]
    llm = env["engine"].llm
    results: List[CaseResult] = []

    for index, case in enumerate(cases[: limit or len(cases)], start=1):
        started = time.perf_counter()
        item = CaseResult(case_id=case.id, question=case.question)
        try:
            # ---- 检索 ----
            hits = retriever.retrieve(case.question, top_k=k)
            item.hits = [
                {
                    "file_name": hit.chunk.file_name,
                    "page": hit.chunk.page,
                    "score": hit.score,
                    "retriever": hit.retriever,
                }
                for hit in hits
            ]
            item.top_score = hits[0].score if hits else 0.0
            item.retrieval_ms = int(retriever.last_stats.get("latency_ms", 0) or 0)
            item.recall_hit, item.reciprocal_rank = compute_recall_and_rr(item.hits, case.gold_sources, k)

            # ---- 回答 ----
            if generate_answer:
                answer = engine.answer(case.question, top_k=k)
                item.answer = answer.answer
                item.refused = answer.refused
                item.prompt_tokens = answer.prompt_tokens
                item.completion_tokens = answer.completion_tokens
                item.cost_est = answer.cost_est
                item.llm_ms = answer.llm_latency_ms

                # 拒答正确性：只有"该拒答的确实拒答了"才算对
                if case.is_unanswerable:
                    item.refusal_correct = answer.refused and answer.answer == REFUSAL_MESSAGE
                else:
                    item.refusal_correct = not answer.refused

                # 引用校验：回答里的编号必须落在片段范围内
                from rag.answer import validate_citations

                item.citation_valid = validate_citations(item.answer, len(answer.hits))["ok"]

                # 关键字覆盖（离线判分）
                item.answer_keyword_score = keyword_coverage(item.answer, case.keywords)

                # LLM 判分（可选）
                if use_judge:
                    correct, reason = judge_answer(llm, case, item.answer)
                    item.llm_judge = correct
                    item.llm_judge_reason = reason
        except Exception as exc:  # noqa: BLE001 - 单条失败不影响整份报告
            item.error = f"{type(exc).__name__}: {exc}"

        item.total_ms = int((time.perf_counter() - started) * 1000)
        results.append(item)
        status = "命中" if item.recall_hit else ("拒答" if item.refused else "未命中")
        print(
            f"    [{index:2d}/{len(cases[: limit or len(cases)])}] {case.id} {status:4s}"
            f" top={item.top_score:.3f} rr={item.reciprocal_rank:.2f}"
            f" kw={item.answer_keyword_score:.2f} {item.total_ms}ms"
        )
    return results


def run_routing_suite(env: Dict[str, Any], cases: List[RoutingCase], limit: Optional[int]) -> List[CaseResult]:
    """跑路由评测集（走真实 Agent 主循环）。

    **判定规则**（很重要，否则会把正确行为判成错误）：

    * ``expected_tool`` 是具体工具名 → 要求**第一个被调用的工具**就是它；
    * ``expected_tool = "decline"`` → 要求**没有调用任何工具**，并且给出了说明性回答。
      这类用例是"能力边界题"：知识库是静态文档，查不了实时天气；
      三个工具都做不了项目排期。此时**正确行为是说明边界，而不是硬调一个工具**——
      第一版数据集把这类问题标成了 knowledge_search，结果把模型的正确判断判成了错误，
      这是设计评测集时最容易犯的错（金标不真实，指标就没有意义）。
    """
    from core.runtime.agent import AgentRuntime
    from core.runtime.planner import Planner
    from core.tools.registry import bootstrap_tools

    registry = bootstrap_tools(force=True)
    llm = env["engine"].llm
    runtime = AgentRuntime(
        registry=registry,
        planner=Planner(registry=registry, llm=llm, use_llm=llm.available),
        llm=llm,
        settings=env["settings"],
    )

    results: List[CaseResult] = []
    for index, case in enumerate(cases[: limit or len(cases)], start=1):
        started = time.perf_counter()
        item = CaseResult(case_id=case.id, question=case.question)
        item.case_type = "routing"
        try:
            result = runtime.run(case.question)
            item.tools_used = list(result.tools_used)
            if case.expected_tool == "decline":
                # 期望不调用工具，且给出有内容的说明（而不是空回答）
                item.route_correct = not item.tools_used and bool(result.answer.strip())
            else:
                item.route_correct = bool(item.tools_used) and item.tools_used[0] == case.expected_tool
            item.answer = result.answer
            item.prompt_tokens = result.prompt_tokens
            item.completion_tokens = result.completion_tokens
            item.cost_est = result.cost_est
        except Exception as exc:  # noqa: BLE001
            item.error = f"{type(exc).__name__}: {exc}"
            item.route_correct = False

        item.total_ms = int((time.perf_counter() - started) * 1000)
        results.append(item)
        mark = "✓" if item.route_correct else "✗"
        used = "、".join(item.tools_used) or "（未调用）"
        print(
            f"    [{index:2d}/{len(cases[: limit or len(cases)])}] {mark} {case.id}"
            f" {used} 期望 {case.expected_tool}  {item.total_ms}ms"
        )
    return results


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------
def render_report(
    config: EvalConfig,
    qa_cases: List[QaCase],
    routing_cases: List[RoutingCase],
    qa_results: List[CaseResult],
    routing_results: List[CaseResult],
    k: int,
    store_stats: Dict[str, Any],
    duration_s: float,
    use_judge: bool,
) -> str:
    """渲染 markdown 报告。"""
    qa_metrics = summarize(qa_results, k)
    routing_metrics = summarize(routing_results, k)

    def metric_table(metrics: Dict[str, Any]) -> str:
        """渲染核心指标表。"""
        return "\n".join(
            [
                "| 指标 | 值 |",
                "| --- | --- |",
                f"| 用例数 | {metrics.get('cases', 0)} |",
                f"| Recall@{k} | {format_metric(metrics.get('recall'))} |",
                f"| MRR | {metrics.get('mrr', 0)} |",
                f"| 拒答准确率 | {format_metric(metrics.get('refusal_accuracy'))} |",
                f"| 关键字覆盖率 | {format_metric(metrics.get('keyword_coverage'))} |",
                f"| LLM 判分准确率 | {format_metric(metrics.get('llm_judge_accuracy'))} |",
                f"| 引用有效率 | {format_metric(metrics.get('citation_validity'))} |",
                f"| P50 延迟 | {metrics.get('latency_ms', {}).get('p50', 0)} ms |",
                f"| P95 延迟 | {metrics.get('latency_ms', {}).get('p95', 0)} ms |",
                f"| 平均 token / 用例 | {metrics.get('tokens', {}).get('avg_per_case', 0)} |",
                f"| 估算成本合计 | {metrics.get('cost_est', 0)} 元 |",
                f"| 执行错误数 | {metrics.get('errors', 0)} |",
            ]
        )

    lines: List[str] = [
        f"# 评测报告：{config.name}",
        "",
        f"- **配置说明**：{config.description}",
        f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **耗时**：{duration_s:.1f} 秒",
        f"- **检索参数**：top_k={k}，BM25={'开' if config.enable_bm25 else '关'}，"
        f"重排={'开' if config.enable_rerank else '关'}，"
        f"查询前缀={'有' if config.query_prefix else '无'}，"
        f"拒答阈值={config.threshold if config.threshold is not None else '默认'}",
        f"- **知识库**：{store_stats.get('chunks', 0)} 个向量块，"
        f"Embedding 后端={store_stats.get('embedding_backend')}",
        f"- **判分方式**：关键字覆盖（离线）{' + LLM judge（在线）' if use_judge else ''}",
        "",
        "## 一、问答集指标",
        "",
        metric_table(qa_metrics),
        "",
        "## 二、路由集指标",
        "",
        f"| 指标 | 值 |",
        f"| --- | --- |",
        f"| 用例数 | {routing_metrics.get('cases', 0)} |",
        f"| 路由准确率 | {format_metric(routing_metrics.get('route_accuracy'))} |",
        f"| P50 延迟 | {routing_metrics.get('latency_ms', {}).get('p50', 0)} ms |",
        f"| P95 延迟 | {routing_metrics.get('latency_ms', {}).get('p95', 0)} ms |",
        f"| 平均 token / 用例 | {routing_metrics.get('tokens', {}).get('avg_per_case', 0)} |",
        "",
        "## 三、问答逐条结果",
        "",
        "| ID | 类型 | 问题 | 检索命中 | Top分数 | 拒答 | 关键字覆盖 | LLM判分 | 耗时 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    case_by_id = {case.id: case for case in qa_cases}
    for item in qa_results:
        case = case_by_id.get(item.case_id)
        kind = "可答" if case and not case.is_unanswerable else "应拒答"
        judge = "—" if item.llm_judge is None else ("✓" if item.llm_judge else "✗")
        lines.append(
            f"| {item.case_id} | {kind} | {item.question[:26]} | "
            f"{'✓' if item.recall_hit else '✗'} | {item.top_score:.3f} | "
            f"{'✓' if item.refusal_correct else '✗'} | {item.answer_keyword_score:.2f} | "
            f"{judge} | {item.total_ms}ms |"
        )

    lines += [
        "",
        "## 四、路由逐条结果",
        "",
        "| ID | 问题 | 实际调用 | 期望 | 结果 | 耗时 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in routing_results:
        case = next((entry for entry in routing_cases if entry.id == item.case_id), None)
        expected = case.expected_tool if case else "?"
        lines.append(
            f"| {item.case_id} | {item.question[:26]} | "
            f"{'、'.join(item.tools_used) or '（未调用）'} | {expected} | "
            f"{'✓' if item.route_correct else '✗'} | {item.total_ms}ms |"
        )

    # 失败用例与诚实说明
    failures = [item for item in qa_results + routing_results if item.route_correct is False or item.refusal_correct is False or item.error]
    if failures:
        lines += ["", "## 五、未通过用例（逐条分析用）", ""]
        for item in failures[:30]:
            reason = item.error or ("路由选错" if item.route_correct is False else "拒答判定不符预期")
            lines.append(f"- **{item.case_id}**（{item.question[:30]}）：{reason}")
            if item.answer:
                lines.append(f"  - 实际回答：{item.answer[:120].replace(chr(10), ' ')}")

    lines += [
        "",
        "## 六、指标口径说明",
        "",
        "* **Recall@k**：期望来源是否出现在前 k 条检索结果（子串匹配文件名）；",
        "* **MRR**：第一个命中的倒数排名，衡量排序质量（1.0 = 命中项排第一）；",
        "* **拒答准确率**：应拒答的用例必须返回**逐字一致**的标准话术才算对；",
        "* **关键字覆盖率**：答案是否包含金标关键信息（数字/专有名词），离线判分，便宜可复现；",
        "* **LLM 判分**：由大模型判断答案正确性（更接近人工，但消耗 token 且不完全可复现）；",
        "* **引用有效率**：回答里的引用编号是否都能对应到检索片段；",
        "* **P50/P95 延迟**：端到端耗时（含检索与生成）。",
        "",
        "> 本报告由 `python -m eval.run_eval` 自动生成，可用 `--compare` 跑多组配置后做横向对比。",
        "",
    ]
    return "\n".join(lines)


def render_comparison(reports: Dict[str, Dict[str, Any]], k: int) -> str:
    """渲染多配置对比表。"""
    lines = [
        "# 配置对比报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- top_k：{k}",
        "",
        "## 问答集",
        "",
        "| 配置 | Recall@k | MRR | 拒答准确率 | 关键字覆盖 | LLM判分 | P50 | P95 | 平均token |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, data in reports.items():
        metrics = data["qa"]
        lines.append(
            f"| {name} | {format_metric(metrics.get('recall'))} | {metrics.get('mrr', 0)} | "
            f"{format_metric(metrics.get('refusal_accuracy'))} | "
            f"{format_metric(metrics.get('keyword_coverage'))} | "
            f"{format_metric(metrics.get('llm_judge_accuracy'))} | "
            f"{metrics.get('latency_ms', {}).get('p50', 0)} ms | "
            f"{metrics.get('latency_ms', {}).get('p95', 0)} ms | "
            f"{metrics.get('tokens', {}).get('avg_per_case', 0)} |"
        )

    lines += [
        "",
        "## 路由集",
        "",
        "| 配置 | 路由准确率 | P50 | 平均token |",
        "| --- | --- | --- | --- |",
    ]
    for name, data in reports.items():
        metrics = data["routing"]
        lines.append(
            f"| {name} | {format_metric(metrics.get('route_accuracy'))} | "
            f"{metrics.get('latency_ms', {}).get('p50', 0)} ms | "
            f"{metrics.get('tokens', {}).get('avg_per_case', 0)} |"
        )

    lines += ["", "## 结论要点", ""]
    names = list(reports)
    if len(names) >= 2:
        base = reports[names[0]]["qa"]
        for name in names[1:]:
            current = reports[name]["qa"]
            delta_recall = (current.get("recall") or 0) - (base.get("recall") or 0)
            delta_p95 = current.get("latency_ms", {}).get("p95", 0) - base.get("latency_ms", {}).get("p95", 0)
            lines.append(
                f"- **{name}** 相对 **{names[0]}**：Recall@k "
                f"{'+' if delta_recall >= 0 else ''}{delta_recall * 100:.1f} 个百分点，"
                f"P95 延迟 {'+' if delta_p95 >= 0 else ''}{delta_p95:.0f} ms"
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def evaluate_config(
    config: EvalConfig,
    qa_cases: List[QaCase],
    routing_cases: List[RoutingCase],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """跑一组配置的完整评测，返回指标与报告文本。"""
    work_dir = WORK_ROOT / config.name
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"配置：{config.name} —— {config.description}")
    print(f"{'=' * 80}")

    started = time.perf_counter()
    env = build_engine(config, work_dir)
    store_stats = ingest_samples(env["pipeline"])
    print(
        f"  向量库就绪：{store_stats.get('chunks')} 块｜"
        f"Embedding={store_stats.get('embedding_backend')}｜"
        f"BM25={'开' if config.enable_bm25 else '关'}｜"
        f"重排={'开' if config.enable_rerank else '关'}"
    )

    print(f"\n  [问答集] {len(qa_cases)} 条（生成答案={args.generate_answer}）")
    qa_results = run_qa_suite(
        config,
        env,
        qa_cases,
        args.top_k,
        args.generate_answer,
        not args.no_judge,
        args.limit,
    )
    attach_gold(qa_results, qa_cases)   # 回填金标来源，供指标口径区分两类用例

    # 检索路径是否真的生效：不同配置的候选数与来源标记应当不同
    probe = env["retriever"].retrieve("年假有几天", top_k=args.top_k)
    retriever_branch = "、".join(sorted({hit.retriever for hit in probe})) or "无"
    print(
        f"  检索自检：候选 {env['retriever'].last_stats.get('vector_hits', 0)} 向量 + "
        f"{env['retriever'].last_stats.get('bm25_hits', 0)} BM25，"
        f"Top-{args.top_k} 来源标记={retriever_branch}"
    )

    print(f"\n  [路由集] {len(routing_cases)} 条")
    routing_results = run_routing_suite(env, routing_cases, args.limit) if args.routing else []

    duration = time.perf_counter() - started
    report_text = render_report(
        config,
        qa_cases,
        routing_cases,
        qa_results,
        routing_results,
        args.top_k,
        store_stats,
        duration,
        not args.no_judge,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = REPORT_DIR / f"{stamp}-{config.name}.md"
    report_path.write_text(report_text, encoding="utf-8")

    payload = {
        "config": config.name,
        "description": config.description,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(duration, 1),
        "top_k": args.top_k,
        "store": store_stats,
        "qa": summarize(qa_results, args.top_k),
        "routing": summarize(routing_results, args.top_k) if routing_results else {},
        "cases": {
            "qa": [item.to_dict() for item in qa_results],
            "routing": [item.to_dict() for item in routing_results],
        },
    }
    json_path = REPORT_DIR / f"{stamp}-{config.name}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  报告已写入：{report_path.relative_to(PROJECT_ROOT)}")
    print(f"  原始数据：  {json_path.relative_to(PROJECT_ROOT)}")

    # 清理临时目录（向量库不含需要保留的状态）
    shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "report_text": report_text,
        "report_path": report_path,
        "payload": payload,
        "qa": payload["qa"],
        "routing": payload["routing"],
    }


def main() -> int:
    """评测入口。"""
    parser = argparse.ArgumentParser(description="Agent 平台离线评测")
    parser.add_argument("--config", default="hybrid", help="配置名（见文件头说明）")
    parser.add_argument("--compare", nargs="+", help="批量对比多组配置")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回条数（默认 3）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（快速验证）")
    parser.add_argument("--no-judge", action="store_true", help="只用关键字判分，不调用大模型判分")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只跑检索指标（不生成答案、不做 LLM 判分，最快且零 token）",
    )
    parser.add_argument("--no-routing", dest="routing", action="store_false", help="跳过路由集")
    args = parser.parse_args()

    if args.retrieval_only:
        args.generate_answer = False
        args.no_judge = True
    else:
        args.generate_answer = True

    qa_cases = load_qa_cases(DATASET_DIR / "rag_qa.jsonl")
    routing_cases = load_routing_cases(DATASET_DIR / "routing.jsonl")
    if not qa_cases:
        print("评测集为空，请检查 eval/datasets/rag_qa.jsonl")
        return 1

    print("=" * 80)
    print("Agent 平台评测")
    print("=" * 80)
    print(f"问答集：{len(qa_cases)} 条（可答 {sum(1 for c in qa_cases if not c.is_unanswerable)}，"
          f"应拒答 {sum(1 for c in qa_cases if c.is_unanswerable)}）")
    print(f"路由集：{len(routing_cases)} 条")

    from config.settings import get_settings

    settings = get_settings()
    print(f"大模型：{'已连接 ' + settings.provider_model if settings.llm_online else '离线（不判分、不生成答案）'}")
    if not settings.llm_online and not args.retrieval_only:
        print("提示：离线模式下无法生成答案，将自动退化为仅检索指标")

    names = args.compare or [args.config]
    reports: Dict[str, Dict[str, Any]] = {}
    for name in names:
        config = resolve_config(name)
        reports[name] = evaluate_config(config, qa_cases, routing_cases, args)

    if len(names) > 1:
        comparison = render_comparison(reports, args.top_k)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        compare_path = REPORT_DIR / f"{stamp}-comparison.md"
        compare_path.write_text(comparison, encoding="utf-8")
        print("\n" + "=" * 80)
        print(comparison)
        print(f"对比报告已写入：{compare_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("评测完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
