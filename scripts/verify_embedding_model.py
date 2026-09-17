"""真实 BGE 模型的端到端验证（不依赖服务，直接跑入库 + 检索）。

用途：确认「真实模型 + Chroma」这条链路可用，并打印**真实分数分布**——
B3 标定 ``REFUSE_THRESHOLD`` 需要的就是这份分布（文档内问题 vs 文档外问题的分数差）。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_embedding_model.py

说明：
* 使用独立的临时目录（``data/_verify_model``），不会污染正式的 chroma_db 与上传目录；
* 会加载真实 bge 模型（约 30s，之后走进程内缓存）。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

# 文档内问题（应当召回并且分数较高）
IN_DOMAIN_QUERIES = [
    ("年假有几天", "员工手册"),
    ("病假需要什么证明", "员工手册"),
    ("报销住宿标准是多少", "员工手册"),
    ("保密义务持续多久", "员工手册"),
    ("离职要提前多久通知", "员工手册"),
]

# 文档外问题（应当拒答，分数应当明显偏低）
OUT_OF_DOMAIN_QUERIES = [
    "今天北京的天气怎么样",
    "如何用 Python 写一个快速排序",
    "推荐几部好看的科幻电影",
    "2026 年世界杯在哪个国家举办",
    "怎么给猫剪指甲",
]


def main() -> int:
    """执行真实模型验证。"""
    from config.settings import ensure_directories, ensure_model_endpoint, get_settings

    ensure_model_endpoint()

    settings = get_settings()
    work_dir = PROJECT_ROOT / "data" / "_verify_model"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = PROJECT_ROOT / "data" / "sample_docs"
    if not sample_dir.exists():
        print("样例文档不存在，请先运行 scripts/make_sample_docs.py")
        return 1

    print("=" * 78)
    print("真实 BGE 模型端到端验证（入库 → 检索 → 分数分布）")
    print("=" * 78)
    print(f"模型      ：{settings.embedding_model}")
    print(f"临时工作区：{work_dir}")
    print("-" * 78)

    # ---- 用独立目录构建一套临时向量库 ----
    object.__setattr__(settings, "chroma_dir", str(work_dir / "chroma_db"))
    object.__setattr__(settings, "upload_dir", str(work_dir / "uploads"))
    ensure_directories()

    from rag.embeddings import EmbeddingService
    from rag.pipeline import IngestPipeline
    from rag.store import VectorStore

    embedding_service = EmbeddingService(settings=settings)   # 真实模型（auto 模式）
    print(f"[1] Embedding 后端：{embedding_service.name}，维度 {embedding_service.dimension}")
    if embedding_service.name != "sentence_transformers":
        print("    [WARN] 未能加载真实模型，已降级为 hash —— 下面的分数不代表真实分布")
        print(f"           降级原因：{embedding_service.degrade_reason}")

    store = VectorStore(settings=settings, embedding_service=embedding_service)
    pipeline = IngestPipeline(settings=settings, embedding_service=embedding_service, vector_store=store)

    files = sorted(path for path in sample_dir.iterdir() if path.is_file())
    report = pipeline.ingest_uploads([(path.name, path.read_bytes()) for path in files])
    print(f"[2] 样例文档入库：{report.succeeded} 个成功，{report.failed} 个失败，"
          f"共 {report.store_stats['chunks']} 个块")

    # ---- 文档内问题：分数与命中情况 ----
    print("-" * 78)
    print("[3] 文档内问题（应当召回，且分数较高）")
    in_domain_scores: list[float] = []
    top1_hits = 0
    for query, expected_file in IN_DOMAIN_QUERIES:
        hits = store.search_with_scores(query, k=settings.retrieve_top_k)
        if not hits:
            print(f"    [FAIL] 「{query}」无召回")
            continue
        top = hits[0]
        score = top.score
        in_domain_scores.append(score)
        hit = expected_file in top.chunk.file_name
        top1_hits += 1 if hit else 0
        page_text = f" 第 {top.chunk.page} 页" if top.chunk.page else ""
        print(
            f"    [{'PASS' if hit else 'FAIL'}] 「{query}」→ {top.chunk.file_name}{page_text}"
            f"  score={score:.4f}"
            f"   （幅面：{', '.join(f'{item.score:.3f}' for item in hits)}）"
        )
    print(f"    Top-1 命中率：{top1_hits}/{len(IN_DOMAIN_QUERIES)}")

    # ---- 文档外问题：分数分布（拒答阈值标定依据）----
    print("-" * 78)
    print("[4] 文档外问题（应当拒答，分数应显著偏低）")
    out_of_domain_scores: list[float] = []
    for query in OUT_OF_DOMAIN_QUERIES:
        hits = store.search_with_scores(query, k=settings.retrieve_top_k)
        if not hits:
            print(f"    「{query}」→ 无召回（等价于拒答）")
            out_of_domain_scores.append(0.0)
            continue
        score = hits[0].score
        out_of_domain_scores.append(score)
        print(
            f"    「{query}」→ Top-1 {hits[0].chunk.file_name}（score={score:.4f}）"
        )

    # ---- 分数分布与阈值建议 ----
    print("-" * 78)
    if in_domain_scores and out_of_domain_scores:
        lowest_in = min(in_domain_scores)
        highest_out = max(out_of_domain_scores)
        print("[5] 分数分布与拒答阈值建议")
        print(f"    文档内问题 Top-1 分数：最低 {lowest_in:.4f}，"
              f"最高 {max(in_domain_scores):.4f}，均值 {sum(in_domain_scores) / len(in_domain_scores):.4f}")
        print(f"    文档外问题 Top-1 分数：最低 {min(out_of_domain_scores):.4f}，"
              f"最高 {highest_out:.4f}，均值 {sum(out_of_domain_scores) / len(out_of_domain_scores):.4f}")
        if lowest_in > highest_out:
            suggested = round((lowest_in + highest_out) / 2, 3)
            print(f"    ✅ 两组分数线性可分，建议 REFUSE_THRESHOLD ≈ {suggested}"
                  f"（区间 {highest_out:.3f} ~ {lowest_in:.3f}）")
            print(f"    当前配置值：{settings.refuse_threshold} → "
                  f"{'在建议区间内，无需调整' if highest_out < settings.refuse_threshold < lowest_in else '需要调整'}")
        else:
            print("    ⚠️ 两组分数存在重叠，单一阈值无法完全分开：")
            print(f"       有 {sum(1 for score in out_of_domain_scores if score > lowest_in)} 个文档外问题的"
                  f"分数高于最低的文档内问题分数")
            print("       建议：降低阈值 + 依赖引用校验兜底，或引入 reranker 提升区分度")

        # 逐阈值扫描，给出准确率矩阵（B6 评测会把它做成正式报告）
        print("\n    阈值扫描（拒答命中率 / 误拒率）：")
        print("      阈值    文档外被正确拒答   文档内被误拒")
        for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            refused_ok = sum(1 for score in out_of_domain_scores if score < threshold)
            mis_refused = sum(1 for score in in_domain_scores if score < threshold)
            print(
                f"      {threshold:.2f}    {refused_ok}/{len(out_of_domain_scores)}"
                f"                {mis_refused}/{len(in_domain_scores)}"
            )

    # ---- 清理临时工作区 ----
    shutil.rmtree(work_dir, ignore_errors=True)
    print("=" * 78)
    print("验证完成（临时工作区已清理）。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
