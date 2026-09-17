"""探测：BM25 字面命中数能否区分「文档外问题」与「文档内问题」。

动机：真实模型上向量分数存在重叠（文档外 "如何用 Python 写快速排序" 0.469
高于文档内最难的 "年假有几天" 0.469），单靠阈值无法分开。
但直觉上——**文档外问题往往在字面上一个关键词都命中不了**。
本脚本用真实语料验证这个直觉是否成立。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

IN_DOMAIN = [
    "年假有几天",
    "病假需要提供什么证明",
    "报销住宿标准是多少",
    "离线编辑怎么同步",
    "接口返回 502 怎么排查",
    "保密义务要遵守多久",
]
OUT_OF_DOMAIN = [
    "今天北京的天气怎么样",
    "推荐几部科幻电影",
    "怎么给猫剪指甲",
    "2022 年世界杯冠军是谁",
    "如何用 Python 写快速排序",
    "红烧肉怎么做才好吃",
]


def main() -> int:
    """打印每个问题的向量分数与 BM25 命中数。"""
    from config.settings import get_settings
    from rag.store import get_vector_store

    settings = get_settings()
    store = get_vector_store()
    print(f"向量库块数：{store.count()}")

    from rag.retriever import HybridRetriever

    retriever = HybridRetriever(settings=settings, vector_store=store)
    if store.count() == 0:
        print("向量库为空，请先上传样例文档（scripts/verify_b3.py 会做）")
        return 1

    print("=" * 96)
    print(f"{'类型':6s} {'问题':30s} {'向量分':>8s} {'BM25命中':>9s} {'BM25最高分':>11s} {'融合后Top1':>10s}")
    print("=" * 96)

    for label, questions in (("文档内", IN_DOMAIN), ("文档外", OUT_OF_DOMAIN)):
        for question in questions:
            hits = retriever.retrieve(question, top_k=3)
            stats = retriever.last_stats
            bm25_scores = [score for _chunk, score in retriever._bm25_search(question, 10)]
            top = hits[0] if hits else None
            print(
                f"{label:6s} {question:30s} "
                f"{(top.score if top else 0.0):8.4f} "
                f"{stats['bm25_hits']:9d} "
                f"{(max(bm25_scores) if bm25_scores else 0.0):11.4f} "
                f"{(top.retriever if top else '-'):>10s}"
            )

    print("=" * 96)
    print("观察：若「文档外」的 BM25 命中数普遍为 0，则可用")
    print("      「向量分中等偏低 且 BM25 零命中」作为额外拒答闸门。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
