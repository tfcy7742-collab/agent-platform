"""下载并本地验证 BGE Embedding 模型。

用途：把 ``BAAI/bge-small-zh-v1.5`` 拉到本地缓存（默认 ``data/models/``），
并做一次真实编码自检——这一步是 B3 标定拒答阈值的前提，
因为阈值必须基于真实模型的分数分布，hash 兜底后端的分数尺度完全不同。

用法：
    .venv\\Scripts\\python.exe scripts\\download_embedding_model.py

输出：
    · 模型缓存目录与体积
    · 向量维度
    · 三条真实相似度对比（验证查询前缀是否生效）
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def directory_size_mb(path: Path) -> float:
    """统计目录体积（MB）。"""
    if not path.exists():
        return 0.0
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return round(total / 1024 / 1024, 1)


def cosine(left: list[float], right: list[float]) -> float:
    """余弦相似度（两个向量都已归一化，等价于点积）。"""
    return sum(a * b for a, b in zip(left, right))


def main() -> int:
    """下载并验证模型。"""
    from config.settings import get_settings

    settings = get_settings()
    cache_dir = settings.model_cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BGE Embedding 模型下载与自检")
    print("=" * 78)
    print(f"模型名称  ：{settings.embedding_model}")
    print(f"缓存目录  ：{cache_dir}")
    print(f"查询前缀  ：{settings.embedding_query_prefix or '(未配置)'}")
    print(f"设备      ：{settings.embedding_device}")
    print("-" * 78)

    # ---- 1. 加载（首次会触发下载）----
    started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
            cache_folder=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001 - 下载失败原因很多（网络、证书、磁盘）
        print(f"[FAIL] 模型加载失败：{type(exc).__name__}: {exc}")
        print("\n排查建议：")
        print("  1) 网络：确认能访问 huggingface.co（国内网络可设置 HF_ENDPOINT=https://hf-mirror.com）")
        print("  2) 磁盘：确认 data/models 目录可写且剩余空间 > 500MB")
        print("  3) 证书：若报 SSL 错误，检查系统证书或代理设置")
        return 1

    elapsed = round(time.perf_counter() - started, 1)
    # sentence-transformers 5.x 把方法改名为 get_embedding_dimension，做跨版本兼容
    dimension_getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension"
    )
    dimension = int(dimension_getter())
    print(f"[ OK ] 模型加载成功，耗时 {elapsed}s，向量维度 {dimension}")
    print(f"       缓存体积约 {directory_size_mb(cache_dir)} MB")

    # ---- 2. 真实编码与相似度对比 ----
    prefix = settings.embedding_query_prefix
    documents = [
        "员工入职满一年后享有五天年假，满十年享有十天年假。",
        "病假需要提供二级以上医院开具的病假证明，病假期间工资按最低工资标准的百分之八十发放。",
        "云笔记支持移动端离线编辑，恢复网络后自动同步。",
        "接口返回 502 时先检查反向代理日志与上游服务存活状态。",
    ]
    queries = [
        ("年假有几天？", 0),          # 期望命中第 0 条
        ("请假需要什么证明？", 1),     # 期望命中第 1 条（同义改写：病假/请假）
        ("手机断网了还能记笔记吗？", 2),  # 期望命中第 2 条（语义检索，字面不重合）
    ]

    doc_vectors = model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
    print("-" * 78)
    print("检索自检（向量空间余弦相似度，越高越相关）：")
    failures = 0
    for query, expected_index in queries:
        encoded_query = model.encode(
            [f"{prefix}{query}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = [cosine(list(encoded_query), list(vector)) for vector in doc_vectors]
        ranking = sorted(range(len(scores)), key=lambda index: -scores[index])
        top = ranking[0]
        ok = top == expected_index
        if not ok:
            failures += 1
        print(
            f"  [{'PASS' if ok else 'FAIL'}] 查询「{query}」\n"
            f"         Top-1 = 文档{top}（{scores[top]:.4f}）"
            f"  期望文档{expected_index}（{scores[expected_index]:.4f}）\n"
            f"         全部分数：" + "，".join(f"doc{i}={score:.4f}" for i, score in enumerate(scores))
        )

    # ---- 3. 前缀效果对比 ----
    print("-" * 78)
    print("查询前缀效果对比（同一查询，加/不加前缀）：")
    query = "请假需要什么证明？"
    with_prefix = model.encode([f"{prefix}{query}"], normalize_embeddings=True)[0]
    without_prefix = model.encode([query], normalize_embeddings=True)[0]
    for label, vector in (("加前缀", with_prefix), ("不加前缀", without_prefix)):
        scores = [cosine(list(vector), list(doc)) for doc in doc_vectors]
        best = max(range(len(scores)), key=lambda index: scores[index])
        print(
            f"  {label:6s} → Top-1 = 文档{best}（{scores[best]:.4f}），"
            f"文档1 得分 {scores[1]:.4f}"
        )

    print("=" * 78)
    if failures:
        print(f"自检结果：{failures} 条查询未命中预期文档（需检查前缀配置或模型）")
        return 1
    print("自检结果：模型可用，语义检索表现正常。")
    print("下一步：B3 将基于真实分数分布标定 REFUSE_THRESHOLD。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
