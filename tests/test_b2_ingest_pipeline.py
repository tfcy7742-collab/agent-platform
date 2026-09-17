"""B2 测试：入库管道（加载 → 切块 → 向量化 → 元数据）。

覆盖点（都是"上线后会真实咬人"的场景）：
1. 单文件入库：块数、字符数、页码统计正确，元数据落库；
2. **重复上传跳过**：同内容第二次上传不重复向量化；
3. **强制重传替换**：``force=True`` 时块数不翻倍（先删旧向量再写）；
4. **失败隔离**：不支持的格式 / 空文件 / 超大文件逐文件报错，不影响其他文件；
5. 删除文档：向量与元数据一起消失，其余文档不受影响；
6. 重建索引：清空后按 uploads 目录重新入库。
"""

from __future__ import annotations

from pathlib import Path

import pytest

LONG_TEXT = "\n\n".join(
    [
        "第一条 年假规定：员工入职满一年后享有年假，工作满一年不满十年者每年五天。",
        "第二条 病假规定：员工因病需要休息的，凭二级以上医院开具的病假证明申请病假。",
        "第三条 事假规定：事假为无薪假，全年累计不超过十五天，需提前三个工作日申请。",
        "第四条 报销标准：一线城市住宿标准为每晚六百元，其他城市为每晚四百元。",
        "第五条 保密义务：员工在职期间接触到的技术资料与客户名单均属公司商业秘密。",
        "第六条 离职交接：员工主动离职需提前三十日以书面形式通知公司。",
    ]
)


@pytest.fixture()
def txt_bytes() -> tuple[str, bytes]:
    """一个 TXT 文件的（文件名, 内容）。"""
    return ("制度.txt", LONG_TEXT.encode("utf-8"))


# ---------------------------------------------------------------------------
# 单文件入库
# ---------------------------------------------------------------------------
def test_ingest_single_file(ingest_pipeline, vector_store, txt_bytes) -> None:
    """单文件入库：向量库、元数据、结果统计三者一致。"""
    from infra import db

    name, data = txt_bytes
    path = ingest_pipeline.save_upload(name, data)
    result = ingest_pipeline.ingest_path(path, file_name=name)

    assert result.ok is True
    assert result.skipped is False
    assert result.chunk_count >= 1
    assert result.char_count > 100
    assert result.doc_id and len(result.doc_id) == 16
    assert result.page_count is None

    # 向量库中确实有这些块
    assert vector_store.count() == result.chunk_count
    assert vector_store.has_document(result.doc_id) is True

    # 元数据已落库且可查询
    record = db.get_document(result.doc_id)
    assert record is not None
    assert record["file_name"] == "制度.txt"
    assert record["chunk_count"] == result.chunk_count
    assert record["char_count"] == result.char_count

    # 落盘文件名使用 doc_id，不含用户提供的路径成分
    assert path.name == f"{result.doc_id}.txt"
    assert path.exists()


def test_ingest_result_metadata_has_file_name(ingest_pipeline, vector_store, txt_bytes) -> None:
    """入库后的块必须保留来源文件名（引用展示依赖它）。"""
    name, data = txt_bytes
    path = ingest_pipeline.save_upload(name, data)
    result = ingest_pipeline.ingest_path(path, file_name=name)

    hits = vector_store.search_with_scores("年假 五天", k=3)
    assert hits
    assert hits[0].chunk.file_name == "制度.txt"
    assert hits[0].chunk.doc_id == result.doc_id


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------
def test_duplicate_upload_is_skipped(ingest_pipeline, vector_store, txt_bytes) -> None:
    """同一内容重复上传：第二次跳过，不重复消耗向量化与存储。"""
    name, data = txt_bytes
    first = ingest_pipeline.ingest_uploads([(name, data)])
    assert first.succeeded == 1
    assert first.skipped == 0
    count_after_first = vector_store.count()

    second = ingest_pipeline.ingest_uploads([(name, data)])
    assert second.total == 1
    assert second.skipped == 1
    assert second.failed == 0
    assert second.results[0].skipped is True
    assert "跳过" in second.results[0].reason or "一致" in second.results[0].reason
    assert vector_store.count() == count_after_first, "重复上传不能增加向量数量"


def test_duplicate_upload_with_force_reindexes(ingest_pipeline, vector_store, txt_bytes) -> None:
    """force=True 时重新向量化，但块数不翻倍（先删旧向量）。"""
    name, data = txt_bytes
    first = ingest_pipeline.ingest_uploads([(name, data)])
    count_after_first = vector_store.count()

    forced = ingest_pipeline.ingest_uploads([(name, data)], force=True)
    assert forced.succeeded == 1
    assert forced.skipped == 0
    assert vector_store.count() == count_after_first, "强制重新入库后块数必须保持不变"


def test_changed_content_creates_new_document(ingest_pipeline, vector_store) -> None:
    """内容变化 → 新 doc_id → 作为新文档入库（旧文档仍在，供人工清理）。"""
    from infra import db

    ingest_pipeline.ingest_uploads([("制度.txt", LONG_TEXT.encode("utf-8"))])
    ingest_pipeline.ingest_uploads([("制度.txt", (LONG_TEXT + "\n\n第七条 新增调休规则。").encode("utf-8"))])

    documents = db.list_documents()
    assert len(documents) == 2
    assert len({item["doc_id"] for item in documents}) == 2
    assert vector_store.count() == sum(int(item["chunk_count"]) for item in documents)


# ---------------------------------------------------------------------------
# 失败隔离
# ---------------------------------------------------------------------------
def test_unsupported_format_is_rejected(ingest_pipeline) -> None:
    """不支持的格式在写盘之前就被拦下，并给出明确原因。"""
    report = ingest_pipeline.ingest_uploads([("表格.xlsx", b"fake-content")])
    assert report.total == 1
    assert report.failed == 1
    assert report.results[0].error_type == "unsupported_format"
    assert "不支持" in (report.results[0].error or "")


def test_empty_file_is_rejected(ingest_pipeline) -> None:
    """空文件报错而不是静默入库。"""
    report = ingest_pipeline.ingest_uploads([("空.txt", b"")])
    assert report.failed == 1
    assert report.results[0].error_type == "empty_file"


def test_oversized_file_is_rejected(ingest_pipeline, monkeypatch) -> None:
    """超过 MAX_UPLOAD_MB 的文件被拒绝（防资源耗尽）。"""
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from config.settings import get_settings
    from rag.pipeline import IngestPipeline

    pipeline = IngestPipeline(settings=get_settings())
    big = ("大文件.txt", b"a" * (2 * 1024 * 1024))       # 2 MB
    report = pipeline.ingest_uploads([big])
    assert report.failed == 1
    assert report.results[0].error_type == "file_too_large"


def test_partial_batch_failure_is_isolated(ingest_pipeline, vector_store, txt_bytes) -> None:
    """批量上传中：好文件入库、坏文件报错，互不影响。"""
    name, data = txt_bytes
    report = ingest_pipeline.ingest_uploads(
        [
            (name, data),
            ("坏文件.xlsx", b"fake"),
            ("空的.txt", b""),
            ("另一个.md", "# 标题\n\n这是一段普通说明文字。".encode("utf-8")),
        ]
    )
    assert report.total == 4
    assert report.succeeded == 2
    assert report.failed == 2
    assert vector_store.count() > 0
    # 每个结果都能对上文件名，方便前端逐条展示
    assert {item.file_name for item in report.results} == {name, "坏文件.xlsx", "空的.txt", "另一个.md"}


def test_save_upload_sanitizes_filename(ingest_pipeline) -> None:
    """上传文件落盘时必须清洗文件名（防路径穿越）。"""
    path = ingest_pipeline.save_upload("../../etc/evil.txt", b"content")
    assert path.parent == ingest_pipeline.settings.upload_path
    assert ".." not in path.name
    assert path.name.endswith(".txt")


# ---------------------------------------------------------------------------
# 目录入库与重建
# ---------------------------------------------------------------------------
def test_ingest_directory(ingest_pipeline, vector_store, tmp_path: Path) -> None:
    """目录批量入库，支持递归且忽略不支持格式。"""
    (tmp_path / "a.txt").write_text(LONG_TEXT, encoding="utf-8")
    (tmp_path / "b.md").write_text(f"# 标题\n\n{LONG_TEXT}", encoding="utf-8")
    (tmp_path / "c.xlsx").write_bytes(b"ignored")
    sub = tmp_path / "子目录"
    sub.mkdir()
    (sub / "d.txt").write_text(LONG_TEXT + "子目录内容。", encoding="utf-8")

    report = ingest_pipeline.ingest_directory(tmp_path)
    assert report.total == 3                      # xlsx 不在允许列表
    assert report.succeeded == 3
    assert report.failed == 0
    assert vector_store.count() > 0


def test_ingest_directory_missing(ingest_pipeline, tmp_path: Path) -> None:
    """目录不存在时给出明确的失败结果（而不是静默返回"成功但 0 个文件"）。"""
    report = ingest_pipeline.ingest_directory(tmp_path / "不存在")
    assert report.total == 1
    assert report.failed == 1
    assert report.succeeded == 0
    assert report.results[0].error_type == "dir_not_found"


def test_load_directory_reports_missing_dir(tmp_path: Path) -> None:
    """目录不存在时，工具层给出明确的 dir_not_found（而不是空列表）。"""
    from rag.loaders import load_directory

    outcomes = load_directory(tmp_path / "不存在")
    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert outcomes[0].error_type == "dir_not_found"


def test_rebuild_index(ingest_pipeline, vector_store, txt_bytes) -> None:
    """重建索引：清空向量库后按 uploads 目录重新入库，块数恢复且不翻倍。"""
    from infra import db

    name, data = txt_bytes
    ingest_pipeline.ingest_uploads([(name, data)])
    before = vector_store.count()
    assert before > 0

    # 再传一份不同内容的文件，确认重建会把它也纳入
    ingest_pipeline.ingest_uploads([("另一个.md", "# 说明\n\n重建索引测试内容。".encode("utf-8"))])

    report = ingest_pipeline.rebuild_index()
    assert report.succeeded == 2
    assert vector_store.count() == sum(
        int(item["chunk_count"]) for item in db.list_documents()
    )
    assert report.store_stats["chunks"] == vector_store.count()


# ---------------------------------------------------------------------------
# 删除
# ---------------------------------------------------------------------------
def test_delete_document_removes_vectors_and_metadata(ingest_pipeline, vector_store, txt_bytes) -> None:
    """删除文档后：向量、元数据、检索结果三处都要消失。"""
    from infra import db

    name, data = txt_bytes
    keep = ingest_pipeline.ingest_uploads([(name, data)]).results[0]
    other = ingest_pipeline.ingest_uploads([("说明.md", "# 标题\n\n完全无关的另一份文档内容。".encode("utf-8"))]).results[0]

    total_before = vector_store.count()

    result = ingest_pipeline.delete_document(keep.doc_id)
    assert result["ok"] is True
    assert result["removed_chunks"] == keep.chunk_count
    assert db.get_document(keep.doc_id) is None
    assert vector_store.has_document(keep.doc_id) is False
    assert vector_store.count() == total_before - keep.chunk_count

    # 另一份文档不受影响
    assert db.get_document(other.doc_id) is not None
    assert vector_store.has_document(other.doc_id) is True

    # 检索结果里不再出现被删文档
    hits = vector_store.search_with_scores("年假 病假 报销", k=5)
    assert all(item.chunk.doc_id != keep.doc_id for item in hits)


def test_delete_document_with_file(ingest_pipeline, txt_bytes) -> None:
    """可选删除落盘文件。"""
    name, data = txt_bytes
    result = ingest_pipeline.ingest_uploads([(name, data)]).results[0]
    stored = ingest_pipeline.settings.upload_path / f"{result.doc_id}.txt"
    assert stored.exists()

    outcome = ingest_pipeline.delete_document(result.doc_id, remove_file=True)
    assert outcome["ok"] is True
    assert outcome["file_removed"] is True
    assert not stored.exists()


def test_delete_unknown_document(ingest_pipeline) -> None:
    """删除不存在的文档返回结构化错误。"""
    result = ingest_pipeline.delete_document("不存在的docid")
    assert result["ok"] is False
    assert "不存在" in str(result["error"])


# ---------------------------------------------------------------------------
# 支持的格式清单
# ---------------------------------------------------------------------------
def test_supported_extensions_match_settings(ingest_pipeline) -> None:
    """管道支持的格式必须与配置一致（否则会出现"配置允许但管道拒绝"的矛盾）。"""
    from config.settings import get_settings

    allowed = get_settings().extension_list
    assert set(allowed) == {".pdf", ".docx", ".txt", ".md"}
    assert ingest_pipeline.is_supported("a.pdf") is True
    assert ingest_pipeline.is_supported("a.docx") is True
    assert ingest_pipeline.is_supported("a.md") is True
    assert ingest_pipeline.is_supported("a.txt") is True
    assert ingest_pipeline.is_supported("a.xlsx") is False
    assert ingest_pipeline.is_supported("无后缀") is False
