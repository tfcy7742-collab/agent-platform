"""B2 测试：文档接口（FastAPI 层）。

覆盖点：
1. 上传响应结构（逐文件结果）与"部分失败也返回 200"的语义；
2. 列表 / 详情 / 删除的完整闭环；
3. 重复上传在接口层表现为 ``skipped``；
4. 格式与大小校验由管道层返回结构化错误；
5. 删除后不能再检索到该文档；
6. /health 会报告向量库块数。
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

LONG_TEXT = "\n\n".join(
    [
        "第一条 年假规定：员工入职满一年后享有年假，工作满一年不满十年者每年五天。",
        "第二条 病假规定：员工因病需要休息的，凭二级以上医院开具的病假证明申请病假。",
        "第三条 报销标准：一线城市住宿标准为每晚六百元，其他城市为每晚四百元。",
        "第四条 保密义务：员工在职期间接触到的技术资料与客户名单均属公司商业秘密。",
    ]
)


def upload(client, files: list[tuple[str, bytes]], force: bool = False):
    """统一的 multipart 上传调用。"""
    payload = [
        ("files", (name, io.BytesIO(data), "application/octet-stream")) for name, data in files
    ]
    return client.post("/api/documents", files=payload, data={"force": str(force).lower()})


# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------
def test_upload_single_document(client) -> None:
    """上传 TXT：返回逐文件结果与向量库统计。"""
    response = upload(client, [("制度.txt", LONG_TEXT.encode("utf-8"))])
    assert response.status_code == 200

    body = response.json()
    assert body["ok"] is True
    assert body["total"] == 1
    assert body["succeeded"] == 1
    assert body["failed"] == 0

    item = body["results"][0]
    assert item["ok"] is True
    assert item["file_name"] == "制度.txt"
    assert item["chunk_count"] >= 1
    assert item["doc_id"] and len(item["doc_id"]) == 16
    assert body["store"]["chunks"] == item["chunk_count"]


def test_upload_multiple_formats(client) -> None:
    """多格式一起上传：TXT / MD 都能入库。"""
    response = upload(
        client,
        [
            ("制度.txt", LONG_TEXT.encode("utf-8")),
            ("FAQ.md", f"# 运维 FAQ\n\n{LONG_TEXT}".encode("utf-8")),
        ],
    )
    body = response.json()
    assert body["succeeded"] == 2
    assert {item["file_name"] for item in body["results"]} == {"制度.txt", "FAQ.md"}


def test_upload_is_idempotent(client) -> None:
    """重复上传同一内容：第二次 skipped，向量库块数不变。"""
    data = ("制度.txt", LONG_TEXT.encode("utf-8"))
    first = upload(client, [data]).json()
    chunks_after_first = first["store"]["chunks"]

    second = upload(client, [data]).json()
    assert second["skipped"] == 1
    assert second["succeeded"] == 0
    assert second["results"][0]["skipped"] is True
    assert second["store"]["chunks"] == chunks_after_first


def test_upload_partial_failure_returns_200(client) -> None:
    """部分失败仍是 200，逐文件说明原因（批量上传中"部分成功"是常态）。"""
    response = upload(
        client,
        [
            ("好的.txt", LONG_TEXT.encode("utf-8")),
            ("表格.xlsx", b"fake"),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False               # 有失败项 → ok=False
    assert body["succeeded"] == 1
    assert body["failed"] == 1

    failures = [item for item in body["results"] if not item["ok"]]
    assert failures[0]["error_type"] == "unsupported_format"
    assert failures[0]["file_name"] == "表格.xlsx"


def test_upload_empty_file(client) -> None:
    """空文件被拒绝并给出原因。"""
    body = upload(client, [("空.txt", b"")]).json()
    assert body["failed"] == 1
    assert body["results"][0]["error_type"] == "empty_file"


def test_upload_requires_files(client) -> None:
    """一个文件都没传 → 422 结构化错误。"""
    response = client.post("/api/documents")
    assert response.status_code == 422


def test_upload_sanitizes_malicious_filename(client) -> None:
    """文件名里的路径成分必须被清洗，且不影响入库。"""
    body = upload(client, [("../../evil.txt", LONG_TEXT.encode("utf-8"))]).json()
    assert body["succeeded"] == 1
    stored = body["results"][0]["file_name"]
    assert ".." not in stored
    assert "/" not in stored and "\\" not in stored


# ---------------------------------------------------------------------------
# 列表与详情
# ---------------------------------------------------------------------------
def test_list_documents(client) -> None:
    """列表接口返回文档元数据与统计。"""
    upload(client, [("制度.txt", LONG_TEXT.encode("utf-8"))])

    body = client.get("/api/documents").json()
    assert body["ok"] is True
    assert body["total"] == 1
    document = body["documents"][0]
    assert document["file_name"] == "制度.txt"
    assert document["chunk_count"] >= 1
    assert document["char_count"] > 0
    assert document["ext"] == ".txt"
    assert body["store"]["chunks"] == document["chunk_count"]


def test_get_document_detail(client) -> None:
    """详情接口：附带已索引块数与向量库信息。"""
    uploaded = upload(client, [("制度.txt", LONG_TEXT.encode("utf-8"))]).json()
    doc_id = uploaded["results"][0]["doc_id"]

    body = client.get(f"/api/documents/{doc_id}").json()
    assert body["ok"] is True
    document = body["document"]
    assert document["doc_id"] == doc_id
    assert document["indexed_chunks"] == document["chunk_count"]
    assert document["store"]["backend"] == "chromadb"


def test_get_unknown_document_returns_404(client) -> None:
    """不存在的文档 → 404。"""
    response = client.get("/api/documents/不存在的docid")
    assert response.status_code == 404


def test_list_empty_store(client) -> None:
    """空库时列表返回 0 条而不是报错。"""
    body = client.get("/api/documents").json()
    assert body["total"] == 0
    assert body["documents"] == []
    assert body["store"]["chunks"] == 0


# ---------------------------------------------------------------------------
# 删除
# ---------------------------------------------------------------------------
def test_delete_document_closes_loop(client) -> None:
    """删除闭环：上传 → 列表可见 → 删除 → 列表为空、块数归零。"""
    uploaded = upload(client, [("制度.txt", LONG_TEXT.encode("utf-8"))]).json()
    doc_id = uploaded["results"][0]["doc_id"]
    chunk_count = uploaded["results"][0]["chunk_count"]

    response = client.delete(f"/api/documents/{doc_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["removed_chunks"] == chunk_count

    assert client.get("/api/documents").json()["total"] == 0
    assert client.get("/api/documents").json()["store"]["chunks"] == 0
    assert client.get(f"/api/documents/{doc_id}").status_code == 404


def test_delete_keeps_other_documents(client) -> None:
    """删除一个文档不影响其他文档。"""
    first = upload(client, [("制度.txt", LONG_TEXT.encode("utf-8"))]).json()["results"][0]
    second = upload(client, [("FAQ.md", f"# FAQ\n\n{LONG_TEXT}".encode("utf-8"))]).json()["results"][0]

    client.delete(f"/api/documents/{first['doc_id']}")

    body = client.get("/api/documents").json()
    assert body["total"] == 1
    assert body["documents"][0]["doc_id"] == second["doc_id"]
    assert body["store"]["chunks"] == second["chunk_count"]


def test_delete_unknown_document_returns_404(client) -> None:
    """删除不存在的文档 → 404。"""
    assert client.delete("/api/documents/不存在").status_code == 404


# ---------------------------------------------------------------------------
# 重建索引与健康检查
# ---------------------------------------------------------------------------
def test_reindex_rebuilds_from_upload_dir(client) -> None:
    """重建索引：清空后按 uploads 目录重新入库，块数一致。"""
    upload(client, [("制度.txt", LONG_TEXT.encode("utf-8"))])
    upload(client, [("FAQ.md", f"# FAQ\n\n{LONG_TEXT}".encode("utf-8"))])

    response = client.post("/api/documents/reindex")
    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 2
    assert body["store"]["chunks"] == client.get("/api/documents").json()["store"]["chunks"]


def test_health_reports_vector_store(client) -> None:
    """健康检查要暴露向量库块数（降级/容量都要可见）。"""
    upload(client, [("制度.txt", LONG_TEXT.encode("utf-8"))])
    body = client.get("/health").json()
    assert body["vector_store"]["chunks"] >= 1
    assert body["vector_store"]["backend"] == "chromadb"
    assert body["storage"]["documents"]["documents"] == 1
    assert body["capabilities"]["embedding"]["backend"] in {"hash", "sentence_transformers"}


def test_openapi_lists_document_endpoints(client) -> None:
    """OpenAPI 文档必须包含文档相关接口（前端/第三方据此对接）。"""
    schema = client.get("/openapi.json").json()
    assert "/api/documents" in schema["paths"]
    assert "/api/documents/{doc_id}" in schema["paths"]
    assert "/api/documents/reindex" in schema["paths"]
