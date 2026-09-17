"""文档管理接口（B2 交付）。

接口列表
--------
====================================  ==================================================
方法 / 路径                            说明
====================================  ==================================================
``POST /api/documents``                上传并向量化（支持多文件，multipart）
``GET  /api/documents``                已入库文档列表 + 向量库统计
``GET  /api/documents/{doc_id}``       单文档详情（含分块统计）
``DELETE /api/documents/{doc_id}``     删除文档（向量 + 元数据，可选删落盘文件）
``POST /api/documents/reindex``        按 uploads 目录重建索引
====================================  ==================================================

设计说明
--------
* 文件内容读取后交给 ``IngestPipeline`` 统一处理（格式/大小/空文件校验都在那层），
  本层只负责 HTTP 语义：multipart 解析、状态码、错误结构。
* 上传接口**永远返回 200**，逐文件的成功/失败放在响应体里——因为批量上传中
  "部分成功"是常态，用 4xx 表达会让前端难以区分"整体失败"与"个别文件失败"。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.models import sanitize_filename
from rag.pipeline import get_ingest_pipeline
from rag.store import get_vector_store
from infra import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["文档"])

# 单次请求的文件数量上限（防止一次拖入几百个文件打满内存）
MAX_FILES_PER_REQUEST = 20


def _invalidate_retrieval_caches() -> None:
    """文档增删后失效检索侧缓存。

    为什么必须做：检索器/引擎在进程内是单例，且 BM25 索引常驻内存。
    文档被删除后如果不清缓存，就会出现"刚删掉的文档还能被检索到"这种
    *数据可见性问题*——在知识库场景里属于严重缺陷（删了敏感文档却还能搜到）。
    """
    try:
        from rag.retriever import get_retriever

        get_retriever().invalidate_cache()
    except Exception as exc:  # noqa: BLE001 - 缓存失效失败不应影响主流程
        logger.warning("失效检索缓存失败（忽略）：%s", exc)


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------
class DocumentItem(BaseModel):
    """文档列表项。"""

    doc_id: str
    file_name: str
    ext: str = ""
    size_bytes: int = 0
    chunk_count: int = 0
    char_count: int = 0
    page_count: Optional[int] = None
    ingest_ms: int = 0
    degraded: bool = False
    created_at: str = ""


class DocumentListResponse(BaseModel):
    """文档列表响应。"""

    ok: bool = True
    total: int = 0
    documents: List[DocumentItem] = Field(default_factory=list)
    store: Dict[str, Any] = Field(default_factory=dict)
    storage: Dict[str, Any] = Field(default_factory=dict)


class IngestFileResult(BaseModel):
    """单个文件的入库结果。"""

    ok: bool
    file_name: str
    doc_id: str = ""
    chunk_count: int = 0
    char_count: int = 0
    page_count: Optional[int] = None
    skipped: bool = False
    reason: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    duplicates: int = 0
    degraded: bool = False
    latency_ms: int = 0


class UploadResponse(BaseModel):
    """批量上传响应。"""

    ok: bool = True
    message: str = ""
    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[IngestFileResult] = Field(default_factory=list)
    store: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0


class DeleteResponse(BaseModel):
    """删除响应。"""

    ok: bool = True
    doc_id: str
    file_name: Optional[str] = None
    removed_chunks: int = 0
    file_removed: bool = False


# ---------------------------------------------------------------------------
# 接口实现
# ---------------------------------------------------------------------------
@router.post("", response_model=UploadResponse, summary="上传并向量化文档")
async def upload_documents(
    files: List[UploadFile] = File(..., description="待上传的文档，支持 PDF / DOCX / TXT / MD"),
    force: bool = Form(False, description="内容未变化时是否强制重新向量化"),
) -> UploadResponse:
    """上传一个或多个文档并写入向量库。

    * 支持 ``.pdf`` / ``.docx`` / ``.txt`` / ``.md``；
    * 相同内容的文件重复上传会被识别并跳过（``skipped=true``）；
    * 单个文件失败不影响其他文件，逐文件返回结果。
    """
    if not files:
        raise HTTPException(status_code=422, detail="未收到任何文件")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=422,
            detail=f"单次最多上传 {MAX_FILES_PER_REQUEST} 个文件，当前 {len(files)} 个",
        )

    pipeline = get_ingest_pipeline()

    # 读取全部文件内容（上传体积上限由 IngestPipeline 按 MAX_UPLOAD_MB 校验）
    payload: List[tuple[str, bytes]] = []
    for upload in files:
        original_name = sanitize_filename(upload.filename or "unnamed")
        data = await upload.read()
        payload.append((original_name, data))

    report = pipeline.ingest_uploads(payload, force=force)
    _invalidate_retrieval_caches()

    message_parts = [f"共 {report.total} 个文件"]
    if report.succeeded:
        message_parts.append(f"{report.succeeded} 个已向量化")
    if report.skipped:
        message_parts.append(f"{report.skipped} 个内容未变化已跳过")
    if report.failed:
        message_parts.append(f"{report.failed} 个失败")

    return UploadResponse(
        ok=report.ok,
        message="，".join(message_parts),
        total=report.total,
        succeeded=report.succeeded,
        skipped=report.skipped,
        failed=report.failed,
        results=[IngestFileResult(**item.model_dump()) for item in report.results],
        store=report.store_stats,
        latency_ms=report.latency_ms,
    )


@router.get("", response_model=DocumentListResponse, summary="文档列表")
async def list_documents() -> DocumentListResponse:
    """列出已入库的所有文档，并附带向量库与存储统计。"""
    records = db.list_documents()
    documents = [
        DocumentItem(
            doc_id=str(item.get("doc_id", "")),
            file_name=str(item.get("file_name", "")),
            ext=str(item.get("ext", "")),
            size_bytes=int(item.get("size_bytes", 0) or 0),
            chunk_count=int(item.get("chunk_count", 0) or 0),
            char_count=int(item.get("char_count", 0) or 0),
            page_count=item.get("page_count"),
            ingest_ms=int(item.get("ingest_ms", 0) or 0),
            degraded=bool(item.get("degraded")),
            created_at=str(item.get("created_at", "")),
        )
        for item in records
    ]
    return DocumentListResponse(
        ok=True,
        total=len(documents),
        documents=documents,
        store=get_vector_store().stats(),
        storage=db.document_stats(),
    )


@router.get("/{doc_id:path}", summary="文档详情")
async def get_document(doc_id: str) -> Dict[str, Any]:
    """查看单个文档的元数据与入库统计。"""
    record = db.get_document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_id}")

    store = get_vector_store()
    record = dict(record)
    record["degraded"] = bool(record.get("degraded"))
    record["indexed_chunks"] = len(store.existing_chunk_ids(doc_id))
    record["store"] = store.stats()
    return {"ok": True, "document": record}


@router.delete("/{doc_id:path}", response_model=DeleteResponse, summary="删除文档")
async def delete_document(doc_id: str, remove_file: bool = False) -> DeleteResponse:
    """删除文档的向量与元数据（可选同时删除落盘文件）。

    删除后再检索该文档的内容应当检索不到——这是"文档生命周期管理"的最小闭环。
    """
    result = get_ingest_pipeline().delete_document(doc_id, remove_file=remove_file)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=str(result.get("error", "删除失败")))
    _invalidate_retrieval_caches()
    return DeleteResponse(
        ok=True,
        doc_id=doc_id,
        file_name=result.get("file_name"),  # type: ignore[arg-type]
        removed_chunks=int(result.get("removed_chunks", 0) or 0),
        file_removed=bool(result.get("file_removed")),
    )


@router.post("/reindex", summary="重建索引")
async def reindex() -> Dict[str, Any]:
    """清空向量库并按 ``UPLOAD_DIR`` 中的文件重新入库。

    更换 Embedding 模型或调整切块参数后需要重建索引（向量维度/切分方式都变了）。
    """
    report = get_ingest_pipeline().rebuild_index()
    _invalidate_retrieval_caches()
    return {
        "ok": report.ok,
        "total": report.total,
        "succeeded": report.succeeded,
        "skipped": report.skipped,
        "failed": report.failed,
        "store": report.store_stats,
        "latency_ms": report.latency_ms,
    }
