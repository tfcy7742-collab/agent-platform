"""文档入库管道（模块 1 + 模块 2 的编排层）。

把「加载 → 切块 → 向量化入库 → 记录元数据」串成一条流水线，
供 API 层与命令行脚本共用。B3 会在此模块内追加 ``answer()``（检索问答），
因此这里叫 pipeline 而不是 ingest。

关键行为
--------
* **内容哈希去重**：``doc_id = sha256(文件字节)[:16]``。第二次上传同一份文件时
  默认直接跳过（``skipped=True``），不会重复消耗 embedding 与存储；
  传 ``force=True`` 才会重新向量化（用于文档被改动但哈希相同等异常场景）。
* **替换语义**：``force=True`` 时先按 doc_id 删除旧向量再写入，
  保证"重传同一份文件后块数不翻倍"。
* **单文件失败隔离**：一个文件解析失败不影响其他文件，结果逐文件返回，
  让用户清楚知道哪份文档没进来、为什么。
* **降级可见**：走 hash 兜底 Embedding、解析出空文本、编码降级等都会标记在结果里。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from config.settings import Settings, get_settings
from infra import db
from rag import loaders
from rag.embeddings import EmbeddingService, get_embedding_service
from rag.models import (
    DocumentRecord,
    IngestReport,
    IngestResult,
    sanitize_filename,
)
from rag.splitter import split_documents
from rag.store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class IngestPipeline:
    """文档入库管道。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

    # ------------------------------------------------------------------
    # 保存上传文件
    # ------------------------------------------------------------------
    def save_upload(self, file_name: str, data: bytes) -> Path:
        """把上传内容落盘。

        安全处理：文件名清洗（去目录、过滤特殊字符），落盘名使用 ``doc_id + 后缀``，
        因此**用户无法通过文件名影响落盘路径**。

        Args:
            file_name: 原始文件名。
            data: 文件字节。

        Returns:
            落盘后的路径。
        """
        import hashlib

        safe_name = sanitize_filename(file_name)
        suffix = Path(safe_name).suffix.lower()
        doc_id = hashlib.sha256(data).hexdigest()[:16]

        self.settings.upload_path.mkdir(parents=True, exist_ok=True)
        target = self.settings.upload_path / f"{doc_id}{suffix}"
        target.write_bytes(data)
        return target

    def is_supported(self, file_name: str) -> bool:
        """判断文件名后缀是否受支持。"""
        return Path(file_name).suffix.lower() in self.settings.extension_list

    # ------------------------------------------------------------------
    # 单文件入库
    # ------------------------------------------------------------------
    def ingest_path(self, file_path: Path, file_name: Optional[str] = None, force: bool = False) -> IngestResult:
        """把磁盘上的单个文件入库。

        Args:
            file_path: 文件路径。
            file_name: 展示用的原始文件名（默认取路径文件名）。
            force: True 时即使内容已入库也重新向量化。
        """
        started = time.perf_counter()
        file_path = Path(file_path)
        # 展示名只取纯文件名：调用方误传整条路径时（例如把目录扫描结果当文件名），
        # 落库的 file_name 会变成一长串路径，前端引用面板将无法阅读。
        display_name = Path(file_name).name if file_name else file_path.name

        # ---- 加载（含格式校验与错误分类）----
        # display_name 必须传给加载器：块 metadata 里的 file_name 是引用面板的展示内容，
        # 而落盘名是 doc_id + 后缀，不能用来做展示。
        outcome = loaders.load_file(file_path, display_name=display_name)
        if not outcome.ok:
            return IngestResult(
                ok=False,
                file_name=display_name,
                doc_id=outcome.doc_id,
                error=outcome.error,
                error_type=outcome.error_type,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        doc_id = outcome.doc_id

        # ---- 幂等：内容未变则跳过 ----
        if not force and self.vector_store.has_document(doc_id):
            record = db.get_document(doc_id) or {}
            logger.info("文档 %s（%s）内容未变化，跳过向量化", display_name, doc_id)
            return IngestResult(
                ok=True,
                file_name=display_name,
                doc_id=doc_id,
                chunk_count=int(record.get("chunk_count", 0)),
                char_count=int(record.get("char_count", 0)),
                page_count=record.get("page_count"),
                skipped=True,
                reason="内容与已入库版本一致（doc_id 相同），已跳过重复向量化",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        # ---- 切块 ----
        chunks, duplicates = split_documents(outcome.documents, self.settings)
        if not chunks:
            return IngestResult(
                ok=False,
                file_name=display_name,
                doc_id=doc_id,
                error="切块后没有任何有效文本",
                error_type="empty_after_split",
                duplicates=duplicates,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        # ---- 强制重传时先清理旧向量，保证不产生重复 ----
        if force:
            self.vector_store.delete_document(doc_id)

        # ---- 写入向量库 ----
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for chunk in chunks:
            chunk.created_at = now
        try:
            written = self.vector_store.add_documents(chunks)
        except Exception as exc:  # noqa: BLE001 - 向量化失败要变成结构化结果
            logger.exception("向量化写入失败：%s", display_name)
            return IngestResult(
                ok=False,
                file_name=display_name,
                doc_id=doc_id,
                error=f"向量化失败：{type(exc).__name__}: {exc}",
                error_type="embedding_error",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        # ---- 记录元数据 ----
        record = DocumentRecord(
            doc_id=doc_id,
            file_name=display_name,
            stored_name=file_path.name,
            ext=file_path.suffix.lower(),
            size_bytes=file_path.stat().st_size,
            chunk_count=written,
            char_count=sum(chunk.char_len for chunk in chunks),
            page_count=outcome.page_count,
            ingest_ms=int((time.perf_counter() - started) * 1000),
            degraded=bool(outcome.degraded or self.embedding_service.degraded),
            created_at=now,
        )
        db.upsert_document({**record.model_dump(), "degraded": int(record.degraded)})

        return IngestResult(
            ok=True,
            file_name=display_name,
            doc_id=doc_id,
            chunk_count=written,
            char_count=record.char_count,
            page_count=record.page_count,
            duplicates=duplicates,
            degraded=record.degraded,
            latency_ms=record.ingest_ms,
        )

    # ------------------------------------------------------------------
    # 批量入库
    # ------------------------------------------------------------------
    def ingest_uploads(
        self,
        files: Sequence[tuple[str, bytes]],
        force: bool = False,
    ) -> IngestReport:
        """批量入库上传的文件。

        Args:
            files: ``[(原始文件名, 文件字节), ...]``
            force: 是否强制重新向量化。
        """
        started = time.perf_counter()
        results: List[IngestResult] = []

        for file_name, data in files:
            # ---- 后缀校验（在写盘之前拦掉非法格式）----
            if not self.is_supported(file_name):
                results.append(
                    IngestResult(
                        ok=False,
                        file_name=file_name,
                        error=(
                            f"不支持的格式 {Path(file_name).suffix or '(无后缀)'}，"
                            f"仅支持 {loaders.supported_extensions_text()}"
                        ),
                        error_type="unsupported_format",
                    )
                )
                continue

            # ---- 空文件校验 ----
            if not data:
                results.append(
                    IngestResult(
                        ok=False, file_name=file_name,
                        error="文件内容为空", error_type="empty_file",
                    )
                )
                continue

            # ---- 大小校验 ----
            size_mb = len(data) / 1024 / 1024
            if size_mb > self.settings.max_upload_mb:
                results.append(
                    IngestResult(
                        ok=False, file_name=file_name,
                        error=f"文件过大（{size_mb:.1f} MB），上限 {self.settings.max_upload_mb} MB",
                        error_type="file_too_large",
                    )
                )
                continue

            try:
                path = self.save_upload(file_name, data)
                results.append(self.ingest_path(path, file_name=file_name, force=force))
            except Exception as exc:  # noqa: BLE001 - 单个文件失败不能中断整批
                logger.exception("处理上传文件失败：%s", file_name)
                results.append(
                    IngestResult(
                        ok=False, file_name=file_name,
                        error=f"{type(exc).__name__}: {exc}", error_type="ingest_error",
                    )
                )

        succeeded = sum(1 for item in results if item.ok and not item.skipped)
        skipped = sum(1 for item in results if item.ok and item.skipped)
        failed = sum(1 for item in results if not item.ok)

        report = IngestReport(
            ok=failed == 0,
            total=len(results),
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
            results=results,
            store_stats=self.vector_store.stats(),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        logger.info(
            "批量入库完成：共 %s 个文件，新增 %s，跳过 %s，失败 %s，耗时 %s ms",
            report.total, succeeded, skipped, failed, report.latency_ms,
        )
        return report

    # ------------------------------------------------------------------
    # 目录入库（供脚本与"知识库初始化"场景使用）
    # ------------------------------------------------------------------
    def ingest_directory(self, directory: Path, force: bool = False) -> IngestReport:
        """把一个目录下的所有受支持文档入库。

        路径直接取自加载结果（``LoadOutcome.file_path``），而不是用
        ``目录 / 文件名`` 重新拼接——后者在递归扫描子目录时会拼错路径。
        """
        started = time.perf_counter()
        results: List[IngestResult] = []
        for outcome in loaders.load_directory(directory, settings=self.settings):
            if not outcome.file_path:
                # 目录不存在 / 加载阶段就失败的条目：转成失败结果
                results.append(
                    IngestResult(
                        ok=False, file_name=outcome.file_name,
                        error=outcome.error, error_type=outcome.error_type,
                    )
                )
                continue
            results.append(
                self.ingest_path(Path(outcome.file_path), file_name=outcome.file_name, force=force)
            )

        succeeded = sum(1 for item in results if item.ok and not item.skipped)
        skipped = sum(1 for item in results if item.ok and item.skipped)
        failed = sum(1 for item in results if not item.ok)
        return IngestReport(
            ok=failed == 0,
            total=len(results),
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
            results=results,
            store_stats=self.vector_store.stats(),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str, remove_file: bool = False) -> Dict[str, object]:
        """删除文档：向量 + 元数据（可选删除落盘文件）。"""
        record = db.get_document(doc_id)
        if record is None:
            return {"ok": False, "error": "文档不存在", "doc_id": doc_id}

        removed_chunks = self.vector_store.delete_document(doc_id)
        db.delete_document(doc_id)

        file_removed = False
        if remove_file:
            stored = self.settings.upload_path / str(record.get("stored_name", ""))
            if stored.exists():
                try:
                    stored.unlink()
                    file_removed = True
                except OSError as exc:
                    logger.warning("删除落盘文件失败：%s", exc)

        return {
            "ok": True,
            "doc_id": doc_id,
            "file_name": record.get("file_name"),
            "removed_chunks": removed_chunks,
            "file_removed": file_removed,
        }

    def rebuild_index(self) -> IngestReport:
        """按 uploads 目录重建整库（清空向量后重新入库）。

        典型用途：更换 Embedding 模型或切块参数后重建索引。
        """
        logger.warning("开始重建索引：清空向量库，然后重新入库 %s", self.settings.upload_path)
        self.vector_store.reset()
        db.reset_all(tables=["documents"])
        return self.ingest_directory(self.settings.upload_path, force=True)


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_pipeline_singleton: Optional[IngestPipeline] = None


def get_ingest_pipeline(reload: bool = False) -> IngestPipeline:
    """获取全局入库管道单例。"""
    global _pipeline_singleton
    if _pipeline_singleton is None or reload:
        _pipeline_singleton = IngestPipeline()
    return _pipeline_singleton


def reset_ingest_pipeline() -> None:
    """丢弃单例（测试中切换向量库/Embedding 时使用）。"""
    global _pipeline_singleton
    _pipeline_singleton = None


__all__ = [
    "IngestPipeline",
    "get_ingest_pipeline",
    "reset_ingest_pipeline",
]
