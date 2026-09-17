"""向量存储模块（模块 2）。

基于 **ChromaDB** 的持久化向量库封装，持久化目录由 ``CHROMA_DIR`` 决定（默认 ``./chroma_db``）。

对外接口保持需求中的命名习惯：
* ``add_documents(chunks)``  —— 写入（幂等，同 id 覆盖）
* ``similarity_search(query, k=3)`` —— 相似度检索
* 另加 ``search_with_scores`` / ``delete_document`` / ``has_document`` / ``stats``

四个工程细节（都是踩过坑才会写的）
----------------------------------
1. **显式指定 cosine 空间**：Chroma 默认是 L2 距离，语义检索场景应使用余弦。
   建集合时传 ``collection_metadata={"hnsw:space": "cosine"}``，
   检索返回的 distance 才是"1 - 余弦相似度"，可以直接映射成 0~1 的分数。
2. **距离 → 相似度**：``score = 1 - distance``（cosine）。若将来换成 L2 空间，
   代码里也做了兜底换算 ``1/(1+distance)``，避免分数语义错乱导致拒答阈值失效。
3. **确定性 chunk_id**：``{doc_id}:{page}:{index}``，同一份文件重复上传时
   走 upsert 覆盖而不是插入重复向量——这是"重传不产生重复"的关键。
4. **空库保护**：向量库为空时直接返回空列表，不把 Chroma 的底层异常抛给上层。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence
from config.settings import Settings, get_settings
from rag.embeddings import EmbeddingService, get_embedding_service
from rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 语料版本号
# ---------------------------------------------------------------------------
# 任何写操作（新增/删除/清空）都会让版本号 +1。
# 用途：BM25 这类需要在内存里维护索引的组件，用它判断"语料是否变了"。
# 为什么不用「块数」做指纹：不同知识库的块数可能相同（例如两个空库、或都恰好 10 块），
# 块数相同但内容不同会导致索引复用错数据——这是实际踩到过的 bug。
_corpus_version = 0
_version_lock = threading.Lock()


def bump_corpus_version() -> int:
    """语料变更时递增版本号（写路径统一调用）。"""
    global _corpus_version
    with _version_lock:
        _corpus_version += 1
        return _corpus_version


def get_corpus_version() -> int:
    """读取当前语料版本号。"""
    with _version_lock:
        return _corpus_version


class VectorStore:
    """Chroma 持久化向量库封装。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_service: Optional[EmbeddingService] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.collection_name = collection_name or self.settings.collection_name
        self.persist_directory = str(persist_directory or self.settings.chroma_path)
        self._store: Any = None   # 延迟创建：避免仅导入模块就建库
        self._client: Any = None  # 独立的 chromadb 客户端，见 _get_client 的说明

    # ------------------------------------------------------------------
    # 底层句柄
    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        """按需创建**本实例专属**的 chromadb 客户端。

        为什么需要它：chromadb 的 ``PersistentClient`` 在进程内按路径全局缓存，
        当同一个进程里先后使用**同一个路径的不同知识库**（测试反复重建临时目录、
        或用户删除 chroma_db 后重建）时，会拿到一个指向"上一个库"的客户端，
        表现为"刚上传的文档检索不到 / 计数为 0"——这是实际踩到的 bug。

        默认不启用（``CHROMA_FRESH_CLIENT=false``）以沿用 langchain-chroma 的
        默认行为；测试与多知识库场景下打开它即可获得完全隔离。
        """
        if not self.settings.chroma_fresh_client:
            return None
        if self._client is not None:
            return self._client
        try:
            import chromadb

            self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        except Exception as exc:  # noqa: BLE001 - 创建失败时回退到默认行为
            logger.warning("创建独立 Chroma 客户端失败，回退到默认客户端：%s", exc)
            self._client = None
        return self._client

    @property
    def store(self) -> Any:
        """懒加载 ``langchain_chroma.Chroma`` 实例。"""
        if self._store is None:
            from langchain_chroma import Chroma

            self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_service.backend,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
                client=self._get_client(),
            )
            logger.info(
                "向量库就绪：collection=%s dir=%s 现有块数=%s",
                self.collection_name, self.persist_directory, self.count(),
            )
        return self._store

    @property
    def collection(self) -> Any:
        """底层 chromadb 集合（用于计数、按条件删除等精确操作）。"""
        return self.store._collection  # noqa: SLF001 - langchain-chroma 稳定的内部句柄

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add_documents(self, chunks: Sequence[Chunk]) -> int:
        """批量写入块（幂等：相同 chunk_id 会被覆盖）。

        Args:
            chunks: 待入库的块。

        Returns:
            实际写入的块数。
        """
        if not chunks:
            return 0

        from langchain_core.documents import Document

        documents = [
            Document(page_content=chunk.text, metadata=chunk.to_metadata()) for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]

        started = time.perf_counter()
        # 分批写入，避免一次提交过大导致内存峰值
        batch = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(documents), batch):
            self.store.add_documents(
                documents=documents[start : start + batch],
                ids=ids[start : start + batch],
            )
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("写入向量库：%s 个块，耗时 %s ms", len(documents), elapsed)
        bump_corpus_version()   # 通知 BM25 等内存索引：语料已变化
        return len(documents)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 3) -> List[Chunk]:
        """相似度检索，返回块列表（不带分数）。

        Args:
            query: 查询文本（内部会自动加 bge 检索前缀）。
            k: 返回条数，默认 3（与需求一致）。
        """
        return [item.chunk for item in self.search_with_scores(query, k=k)]

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        """相似度检索，返回带分数的结果。

        Args:
            query: 查询文本。
            k: 返回条数；默认取 ``RETRIEVE_CANDIDATES``（给后续融合/重排留候选）。
            min_score: 低于该分数的结果直接丢弃（0 表示不过滤）。

        Returns:
            按分数降序排列的 ``RetrievedChunk`` 列表；空库或异常时返回空列表。
        """
        query = (query or "").strip()
        if not query:
            return []

        k = k or self.settings.retrieve_candidates
        if self.count() == 0:
            return []

        try:
            pairs = self.store.similarity_search_with_score(query, k=k)
        except Exception as exc:  # noqa: BLE001 - 向量库异常不应让整个问答崩掉
            logger.warning("向量检索失败，返回空结果：%s", exc)
            return []

        results: List[RetrievedChunk] = []
        for rank, (document, distance) in enumerate(pairs, start=1):
            score = self._distance_to_score(float(distance))
            if score < min_score:
                continue
            chunk = Chunk.from_metadata(
                chunk_id=str(document.metadata.get("chunk_id") or document.id or ""),
                text=document.page_content,
                metadata=document.metadata,
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    vec_score=score,
                    rank_fused=rank,
                    rank_final=rank,
                    retriever="vector",
                )
            )
        return results

    def _distance_to_score(self, distance: float) -> float:
        """把 Chroma 的距离换算成 0~1 的相似度分数（越高越相关）。"""
        space = "cosine"
        try:
            metadata = self.collection.metadata or {}
            space = str(metadata.get("hnsw:space", "cosine")).lower()
        except Exception:  # noqa: BLE001 - 读不到元数据时按默认 cosine 处理
            space = "cosine"

        if space == "cosine":
            # cosine 距离范围 [0, 2]，正常相似内容接近 0
            score = 1.0 - distance
        elif space == "l2":
            score = 1.0 / (1.0 + distance)
        elif space == "ip":
            score = distance  # 内积：越大越相似
        else:
            score = 1.0 - distance
        # 夹到 [0, 1]，避免浮点误差产生 -1e-7 这样的分数影响阈值判断
        return max(0.0, min(1.0, round(score, 6)))

    # ------------------------------------------------------------------
    # 文档级操作
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str) -> int:
        """删除某文档的全部向量，返回删除条数。

        用于"删除文档"与"重新上传前清理旧向量"两种场景。
        """
        if not doc_id:
            return 0
        try:
            existing = self.collection.get(where={"doc_id": doc_id}, include=[])
            ids = list(existing.get("ids") or [])
            if not ids:
                return 0
            self.collection.delete(ids=ids)
            bump_corpus_version()
            logger.info("已从向量库删除文档 %s 的 %s 个块", doc_id, len(ids))
            return len(ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除文档向量失败（doc_id=%s）：%s", doc_id, exc)
            return 0

    def has_document(self, doc_id: str) -> bool:
        """判断某文档是否已入库（只取一条，代价极低）。"""
        if not doc_id:
            return False
        try:
            existing = self.collection.get(where={"doc_id": doc_id}, limit=1, include=[])
            return bool(existing.get("ids"))
        except Exception:  # noqa: BLE001
            return False

    def existing_chunk_ids(self, doc_id: str) -> List[str]:
        """列出某文档已入库的 chunk_id（用于诊断重复块）。"""
        try:
            existing = self.collection.get(where={"doc_id": doc_id}, include=[])
            return list(existing.get("ids") or [])
        except Exception:  # noqa: BLE001
            return []

    def count(self) -> int:
        """向量库中的块总数。"""
        try:
            return int(self.collection.count())
        except Exception:  # noqa: BLE001 - 集合尚未创建时返回 0
            return 0

    def stats(self) -> Dict[str, Any]:
        """向量库统计（供 /health 与 /api/documents 展示）。"""
        try:
            count = self.count()
            metadata = dict(self.collection.metadata or {})
        except Exception:  # noqa: BLE001
            count, metadata = 0, {}
        return {
            "backend": "chromadb",
            "collection": self.collection_name,
            "persist_directory": self.persist_directory,
            "space": metadata.get("hnsw:space", "cosine"),
            "chunks": count,
            "embedding_backend": self.embedding_service.name,
            "embedding_dim": self.embedding_service.dimension,
        }

    def reset(self) -> None:
        """清空集合（危险操作，仅用于测试或重建索引）。"""
        try:
            existing = self.collection.get(include=[])
            ids = list(existing.get("ids") or [])
            if ids:
                self.collection.delete(ids=ids)
            logger.warning("向量库已清空，共删除 %s 个块", len(ids))
            bump_corpus_version()
        except Exception as exc:  # noqa: BLE001
            logger.warning("清空向量库失败：%s", exc)


_store_singleton: Optional[VectorStore] = None


def get_vector_store(reload: bool = False) -> VectorStore:
    """获取全局向量库单例。"""
    global _store_singleton
    if _store_singleton is None or reload:
        _store_singleton = VectorStore()
    return _store_singleton


def reset_vector_store_singleton() -> None:
    """丢弃单例（测试中切换持久化目录时使用）。"""
    global _store_singleton
    _store_singleton = None
