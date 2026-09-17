"""RAG 包：文档加载、切块、向量化、向量存储、入库管道。

模块职责：

============================  ==========================================================
模块                           职责
============================  ==========================================================
``rag.models``                数据模型（Chunk / RetrievedChunk / SourceRef / 入库结果）
``rag.loaders``               PDF / DOCX / TXT / MD 加载，统一 metadata，内容哈希 doc_id
``rag.splitter``              RecursiveCharacterTextSplitter（500/50）+ 中文分隔符 + 块级去重
``rag.embeddings``            bge-small-zh 本地 Embedding + 查询前缀 + hash 兜底后端
``rag.store``                 Chroma 持久化向量库（cosine 空间、幂等写入、按文档删除）
``rag.pipeline``              入库编排（加载→切块→向量化→记录元数据）
``rag.retriever``             混合检索（向量 + BM25 + RRF 融合 + 可选重排）
``rag.answer``                RAG 问答引擎（三层拒答 + 引用校验 + 离线兜底）
============================  ==========================================================
"""

from . import (  # noqa: F401
    answer,
    embeddings,
    loaders,
    models,
    pipeline,
    retriever,
    splitter,
    store,
)

__all__ = [
    "answer",
    "embeddings",
    "loaders",
    "models",
    "pipeline",
    "retriever",
    "splitter",
    "store",
]
