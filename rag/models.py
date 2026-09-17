"""知识库数据模型。

这里定义检索链路上流转的结构（文档 → 切块 → 命中片段），
与 ``core`` 里的工具/事件模型分开，避免基础层反向依赖业务层。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# 合法文件名：中英文、数字、下划线、短横线、点、括号（半角与全角）、空格
_SAFE_NAME_PATTERN = re.compile(r"[^\w\u4e00-\u9fff\-.()\[\]（）【】 ]+", re.UNICODE)


def sanitize_filename(name: str, fallback: str = "document") -> str:
    """清洗上传文件名。

    安全要点：
    * 只取 basename，丢掉任何目录成分（防路径穿越 ``../../etc/passwd``）；
    * 过滤控制字符与特殊符号（保留中文、括号等常见字符，避免把正常文件名洗成乱码）；
    * 清洗后若只剩分隔符（例如输入是 ``!!!``），回退到默认名；
    * 限制长度并保留原始后缀，避免超长文件名撑爆文件系统。
    """
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    base = _SAFE_NAME_PATTERN.sub("_", base)
    # 去掉首尾的点、下划线、空格：既防止隐藏文件，也让"只剩分隔符"能被识别出来
    base = base.strip(" ._")
    if not base:
        return fallback
    if len(base) > 120:
        stem, dot, ext = base.rpartition(".")
        base = (stem[:100] + dot + ext) if dot else base[:120]
    return base


class Chunk(BaseModel):
    """向量库中的最小单元。"""

    chunk_id: str = Field(..., description="稳定唯一 id：{doc_id}:{page}:{index}")
    text: str = Field(..., description="块文本")
    file_name: str = Field(..., description="来源文件名（原始名，非落盘名）")
    doc_id: str = Field(..., description="文档 id（文件内容 sha256 前 16 位）")
    page: Optional[int] = Field(None, description="页码（PDF 有值，txt/md 为 None）")
    chunk_index: int = Field(0, description="块在文档内的序号，从 0 开始")
    char_len: int = Field(0, description="字符数")
    created_at: str = Field("", description="入库时间")

    def to_metadata(self) -> Dict[str, Any]:
        """转成 Chroma 的 metadata（只允许 str/int/float/bool，None 需剔除）。"""
        return {
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "doc_id": self.doc_id,
            "page": self.page if self.page is not None else -1,  # -1 表示无页码
            "chunk_index": self.chunk_index,
            "char_len": self.char_len,
            "created_at": self.created_at,
        }

    @classmethod
    def from_metadata(cls, chunk_id: str, text: str, metadata: Dict[str, Any]) -> "Chunk":
        """从 Chroma 返回的 metadata 还原对象。"""
        page = metadata.get("page", -1)
        return cls(
            chunk_id=chunk_id,
            text=text,
            file_name=str(metadata.get("file_name", "")),
            doc_id=str(metadata.get("doc_id", "")),
            page=None if page in (-1, None) else int(page),
            chunk_index=int(metadata.get("chunk_index", 0)),
            char_len=int(metadata.get("char_len", len(text))),
            created_at=str(metadata.get("created_at", "")),
        )


class RetrievedChunk(BaseModel):
    """检索命中的片段（含各路分数，便于调试检索质量）。"""

    chunk: Chunk
    score: float = Field(..., description="最终分数（余弦相似度或 RRF 归一化分，0~1）")
    vec_score: Optional[float] = Field(None, description="向量分支分数")
    bm25_score: Optional[float] = Field(None, description="BM25 分支原始分")
    rank_fused: int = Field(0, description="融合后的名次，从 1 开始")
    rank_final: int = Field(0, description="重排后的最终名次，从 1 开始")
    retriever: str = Field("vector", description="命中来源：vector | bm25 | hybrid | rerank")

    @property
    def citation(self) -> str:
        """人类可读的引用标注，例如「员工手册.pdf 第 3 页」。"""
        page = f" 第 {self.chunk.page} 页" if self.chunk.page else ""
        return f"{self.chunk.file_name}{page}"


class SourceRef(BaseModel):
    """返回给前端的引用片段。"""

    file_name: str
    page: Optional[int] = None
    chunk_id: str
    score: float
    text: str = Field(..., description="片段原文（已按上限截断）")
    file_path: str = Field("", description="落盘路径，便于前端'打开原文'")
    doc_id: str = ""

    @field_validator("score")
    @classmethod
    def _round_score(cls, value: float) -> float:
        """分数保留 4 位小数，避免前端展示一长串浮点数。"""
        return round(float(value), 4)


class DocumentRecord(BaseModel):
    """已入库文档的元信息（与 documents 表字段对应）。"""

    doc_id: str
    file_name: str
    stored_name: str = ""
    ext: str = ""
    size_bytes: int = 0
    chunk_count: int = 0
    char_count: int = 0
    page_count: Optional[int] = None
    ingest_ms: int = 0
    degraded: bool = False
    created_at: str = ""


class IngestResult(BaseModel):
    """单个文件的入库结果。"""

    ok: bool
    file_name: str
    doc_id: str = ""
    chunk_count: int = 0
    char_count: int = 0
    page_count: Optional[int] = None
    skipped: bool = Field(False, description="内容未变化而跳过时置 True")
    reason: str = Field("", description="跳过或失败的原因")
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: int = 0
    degraded: bool = False
    duplicates: int = Field(0, description="因内容重复而被丢弃的块数")


class IngestReport(BaseModel):
    """一次批量上传的汇总结果。"""

    ok: bool = True
    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[IngestResult] = Field(default_factory=list)
    store_stats: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
