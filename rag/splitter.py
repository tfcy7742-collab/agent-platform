"""文本切块模块（模块 1 的切块部分）。

参数与需求一致：``chunk_size=500``、``chunk_overlap=50``，使用
``RecursiveCharacterTextSplitter``。

中文切分的关键：**分隔符优先级**
--------------------------------
``RecursiveCharacterTextSplitter`` 会按分隔符列表从"最语义化"到"最暴力"依次尝试。
默认分隔符是为英文设计的（``\\n\\n`` / ``\\n`` / ``" "`` / ``""``），
中文文档里句号、问号、分号后面没有空格，按空格切会把句子切碎。
因此这里优先使用中文标点：段落 → 换行 → 中文句末标点 → 分号/逗号 → 字符。

另外做了三件"工程上必须做但教程常忽略"的事：

1. **块级去重**：同一文档内（尤其是有重叠或重复段落的文档）内容完全相同的块只保留一份，
   避免"同一段话被检索命中 3 次、挤占 top_k"这类典型的检索质量事故；
2. **稳定的 chunk_id**：``{doc_id}:{page}:{index}``，index 在整篇文档内连续递增，
   便于引用回原文与幂等重写；
3. **页码透传**：PDF 的 ``page`` 元数据必须原样带到每个块上，否则无法给出"第几页"的引用。
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import Settings, get_settings
from rag.models import Chunk

logger = logging.getLogger(__name__)

# 中文优先的分隔符链：段落 → 换行 → 中文句末 → 英文句末 → 分号/逗号 → 字符
DEFAULT_SEPARATORS: List[str] = [
    "\n\n",      # 段落
    "\n",        # 行
    "。", "！", "？", "；",   # 中文句末标点
    ". ", "! ", "? ", "; ",  # 英文句末（带空格，避免切坏小数点/缩写）
    "，", ", ",  # 逗号
    " ",         # 空格
    "",          # 最后手段：按字符切
]

# 用于判断"两个块内容是否相同"的规范化：去掉所有空白与标点差异
_WHITESPACE_PATTERN = re.compile(r"\s+")


def build_splitter(settings: Optional[Settings] = None) -> RecursiveCharacterTextSplitter:
    """按配置构造切分器。"""
    settings = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,            # 以字符计长，对中文更直观
        separators=DEFAULT_SEPARATORS,
        keep_separator=True,            # 保留句末标点，块读起来更完整
        is_separator_regex=False,
    )


def normalize_for_dedup(text: str) -> str:
    """计算去重用的指纹（忽略空白差异）。"""
    cleaned = _WHITESPACE_PATTERN.sub("", text)
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def split_documents(
    documents: List[Document],
    settings: Optional[Settings] = None,
) -> Tuple[List[Chunk], int]:
    """把加载得到的 Document 列表切成块。

    Args:
        documents: ``loaders.load_file`` 产出的 Document 列表。
        settings: 配置。

    Returns:
        ``(chunks, duplicates)``——``duplicates`` 是因内容重复被丢弃的块数。
    """
    settings = settings or get_settings()
    if not documents:
        return [], 0

    splitter = build_splitter(settings)
    pieces = splitter.split_documents(documents)

    chunks: List[Chunk] = []
    seen: set = set()
    duplicates = 0

    for index, piece in enumerate(pieces):
        text = (piece.page_content or "").strip()
        if not text:
            continue

        fingerprint = normalize_for_dedup(text)
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)

        metadata: Dict[str, Any] = dict(piece.metadata or {})
        doc_id = str(metadata.get("doc_id", "unknown"))
        page = metadata.get("page")
        page_value: Optional[int] = None
        if page is not None:
            try:
                page_value = int(page)
            except (TypeError, ValueError):
                page_value = None

        # chunk_id 中 page 缺省用 0 占位，保证字符串稳定且可解析
        chunk_id = f"{doc_id}:{page_value or 0}:{index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                file_name=str(metadata.get("file_name", "")),
                doc_id=doc_id,
                page=page_value,
                chunk_index=index,
                char_len=len(text),
            )
        )

    logger.info(
        "切块完成：%s 个块 → 去重后 %s 个（丢弃重复 %s 个），chunk_size=%s overlap=%s",
        len(pieces), len(chunks), duplicates, settings.chunk_size, settings.chunk_overlap,
    )
    return chunks, duplicates


def split_text(
    text: str,
    file_name: str = "text",
    doc_id: str = "text",
    settings: Optional[Settings] = None,
) -> List[Chunk]:
    """直接切分一段文本（便于测试与"粘贴文本"场景）。"""
    document = Document(
        page_content=text,
        metadata={"file_name": file_name, "doc_id": doc_id, "ext": ".txt"},
    )
    chunks, _ = split_documents([document], settings)
    return chunks


def splitter_summary(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """返回切分参数的摘要（供 /health 与调试展示）。"""
    settings = settings or get_settings()
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "separators": DEFAULT_SEPARATORS[:5],
        "length_function": "len(字符)",
    }
