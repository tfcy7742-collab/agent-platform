"""文档加载模块（模块 1 的加载部分）。

支持格式
--------
===========  ==========================  ==============================
格式          加载器                       页码来源
===========  ==========================  ==============================
``.pdf``     ``PyPDFLoader``              每页一个 Document，自带页码
``.docx``    ``Docx2txtLoader``           无页码（整篇一个 Document）
``.txt``     ``TextLoader``               无页码
``.md``      ``TextLoader``               无页码
===========  ==========================  ==============================

设计要点
--------
* **统一 metadata**：无论哪种加载器，最终每个 Document 都带
  ``file_name / file_path / ext / doc_id``，PDF 额外带 ``page``，
  保证下游切块与引用展示不必区分格式。
* **doc_id 用内容哈希**：``sha256(文件字节)[:16]``。这样"同一份文件重复上传"
  天然可以识别（配合向量库的幂等写入实现去重），改名重传也不会产生重复向量。
* **编码兜底**：中文 txt 常见 GBK/GB18030 编码，UTF-8 解码失败时自动回退尝试，
  并用 ``errors="replace"`` 保证不因个别坏字节整篇失败。
* **失败隔离**：单文件加载失败只影响该文件，返回结构化错误而非抛异常中断批量导入。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import Settings, get_settings
from rag.models import sanitize_filename

logger = logging.getLogger(__name__)

# 各扩展名对应的加载器标识（便于日志与错误提示）
SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".pdf": "PyPDFLoader",
    ".docx": "Docx2txtLoader",
    ".txt": "TextLoader",
    ".md": "TextLoader",
}


@dataclass
class LoadOutcome:
    """一次加载的结果（成功或失败都返回，便于批量场景收集错误）。"""

    ok: bool
    file_name: str
    file_path: str = ""
    doc_id: str = ""
    documents: List[Document] = field(default_factory=list)
    page_count: Optional[int] = None
    char_count: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None
    degraded: bool = False


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def compute_doc_id(file_path: Path) -> str:
    """计算文件内容哈希作为 doc_id（sha256 前 16 位）。

    用内容而不是文件名做标识，好处：
    1. 同一文件重复上传可被识别，跳过重复向量化；
    2. 文件改名后仍指向同一份知识内容。
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def is_supported(file_path: Path, settings: Optional[Settings] = None) -> bool:
    """判断扩展名是否在允许列表内。"""
    settings = settings or get_settings()
    return file_path.suffix.lower() in settings.extension_list


def read_text_with_fallback(file_path: Path) -> Tuple[str, bool]:
    """读取文本文件，UTF-8 失败时回退 GB18030。

    Returns:
        ``(文本, 是否发生了编码降级)``
    """
    raw = file_path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding), encoding != "utf-8"
        except UnicodeDecodeError:
            continue
    # 最后兜底：忽略坏字节，保证内容可用
    return raw.decode("utf-8", errors="replace"), True


def _build_documents(
    file_path: Path,
    doc_id: str,
    texts: List[str],
    pages: Optional[List[int]] = None,
    display_name: Optional[str] = None,
) -> List[Document]:
    """把纯文本列表包装成带统一 metadata 的 Document 列表。

    Args:
        display_name: 展示用的文件名。落盘名是 ``doc_id + 后缀``（为了安全），
            但引用面板需要展示用户看到的原始文件名，因此这里单独传入。
    """
    file_name = display_name or file_path.name
    documents: List[Document] = []
    for index, text in enumerate(texts):
        if not text or not text.strip():
            continue  # 跳过空白页/空文件
        metadata: Dict[str, Any] = {
            "file_name": file_name,
            "file_path": str(file_path),
            "ext": file_path.suffix.lower(),
            "doc_id": doc_id,
        }
        if pages and index < len(pages):
            metadata["page"] = pages[index]
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


# ---------------------------------------------------------------------------
# 单文件加载
# ---------------------------------------------------------------------------
def load_file(
    file_path: Path,
    doc_id: Optional[str] = None,
    display_name: Optional[str] = None,
) -> LoadOutcome:
    """加载单个文档。

    Args:
        file_path: 文件路径。
        doc_id: 可复用的内容哈希；不传则内部计算。
        display_name: 展示用的原始文件名；不传则取路径文件名。
            上传场景下文件以 ``doc_id + 后缀`` 落盘，必须由调用方传入原始名，
            否则引用面板会显示成一串哈希。

    Returns:
        ``LoadOutcome``；任何异常都被捕获并转换成 ``ok=False`` 的结果。
    """
    file_path = Path(file_path)
    file_name = display_name or file_path.name
    suffix = file_path.suffix.lower()

    if not file_path.exists():
        return LoadOutcome(
            ok=False, file_name=file_name, file_path=str(file_path),
            error="文件不存在", error_type="file_not_found",
        )
    if suffix not in SUPPORTED_EXTENSIONS:
        return LoadOutcome(
            ok=False,
            file_name=file_name,
            file_path=str(file_path),
            error=f"不支持的格式 {suffix}，仅支持 {', '.join(SUPPORTED_EXTENSIONS)}",
            error_type="unsupported_format",
        )

    try:
        doc_id = doc_id or compute_doc_id(file_path)
    except OSError as exc:
        return LoadOutcome(
            ok=False, file_name=file_name, file_path=str(file_path),
            error=f"读取文件失败：{exc}", error_type="file_read_error",
        )

    degraded = False
    try:
        if suffix == ".pdf":
            documents, page_count = _load_pdf(file_path, doc_id, file_name)
        elif suffix == ".docx":
            documents = _load_docx(file_path, doc_id, file_name)
            page_count = None
        else:
            documents, encoding_degraded = _load_text(file_path, doc_id, file_name)
            page_count = None
            degraded = encoding_degraded
    except ImportError as exc:  # 依赖缺失（例如未装 docx2txt）
        return LoadOutcome(
            ok=False, file_name=file_name, file_path=str(file_path), doc_id=doc_id,
            error=f"缺少解析依赖：{exc}", error_type="missing_dependency",
        )
    except Exception as exc:  # noqa: BLE001 - 单文件失败不应中断批量导入
        logger.warning("加载文件失败：%s → %s", file_name, exc)
        return LoadOutcome(
            ok=False, file_name=file_name, file_path=str(file_path), doc_id=doc_id,
            error=f"{type(exc).__name__}: {exc}", error_type="parse_error",
        )

    if not documents:
        return LoadOutcome(
            ok=False, file_name=file_name, file_path=str(file_path), doc_id=doc_id,
            error="文档解析后没有任何可用文本（可能是扫描版 PDF 或空文件）",
            error_type="empty_document",
        )

    char_count = sum(len(doc.page_content) for doc in documents)
    return LoadOutcome(
        ok=True,
        file_name=file_name,
        file_path=str(file_path),
        doc_id=doc_id,
        documents=documents,
        page_count=page_count,
        char_count=char_count,
        degraded=degraded,
    )


def _load_pdf(file_path: Path, doc_id: str, display_name: Optional[str] = None) -> Tuple[List[Document], Optional[int]]:
    """加载 PDF：每页一个 Document，页码写入 metadata。"""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    raw_documents = loader.load()

    texts: List[str] = []
    pages: List[int] = []
    for index, doc in enumerate(raw_documents):
        texts.append(doc.page_content or "")
        # PyPDFLoader 的 page 从 0 开始，转成人类习惯的 1 起页码
        page_number = doc.metadata.get("page")
        pages.append(int(page_number) + 1 if isinstance(page_number, int) else index + 1)

    documents = _build_documents(file_path, doc_id, texts, pages, display_name)
    return documents, len(raw_documents)


def _load_docx(file_path: Path, doc_id: str, display_name: Optional[str] = None) -> List[Document]:
    """加载 DOCX：Docx2txtLoader 无页码概念，整篇作为一个 Document。"""
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(str(file_path))
    raw_documents = loader.load()
    texts = [doc.page_content or "" for doc in raw_documents]
    return _build_documents(file_path, doc_id, texts, None, display_name)


def _load_text(file_path: Path, doc_id: str, display_name: Optional[str] = None) -> Tuple[List[Document], bool]:
    """加载 TXT / Markdown：自实现编码兜底，比 TextLoader 更耐受中文 GBK 文件。"""
    text, degraded = read_text_with_fallback(file_path)
    return _build_documents(file_path, doc_id, [text], None, display_name), degraded


# ---------------------------------------------------------------------------
# 目录批量加载
# ---------------------------------------------------------------------------
def load_directory(
    directory: Path,
    recursive: bool = True,
    settings: Optional[Settings] = None,
) -> List[LoadOutcome]:
    """批量加载目录下的所有受支持文档。

    Args:
        directory: 目录路径。
        recursive: 是否递归子目录。
        settings: 配置（决定允许的扩展名）。

    Returns:
        每个文件一个 ``LoadOutcome``（含失败项）。
    """
    settings = settings or get_settings()
    directory = Path(directory)
    if not directory.exists():
        return [
            LoadOutcome(
                ok=False, file_name=directory.name,
                file_path="",          # 明确置空：调用方据此判断"没有可用文件"
                error=f"目录不存在：{directory}", error_type="dir_not_found",
            )
        ]

    pattern = "**/*" if recursive else "*"
    files = sorted(
        path for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in settings.extension_list
    )
    return [load_file(path) for path in files]


def supported_extensions_text() -> str:
    """返回受支持格式的可读文本（用于错误提示与 UI 展示）。"""
    return " / ".join(sorted(SUPPORTED_EXTENSIONS))


def ensure_safe_upload_name(file_name: str) -> str:
    """清洗上传文件名（供 API 层调用，避免路径穿越）。"""
    return sanitize_filename(file_name)
