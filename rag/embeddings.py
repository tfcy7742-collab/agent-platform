"""Embedding 模块（模块 2 的向量化部分）。

默认使用 **BAAI/bge-small-zh-v1.5**（通过 sentence-transformers 本地加载），
并提供 ``hash`` 兜底后端，保证在没有模型权重、没有网络的环境里也能跑通全链路
（CI 回归、离线演示都靠它）。

三个关键工程点
--------------
1. **查询侧必须加前缀**：bge 系列在检索场景下，查询要加
   ``"为这个句子生成表示以用于检索相关文章："``，而文档侧**不加**。
   网上大量教程两边都不加或两边都加，会明显拉低召回率——评测里会给出对比。
2. **归一化**：向量做 L2 归一化后，余弦相似度退化为点积，与 Chroma 的
   ``cosine`` 空间配合可以稳定地把"距离 → 相似度"映射到 0~1。
3. **后端降级可见**：``auto`` 模式下若模型加载失败（无网络、磁盘不足），
   自动切到 ``hash`` 后端，并在 ``/health`` 标注 ``backend``，绝不静默降级。
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from typing import Any, Callable, Dict, List, Optional

from langchain_core.embeddings import Embeddings

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model_cache: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 哈希兜底后端
# ---------------------------------------------------------------------------
class HashEmbeddings(Embeddings):
    """确定性的哈希向量后端（无模型依赖）。

    原理：把文本切分为"字 + 二元字组"（对中文友好）后，用 md5 把每个 token
    映射到固定维度的一个分量上并累加（符号由哈希决定），最后 L2 归一化。

    它不是语义模型——无法理解同义改写，但具备两个关键性质：
    * **确定性**：同一文本永远得到同一向量；
    * **字面相似性**：共享字/词越多的文本，余弦相似度越高。

    因此足以支撑"检索链路、拒答阈值、Agent 循环、评测框架"的自动化测试。
    """

    def __init__(self, dim: int = 512, prefix: str = "", name: str = "hash") -> None:
        self.dim = dim
        self.prefix = prefix
        self.name = name

    # -- 内部实现 ------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """切分为单字 + 相邻二元组（中文用字，英文用词）。"""
        tokens: List[str] = []
        buffer: List[str] = []

        def flush() -> None:
            if buffer:
                tokens.append("".join(buffer).lower())
                buffer.clear()

        previous = ""
        for char in text.lower():
            if "\u4e00" <= char <= "\u9fff":          # 中文字符
                flush()
                tokens.append(char)
                if previous and "\u4e00" <= previous <= "\u9fff":
                    tokens.append(previous + char)     # 二元组：捕捉"年假""假期"这类词
                previous = char
            elif char.isalnum():
                buffer.append(char)
                previous = char
            else:
                flush()
                previous = ""
        flush()
        return [token for token in tokens if token]

    def _embed(self, text: str) -> List[float]:
        """计算单个文本的向量。"""
        vector = [0.0] * self.dim
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            # 用两个独立位置降低碰撞影响；符号由哈希字节决定，保证不同 token 方向不同
            for offset in (0, 4):
                index = int.from_bytes(digest[offset : offset + 4], "big") % self.dim
                sign = 1.0 if digest[offset + 4] % 2 == 0 else -1.0
                vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            # 空文本或全部被过滤：返回一个固定方向的单位向量，避免除零
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    # -- Embeddings 接口 ----------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档侧：**不加**查询前缀。"""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """查询侧：加上前缀（与真实 bge 后端行为保持一致）。"""
        return self._embed(f"{self.prefix}{text}")


# ---------------------------------------------------------------------------
# sentence-transformers 后端
# ---------------------------------------------------------------------------
class SentenceTransformerEmbeddings(Embeddings):
    """基于 sentence-transformers 的本地 Embedding（bge 系列）。"""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
        cache_dir: Optional[str] = None,
        normalize: bool = True,
        local_files_only: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.cache_dir = cache_dir
        self.normalize = normalize
        self.local_files_only = local_files_only
        self.name = "sentence_transformers"
        self._model = self._load_model()

    def _load_model(self) -> Any:
        """加载模型（进程内按 模型名|设备|缓存目录 缓存，避免重复加载）。

        ``local_files_only`` 的取值很关键，这是实际踩过的坑：
        **``HF_HUB_OFFLINE`` 环境变量在 ``huggingface_hub`` 被 import 之后就失效了**
        （库在导入时读取该变量并缓存为常量），因此"先 import 再设环境变量"完全没用。
        正确做法是显式传参：

        * 模型已在本地缓存 → ``local_files_only=True``，一个网络请求都不发；
        * 未缓存 → ``None``（走默认行为，允许联网/镜像下载）。

        否则每次启动都会去连 huggingface.co，在无法访问的网络里要等
        **5 次重试、每个文件 20~80 秒**，多个文件叠加就是十几分钟的假死。
        """
        cache_key = f"{self.model_name}|{self.device}|{self.cache_dir}|{self.local_files_only}"
        with _lock:
            if cache_key in _model_cache:
                return _model_cache[cache_key]

            from sentence_transformers import SentenceTransformer

            logger.info(
                "正在加载 Embedding 模型：%s（device=%s，local_files_only=%s）",
                self.model_name, self.device, self.local_files_only,
            )
            model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            _model_cache[cache_key] = model
            dimension = getattr(model, "get_embedding_dimension", None) or getattr(
                model, "get_sentence_embedding_dimension", None
            )
            logger.info(
                "Embedding 模型加载完成，向量维度：%s",
                dimension() if callable(dimension) else "未知",
            )
            return model

    @property
    def dimension(self) -> int:
        """向量维度。

        sentence-transformers 5.x 把 ``get_sentence_embedding_dimension`` 改名为
        ``get_embedding_dimension``，旧版本则没有新名——两个都试，保证跨版本可用。
        """
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = getattr(self._model, "get_sentence_embedding_dimension")
        return int(getter())

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码并归一化。"""
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档侧编码（不加前缀）。"""
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """查询侧编码（加检索前缀）。"""
        return self._encode([f"{self.query_prefix}{text}"])[0]


# ---------------------------------------------------------------------------
# 工厂与服务
# ---------------------------------------------------------------------------
def _build_backend(settings: Settings, on_degrade: Optional[Callable[[str], None]] = None) -> Embeddings:
    """按配置构建 Embedding 后端；auto 模式下失败则降级为 hash。"""
    backend = settings.embedding_backend
    prefix = settings.embedding_query_prefix

    if backend == "hash":
        return HashEmbeddings(dim=settings.embedding_dim, prefix=prefix)

    if backend in ("auto", "sentence_transformers"):
        try:
            # 模型已缓存则完全离线加载（不发任何网络请求）；
            # 未缓存时先做一次 HuggingFace 可达性兜底（官方站不通就切镜像）。
            from config.settings import _model_is_cached, ensure_model_endpoint

            cached = _model_is_cached(settings.embedding_model, settings.model_cache_path)
            if not cached:
                ensure_model_endpoint(settings=settings)

            return SentenceTransformerEmbeddings(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                query_prefix=prefix,
                cache_dir=str(settings.model_cache_path),
                local_files_only=True if cached else None,
            )
        except Exception as exc:  # noqa: BLE001 - 离线/磁盘/OOM 等情况都可能失败
            if backend == "sentence_transformers":
                # 显式指定时不静默降级，让使用者立刻发现配置或环境问题
                raise
            message = f"加载 Embedding 模型失败，已降级为 hash 后端：{type(exc).__name__}: {exc}"
            logger.warning(message)
            if on_degrade:
                on_degrade(message)
            return HashEmbeddings(dim=settings.embedding_dim, prefix=prefix)

    raise ValueError(f"未知的 EMBEDDING_BACKEND：{backend}")


class EmbeddingService:
    """Embedding 服务：屏蔽"真实模型 / 兜底后端"的差异，并记录降级原因。"""

    def __init__(self, settings: Optional[Settings] = None, force_backend: Optional[str] = None) -> None:
        self.settings = settings or get_settings()
        self.degraded = False
        self.degrade_reason: Optional[str] = None

        if force_backend:
            # 覆盖后端（测试或评测对比用）。pydantic 模型默认禁止改字段，这里临时放开。
            original = self.settings.embedding_backend
            object.__setattr__(self.settings, "embedding_backend", force_backend)
            try:
                self.backend = _build_backend(self.settings, self._mark_degraded)
            finally:
                object.__setattr__(self.settings, "embedding_backend", original)
        else:
            self.backend = _build_backend(self.settings, self._mark_degraded)

    def _mark_degraded(self, reason: str) -> None:
        """记录降级原因（由工厂回调触发）。"""
        self.degraded = True
        self.degrade_reason = reason

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """当前后端名称：sentence_transformers | hash。"""
        return str(getattr(self.backend, "name", "unknown"))

    @property
    def dimension(self) -> Optional[int]:
        """向量维度（未知返回 None）。"""
        dim = getattr(self.backend, "dimension", None)
        if dim is None:
            dim = getattr(self.backend, "dim", None)
        return int(dim) if dim is not None else None

    @property
    def query_prefix(self) -> str:
        """当前查询前缀（评测"加/不加前缀"的召回对比时用）。"""
        return self.settings.embedding_query_prefix

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码文档。"""
        return self.backend.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """编码查询（含前缀）。"""
        return self.backend.embed_query(text)

    def describe(self) -> Dict[str, Any]:
        """能力描述（供 /health）。"""
        is_real = self.name == "sentence_transformers"
        return {
            "backend": self.name,
            "model": self.settings.embedding_model if is_real else "hash-fallback",
            "dimension": self.dimension,
            "query_prefix": self.query_prefix,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "device": self.settings.embedding_device if is_real else None,
        }


_service_singleton: Optional[EmbeddingService] = None


def get_embedding_service(reload: bool = False) -> EmbeddingService:
    """获取全局 Embedding 服务单例。"""
    global _service_singleton
    if _service_singleton is None or reload:
        _service_singleton = EmbeddingService()
    return _service_singleton


def reset_embedding_service() -> None:
    """清空单例（测试中切换后端时使用）。"""
    global _service_singleton
    _service_singleton = None
