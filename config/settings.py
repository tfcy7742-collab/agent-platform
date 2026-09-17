"""全局配置。

设计要点
--------
1. **全部配置外置**：LLM / Embedding / 检索 / Agent / Web / 存储 六类参数全部可通过
   环境变量或项目根目录的 ``.env`` 覆盖，代码里不出现任何魔法数字。
2. **配置快照可追溯**：启动时打印一份脱敏配置快照（API Key 只显示前 6 后 4 位），
   线上出问题时能一眼看出"当时用的是哪个模型、哪个阈值"。
3. **三种运行模式**：``LLM_MODE=online|offline|auto`` 决定是否调用大模型；
   离线模式下 Planner 走规则路由、回答走模板拼接，保证无 Key 环境（含 CI）也能跑通全链路。
4. **能力降级可见**：Embedding 支持 ``hash`` 兜底后端，向量库/BM25/Reranker 各有开关，
   当前哪些能力可用由 ``capabilities()`` 统一汇总给 ``/health``。

环境变量读取优先级：真实环境变量 > 项目根 .env > 字段默认值
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# 路径常量：全部以项目根目录为基准，避免受当前工作目录影响
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """平台运行期配置（字段名与 .env 中的键一一对应，大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 允许 .env 中存在本项目不认识的键（例如给别的工具用）
    )

    # ==================================================================
    # 1. LLM
    # ==================================================================
    llm_provider: Literal["deepseek", "dashscope"] = "deepseek"
    llm_mode: Literal["auto", "online", "offline"] = "auto"
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # DeepSeek（默认 provider）
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"

    # 阿里云百炼（OpenAI 兼容，切换 provider 时使用）
    dashscope_api_key: Optional[str] = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-plus"

    # ==================================================================
    # 2. Embedding
    # ==================================================================
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    # bge 系列做检索时，查询侧必须加此前缀（doc 侧不加）。留空则不添加，用于评测对比。
    embedding_query_prefix: str = "为这个句子生成表示以用于检索相关文章："
    embedding_backend: Literal["auto", "sentence_transformers", "hash"] = "auto"
    embedding_cache_dir: str = "./data/models"
    embedding_dim: int = 512  # hash 兜底后端的向量维度（bge-small-zh 为 512）

    # ==================================================================
    # 3. 检索
    # ==================================================================
    retrieve_top_k: int = 3          # 最终交给 LLM 的片段数
    retrieve_candidates: int = 20    # 每路召回的候选数
    enable_bm25: bool = True
    enable_rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 10
    rrf_k: int = 60                  # RRF 平滑常数
    refuse_threshold: float = 0.35   # 融合分数低于此值直接拒答（不调用 LLM）
    enable_citation_check: bool = True
    retrieve_cache_size: int = 200   # 检索结果 LRU 缓存条数

    # ==================================================================
    # 4. 文档处理
    # ==================================================================
    chunk_size: int = 500
    chunk_overlap: int = 50
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 50
    allowed_extensions: str = ".pdf,.docx,.txt,.md"
    chroma_dir: str = "./chroma_db"
    collection_name: str = "knowledge_base"
    # 为每个 VectorStore 实例创建独立的 chromadb 客户端。
    # 默认关闭（沿用 langchain-chroma 的默认行为，生产更省资源）；
    # 测试环境会打开——chromadb 的 PersistentClient 按路径全局缓存，
    # 同一进程里复用同一路径的不同库时会拿到旧客户端对象（踩过的坑）。
    chroma_fresh_client: bool = False

    # ==================================================================
    # 5. Agent
    # ==================================================================
    agent_max_steps: int = 4
    agent_max_context_chars: int = 8000
    agent_history_turns: int = 6
    enable_query_rewrite: bool = True
    agent_stream: bool = True
    # 工具执行超时。行程规划工具内部要跑 4 个智能体、开启大模型时可能多次调用模型，
    # 实测单次 20~120 秒不等（取决于模型速度与网络），因此单独给它一个可配置的宽松上限；
    # 其余工具用默认值即可。
    tool_timeout_s: float = 60.0
    trip_planner_timeout_s: float = 300.0

    # ==================================================================
    # 6. Web
    # ==================================================================
    host: str = "127.0.0.1"
    port: int = 8000
    ui_mount_path: str = "/ui"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ==================================================================
    # 7. 存储与日志
    # ==================================================================
    db_path: str = "./data/agent_platform.db"
    log_level: str = "INFO"
    log_json: bool = False
    trace_enabled: bool = True

    # ------------------------------------------------------------------
    # 校验器
    # ------------------------------------------------------------------
    @field_validator("refuse_threshold")
    @classmethod
    def _check_threshold(cls, value: float) -> float:
        """拒答阈值必须落在 (0, 1) 区间，否则拒答行为会完全失效。"""
        if not 0.0 < value < 1.0:
            raise ValueError("REFUSE_THRESHOLD 必须位于 (0, 1) 之间")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _check_overlap(cls, value: int, info: Any) -> int:
        """切块重叠必须小于块大小，否则 splitter 会无限循环。"""
        chunk_size = info.data.get("chunk_size", 500)
        if value >= chunk_size:
            raise ValueError(f"CHUNK_OVERLAP({value}) 必须小于 CHUNK_SIZE({chunk_size})")
        return value

    @field_validator("agent_max_steps")
    @classmethod
    def _check_steps(cls, value: int) -> int:
        """步数预算至少 1，且不超过 12（防止无意义的长循环）。"""
        if not 1 <= value <= 12:
            raise ValueError("AGENT_MAX_STEPS 必须位于 1~12 之间")
        return value

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def provider_api_key(self) -> Optional[str]:
        """当前 provider 对应的 API Key。"""
        return self.dashscope_api_key if self.llm_provider == "dashscope" else self.deepseek_api_key

    @property
    def provider_base_url(self) -> str:
        """当前 provider 的 OpenAI 兼容地址。"""
        return self.dashscope_base_url if self.llm_provider == "dashscope" else self.deepseek_base_url

    @property
    def provider_model(self) -> str:
        """当前 provider 的模型名。"""
        return self.dashscope_model if self.llm_provider == "dashscope" else self.deepseek_model

    @property
    def has_api_key(self) -> bool:
        """是否配置了有效（非占位符）的 API Key。"""
        key = (self.provider_api_key or "").strip()
        if not key:
            return False
        placeholders = {"your_key_here", "sk-xxx", "sk-your-key", "none", "null", "test", "changeme"}
        return key.lower() not in placeholders

    @property
    def llm_online(self) -> bool:
        """最终是否真的会调用大模型（综合 LLM_MODE 与 Key 配置）。"""
        if self.llm_mode == "offline":
            return False
        if self.llm_mode == "online":
            # online 模式下即使没 Key 也返回 True，让调用方拿到明确的报错而不是静默降级
            return True
        return self.has_api_key  # auto

    @property
    def masked_key(self) -> str:
        """脱敏后的 Key，用于日志与 /health 展示。"""
        key = (self.provider_api_key or "").strip()
        if not key:
            return "(未配置)"
        return f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"

    @property
    def extension_list(self) -> List[str]:
        """允许上传的后缀列表（统一小写、带点）。"""
        return [item.strip().lower() for item in self.allowed_extensions.split(",") if item.strip()]

    @property
    def cors_origin_list(self) -> List[str]:
        """CORS 白名单列表。"""
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    # 路径类字段统一解析为绝对路径（相对项目根目录）
    def _abs(self, relative: str) -> Path:
        """把配置里的相对路径解析为基于项目根目录的绝对路径。"""
        path = Path(relative)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def upload_path(self) -> Path:
        return self._abs(self.upload_dir)

    @property
    def chroma_path(self) -> Path:
        return self._abs(self.chroma_dir)

    @property
    def db_file(self) -> Path:
        return self._abs(self.db_path)

    @property
    def model_cache_path(self) -> Path:
        return self._abs(self.embedding_cache_dir)

    # ------------------------------------------------------------------
    # 能力汇总与脱敏快照
    # ------------------------------------------------------------------
    def capabilities(self) -> Dict[str, Any]:
        """汇总当前各项能力的可用状态，供 /health 展示（降级必须可见）。"""
        from core import __version__  # 延迟导入：避免 config 与 core 之间的循环依赖

        return {
            "version": __version__,
            "llm": {
                "provider": self.llm_provider,
                "model": self.provider_model,
                "base_url": self.provider_base_url,
                "api_key": self.masked_key,
                "online": self.llm_online,
                "mode": self.llm_mode,
            },
            "embedding": {
                "model": self.embedding_model,
                "backend": self.embedding_backend,
                "query_prefix": bool(self.embedding_query_prefix),
            },
            "retrieval": {
                "top_k": self.retrieve_top_k,
                "bm25": self.enable_bm25,
                "rerank": self.enable_rerank,
                "refuse_threshold": self.refuse_threshold,
                "citation_check": self.enable_citation_check,
            },
            "agent": {
                "max_steps": self.agent_max_steps,
                "query_rewrite": self.enable_query_rewrite,
                "stream": self.agent_stream,
            },
            "storage": {
                "chroma_dir": str(self.chroma_path),
                "db_path": str(self.db_file),
                "upload_dir": str(self.upload_path),
            },
        }

    def masked_snapshot(self) -> Dict[str, Any]:
        """启动日志用的脱敏配置快照（不含任何密钥明文）。"""
        return {
            "llm_provider": self.llm_provider,
            "llm_mode": self.llm_mode,
            "llm_online": self.llm_online,
            "model": self.provider_model,
            "base_url": self.provider_base_url,
            "api_key": self.masked_key,
            "embedding_model": self.embedding_model,
            "embedding_backend": self.embedding_backend,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieve_top_k": self.retrieve_top_k,
            "enable_bm25": self.enable_bm25,
            "enable_rerank": self.enable_rerank,
            "refuse_threshold": self.refuse_threshold,
            "agent_max_steps": self.agent_max_steps,
            "host": f"{self.host}:{self.port}",
            "db_path": str(self.db_file),
            "chroma_dir": str(self.chroma_path),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一配置实例（进程内缓存，避免重复解析 .env）。"""
    return Settings()


def ensure_directories() -> None:
    """创建运行所需的目录（上传目录 / Chroma 目录 / 模型缓存 / 数据库目录）。"""
    settings = get_settings()
    for path in (
        settings.upload_path,
        settings.chroma_path,
        settings.model_cache_path,
        settings.db_file.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# HuggingFace 可达性兜底
# ---------------------------------------------------------------------------
def _host_reachable(url: str, timeout: float = 5.0) -> bool:
    """探测一个 URL 是否可连通（不抛异常，只返回布尔值）。"""
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "agent-platform/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 400
    except Exception:  # noqa: BLE001 - 网络问题类型很多，统一视为不可达
        return False


def _model_is_cached(model_name: str, cache_dir: Path) -> bool:
    """判断模型是否已在本地缓存（HuggingFace 的 snapshots 目录布局）。

    检查关键文件是否存在。只要权重与配置都在，就可以完全离线加载，
    不必再向 huggingface.co 发任何请求。
    """
    if not cache_dir.exists():
        return False
    folder = "models--" + model_name.replace("/", "--")
    snapshots = cache_dir / folder / "snapshots"
    if not snapshots.exists():
        return False
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        has_weights = any(
            (snapshot / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin", "model.onnx")
        )
        if has_weights and (snapshot / "config.json").exists():
            return True
    return False


def ensure_model_endpoint(timeout: float = 5.0, settings: Optional[Settings] = None) -> Optional[str]:
    """确保首次加载模型时不会因为"官方站不可达"而卡住或报错。

    按优先级依次处理三种情况：

    1. **模型已在本地缓存** → 直接置 ``HF_HUB_OFFLINE=1``，彻底跳过网络请求。
       这一步很关键：国内网络访问 huggingface.co 会**连接超时后重试 5 次**
       （每次递增等待），实测每个文件耗时 20~80 秒，多个文件叠加会让
       "加载模型"变成几分钟的卡顿——而模型其实早就在本地了。
    2. 用户显式设置了 ``HF_ENDPOINT`` → 尊重，不做探测。
    3. 官方站可达 → 直接返回；不可达 → 自动切到镜像 ``hf-mirror.com``
       （可用 ``HF_MIRROR_ENDPOINT`` 覆盖），并写回环境变量。

    Returns:
        实际生效的 endpoint；完全离线或无法确定时返回 ``None``。
    """
    import os

    settings = settings or get_settings()
    if _model_is_cached(settings.embedding_model, settings.model_cache_path):
        # 关键：模型已在本地，**彻底关掉联网尝试**。
        # 两个变量都要设：huggingface_hub 用 HF_HUB_OFFLINE，transformers 用 TRANSFORMERS_OFFLINE。
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import logging

        logging.getLogger(__name__).info(
            "模型已在本地缓存，跳过所有联网检查：%s", settings.model_cache_path,
        )
        return None

    if os.environ.get("HF_ENDPOINT"):
        return os.environ["HF_ENDPOINT"]

    official = "https://huggingface.co"
    if _host_reachable(f"{official}/api/models/BAAI/bge-small-zh-v1.5", timeout=timeout):
        return official

    mirror = os.environ.get("HF_MIRROR_ENDPOINT", "https://hf-mirror.com")
    if _host_reachable(f"{mirror}/api/models/BAAI/bge-small-zh-v1.5", timeout=timeout):
        os.environ["HF_ENDPOINT"] = mirror
        import logging

        logging.getLogger(__name__).warning(
            "huggingface.co 不可达，已自动切换到镜像 %s 下载模型（模型尚未缓存）。"
            "如需禁用该行为，请显式设置 HF_ENDPOINT。",
            mirror,
        )
        return mirror
    return None
