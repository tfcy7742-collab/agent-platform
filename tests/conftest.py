"""pytest 全局配置与共用 fixture。

关键设计：**所有测试都在离线模式下运行**（LLM_MODE=offline）。
这样无 API Key 的机器与 CI 都能跑回归，不会因为网络或额度问题变红。
真实模型的端到端验证放在手工验收与 ``eval/`` 评测里。
"""

from __future__ import annotations

import importlib
import itertools
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator

import pytest

# 项目根目录入 sys.path，保证 `import app` / `from core... import ...` 可用
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 测试临时目录放在**项目内的隐藏目录**，而不是系统临时目录。
# 原因：部分受管环境（沙箱 / 受控容器）只允许写工作区，系统 TEMP 会被拒绝，
# 导致 pytest 的 tmp_path fixture 直接报 PermissionError。放在项目内可以绕开该限制。
#
# 目录名带 pid：受管环境里 rmtree 可能失败，残留目录会被下次运行复用，
# 而 Chroma 的数据是持久化的——复用目录会让"上次测试遗留的向量"污染本次断言。
# 每次进程一个独立命名空间可以从根上避免这类脏数据。
_TMP_ROOT = PROJECT_ROOT / ".pytest_tmp"
_TMP_SESSION = f"run{os.getpid()}"
_tmp_counter = itertools.count()


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """覆盖 pytest 内置的 tmp_path，改为在项目内创建临时目录。

    每个测试一个独立子目录；测试结束后尽力清理（清理失败不影响测试结果）。
    """
    path = _TMP_ROOT / _TMP_SESSION / f"case{next(_tmp_counter):03d}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _offline_env() -> Iterator[None]:
    """整个测试会话强制离线，避免任何真实网络调用。"""
    original = dict(os.environ)
    os.environ["LLM_MODE"] = "offline"
    os.environ["EMBEDDING_BACKEND"] = "hash"   # 不加载真实模型，测试跑得快
    os.environ["LOG_LEVEL"] = "WARNING"
    # 禁止任何 HuggingFace 下载尝试：离线环境里下载会挂起几十秒才失败，
    # 让"降级路径"这类测试变得极慢。真正需要模型的场景由手工验收与评测覆盖。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.pop("DEEPSEEK_API_KEY", None)   # 确保不会被误认为已配置 Key
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    yield
    os.environ.clear()
    os.environ.update(original)
    # 尽力清理本次会话的临时目录；受管环境下可能失败，忽略即可（已在 .gitignore 中排除）
    shutil.rmtree(_TMP_ROOT / _TMP_SESSION, ignore_errors=True)


@pytest.fixture()
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """把数据库指向临时文件，并清空配置缓存。

    同时重置 ``infra.db`` 的线程本地连接，避免上一个测试的连接泄漏到下一个。
    """
    db_file = tmp_path / "test_agent_platform.db"
    monkeypatch.setenv("DB_PATH", str(db_file))

    from config import settings as settings_module
    from infra import db as db_module

    settings_module.get_settings.cache_clear()
    db_module.close_connection()
    db_module.init_db()

    yield db_file

    db_module.close_connection()
    settings_module.get_settings.cache_clear()


@pytest.fixture()
def client(temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """FastAPI 测试客户端（带 lifespan，走真实的启动/关闭钩子）。

    通过 reload 让 ``app`` 模块在临时环境下重新构建，确保 lifespan 使用测试配置；
    同时重置 RAG 侧单例，避免上一个测试的向量库/Embedding 状态泄漏进来。
    """
    monkeypatch.setenv("UPLOAD_DIR", str(temp_db.parent / "uploads"))
    monkeypatch.setenv("CHROMA_DIR", str(temp_db.parent / "chroma_db"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")   # 不加载真实模型
    # 每个测试用独立的临时 Chroma 目录；开启独立客户端避免 chromadb
    # 按路径全局缓存导致的"拿到上一个库的客户端"问题（详见 rag/store.py）
    monkeypatch.setenv("CHROMA_FRESH_CLIENT", "true")

    from config import settings as settings_module
    from infra import db as db_module
    from rag import answer as answer_module
    from rag import embeddings as embeddings_module
    from rag import pipeline as pipeline_module
    from rag import retriever as retriever_module
    from rag import store as store_module

    def _reset_singletons() -> None:
        """重置 RAG 侧与 Agent 侧全部单例。

        每个测试使用独立的临时 Chroma 目录，如果单例（向量库 / 检索器 / 引擎 /
        工具注册表 / Agent 运行时）跨测试复用，就会指向**上一个测试的库**，
        表现为"刚上传却检索不到"。这里统一重置，保证测试之间完全隔离。
        """
        embeddings_module.reset_embedding_service()
        store_module.reset_vector_store_singleton()
        pipeline_module.reset_ingest_pipeline()
        retriever_module.reset_retriever_singleton()
        answer_module.reset_rag_engine()

        from core.runtime.agent import reset_agent_runtime
        from core.runtime.executor import reset_executor
        from core.tools.registry import reset_registry

        reset_agent_runtime()
        reset_executor()
        reset_registry()

    settings_module.get_settings.cache_clear()
    db_module.close_connection()
    _reset_singletons()

    import app as app_module

    importlib.reload(app_module)

    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as test_client:
        yield test_client

    db_module.close_connection()
    settings_module.get_settings.cache_clear()
    _reset_singletons()


@pytest.fixture()
def settings_factory(monkeypatch: pytest.MonkeyPatch):
    """按给定环境变量构造一份新的 Settings（用于配置校验测试）。

    用法::

        settings = settings_factory(REFUSE_THRESHOLD="0.5")
    """
    from config.settings import Settings

    def _factory(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return Settings(_env_file=None)  # 忽略 .env，只认显式传入的环境变量

    return _factory


# ---------------------------------------------------------------------------
# B2 起共用的 RAG 组件 fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def embedding_service(temp_db: Path, monkeypatch: pytest.MonkeyPatch):
    """强制使用 hash 兜底后端的 Embedding 服务。

    测试环境不下载 bge 模型（体积大、需要网络），hash 后端是确定性的，
    完全满足"检索链路、阈值、幂等、删除"这些行为验证。
    """
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")

    from rag.embeddings import EmbeddingService

    return EmbeddingService(force_backend="hash")


@pytest.fixture()
def vector_store(temp_db: Path, embedding_service, monkeypatch: pytest.MonkeyPatch):
    """使用临时目录的向量库（每个测试独立，互不污染）。"""
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("CHROMA_DIR", str(temp_db.parent / "chroma_db"))

    from config.settings import get_settings
    from rag.store import VectorStore

    return VectorStore(settings=get_settings(), embedding_service=embedding_service)


@pytest.fixture()
def ingest_pipeline(temp_db: Path, embedding_service, vector_store, monkeypatch: pytest.MonkeyPatch):
    """入库管道（复用同一个向量库实例，便于断言）。"""
    monkeypatch.setenv("UPLOAD_DIR", str(temp_db.parent / "uploads"))

    from config import settings as settings_module

    settings_module.get_settings.cache_clear()

    from config.settings import get_settings
    from rag.pipeline import IngestPipeline

    return IngestPipeline(
        settings=get_settings(),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


# ---------------------------------------------------------------------------
# B3 起共用：检索器 / 问答引擎
# ---------------------------------------------------------------------------
@pytest.fixture()
def retriever(temp_db: Path, embedding_service, vector_store, monkeypatch: pytest.MonkeyPatch):
    """混合检索器（hash 向量 + BM25，全部离线）。"""
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("ENABLE_BM25", "true")
    monkeypatch.setenv("ENABLE_RERANK", "false")

    from config.settings import get_settings
    from rag.retriever import HybridRetriever, reset_retriever_singleton

    reset_retriever_singleton()
    return HybridRetriever(
        settings=get_settings(),
        vector_store=vector_store,
        embedding_service=embedding_service,
    )


@pytest.fixture()
def rag_engine(temp_db: Path, embedding_service, vector_store, retriever, monkeypatch: pytest.MonkeyPatch):
    """RAG 问答引擎（强制离线：不调用大模型，走片段摘要）。

    大模型相关的分支（LLM 拒答、引用校验）由测试直接替换 ``llm.chat_json`` 来覆盖，
    这样既不需要网络，也能精确构造边界情况。
    """
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("LLM_MODE", "offline")

    from config.settings import get_settings
    from core.llm import LLMClient
    from rag.answer import RagEngine, reset_rag_engine

    reset_rag_engine()
    settings = get_settings()
    return RagEngine(
        settings=settings,
        retriever=retriever,
        llm=LLMClient(settings=settings, enable=False),   # 强制离线
        vector_store=vector_store,
    )
