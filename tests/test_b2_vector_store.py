"""B2 测试：向量存储与 Embedding（模块 2）。

覆盖四件事：
1. 写入 / 检索基本可用，``similarity_search(query, k=3)`` 返回 3 条；
2. **幂等**：同一份文件重复入库不产生重复向量（按确定性 chunk_id 覆盖）；
3. **文档级删除**：删除后再检索不到该文档，其他文档不受影响；
4. Embedding 的两个后端：hash 兜底具备确定性与字面相似性；查询前缀确实生效。

注意：测试统一使用 hash 后端，不加载 bge 模型（离线、快、确定性）。
真实 bge 的行为差异由 B6 评测报告量化。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.models import Chunk


def make_chunk(text: str, doc_id: str = "docA", index: int = 0, page: int | None = 1) -> Chunk:
    """构造测试用块。"""
    return Chunk(
        chunk_id=f"{doc_id}:{page or 0}:{index}",
        text=text,
        file_name="员工手册.pdf",
        doc_id=doc_id,
        page=page,
        chunk_index=index,
        char_len=len(text),
        created_at="2025-01-01 00:00:00",
    )


CORPUS = [
    ("公司的年假政策是入职满一年享有五天年假，满十年享有十天。", "docA", 1),
    ("病假需要提供二级以上医院开具的病假证明，病假期间工资按最低工资标准的百分之八十发放。", "docA", 2),
    ("报销标准：一线城市住宿每晚六百元，其他城市每晚四百元，餐补每人每天一百元。", "docB", 1),
    ("保密义务在劳动合同解除后继续有效三年，不得泄露技术资料与客户名单。", "docB", 2),
    ("离职需提前三十日以书面形式通知公司，交接完成后方可办理离职证明。", "docC", 1),
]


@pytest.fixture()
def filled_store(vector_store):
    """写入 5 个块的向量库。"""
    chunks = [make_chunk(text, doc_id, index) for index, (text, doc_id, _page) in enumerate(CORPUS)]
    vector_store.add_documents(chunks)
    return vector_store


# ---------------------------------------------------------------------------
# 写入与检索
# ---------------------------------------------------------------------------
def test_add_and_count(filled_store) -> None:
    """写入后计数正确。"""
    assert filled_store.count() == len(CORPUS)
    stats = filled_store.stats()
    assert stats["chunks"] == len(CORPUS)
    assert stats["backend"] == "chromadb"
    assert stats["space"] == "cosine"          # 必须显式使用余弦空间
    assert stats["embedding_backend"] == "hash"


def test_similarity_search_returns_k(filled_store) -> None:
    """``similarity_search(query, k=3)`` 返回块对象，条数受 k 限制。"""
    results = filled_store.similarity_search("年假有几天", k=3)
    assert len(results) == 3
    assert all(isinstance(item, Chunk) for item in results)


def test_search_with_scores_ranks_relevant_first(filled_store) -> None:
    """相关块必须排在前面，且分数落在 (0, 1]。"""
    results = filled_store.search_with_scores("年假 满一年 五天", k=5)
    assert results
    top = results[0]
    assert "年假" in top.chunk.text
    assert top.retriever == "vector"
    assert 0.0 < top.score <= 1.0
    # 分数按降序排列
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)
    # 元数据完整性（引用展示依赖这些字段）
    assert top.chunk.file_name == "员工手册.pdf"
    assert top.chunk.page == 1
    assert top.chunk.chunk_id


def test_search_min_score_filter(filled_store) -> None:
    """低于 min_score 的结果必须被过滤掉。"""
    all_results = filled_store.search_with_scores("年假", k=5)
    assert all_results
    highest = max(item.score for item in all_results)
    filtered = filled_store.search_with_scores("年假", k=5, min_score=highest + 0.01)
    assert filtered == []


def test_search_does_not_mutate_metadata(filled_store) -> None:
    """检索回来的块，page 为 None 的情况要正确还原（入库时用 -1 占位）。"""
    from rag.models import Chunk as ChunkModel

    no_page = ChunkModel(
        chunk_id="docX:0:0",
        text="Markdown 文档没有页码概念，这里是一段说明文字。",
        file_name="FAQ.md",
        doc_id="docX",
        page=None,
        chunk_index=0,
    )
    filled_store.add_documents([no_page])
    results = filled_store.search_with_scores("Markdown 页码", k=3)
    assert results
    target = next(item for item in results if item.chunk.doc_id == "docX")
    assert target.chunk.page is None           # -1 必须还原为 None，而不是展示成第 -1 页


def test_empty_store_returns_empty(vector_store) -> None:
    """空库检索返回空列表，不抛异常。"""
    assert vector_store.count() == 0
    assert vector_store.similarity_search("任意问题", k=3) == []
    assert vector_store.search_with_scores("任意问题") == []


def test_empty_query_returns_empty(filled_store) -> None:
    """空查询直接返回空，不浪费一次向量计算。"""
    assert filled_store.search_with_scores("") == []
    assert filled_store.search_with_scores("   ") == []


# ---------------------------------------------------------------------------
# 幂等与去重
# ---------------------------------------------------------------------------
def test_repeated_add_is_idempotent(vector_store) -> None:
    """同一批块重复写入不增加总数（chunk_id 确定性 → upsert 覆盖）。"""
    chunks = [make_chunk(text, doc_id, index) for index, (text, doc_id, _page) in enumerate(CORPUS)]
    vector_store.add_documents(chunks)
    assert vector_store.count() == len(CORPUS)

    # 再写一次（模拟同一文件重复上传后强制重新向量化）
    vector_store.add_documents(chunks)
    assert vector_store.count() == len(CORPUS), "重复写入必须是覆盖而不是追加"


def test_content_update_replaces_chunk(vector_store) -> None:
    """内容变化但 chunk_id 相同 → 覆盖为新内容（检索到的是最新文本）。"""
    original = make_chunk("旧内容：年假为五天。", doc_id="docA", index=0)
    vector_store.add_documents([original])

    updated = make_chunk("新内容：年假为七天，且可结转。", doc_id="docA", index=0)
    vector_store.add_documents([updated])

    assert vector_store.count() == 1
    results = vector_store.search_with_scores("年假 七天 结转", k=1)
    assert results
    assert "七天" in results[0].chunk.text


# ---------------------------------------------------------------------------
# 文档级操作
# ---------------------------------------------------------------------------
def test_has_document(filled_store) -> None:
    """文档存在性判断（入库幂等逻辑依赖它）。"""
    assert filled_store.has_document("docA") is True
    assert filled_store.has_document("docZ") is False
    assert filled_store.has_document("") is False


def test_existing_chunk_ids(filled_store) -> None:
    """能列出某文档的块 id，便于诊断重复。"""
    ids = filled_store.existing_chunk_ids("docA")
    assert len(ids) == 2
    assert all(item.startswith("docA:") for item in ids)


def test_delete_document(filled_store) -> None:
    """删除某文档的全部向量，其他文档不受影响。"""
    before = filled_store.count()
    removed = filled_store.delete_document("docA")
    assert removed == 2
    assert filled_store.count() == before - 2
    assert filled_store.has_document("docA") is False
    assert filled_store.has_document("docB") is True

    # 删除后相关内容的检索结果里不应再出现该文档
    results = filled_store.search_with_scores("年假 病假", k=5)
    assert all(item.chunk.doc_id != "docA" for item in results)


def test_delete_unknown_document_is_safe(filled_store) -> None:
    """删除不存在的文档返回 0，不抛异常。"""
    assert filled_store.delete_document("不存在") == 0
    assert filled_store.delete_document("") == 0


def test_reset_clears_store(filled_store) -> None:
    """清空向量库（重建索引流程的第一步）。"""
    filled_store.reset()
    assert filled_store.count() == 0


def test_persistence_across_instances(temp_db: Path, embedding_service, monkeypatch) -> None:
    """持久化：新建实例（模拟进程重启）后仍能检索到之前的数据。"""
    monkeypatch.setenv("CHROMA_DIR", str(temp_db.parent / "chroma_db"))

    from config import settings as settings_module

    settings_module.get_settings.cache_clear()
    from config.settings import get_settings
    from rag.store import VectorStore

    first = VectorStore(settings=get_settings(), embedding_service=embedding_service)
    first.add_documents([make_chunk("持久化测试：年假五天。", doc_id="persist", index=0)])
    assert first.count() == 1

    # 新实例指向同一目录
    second = VectorStore(settings=get_settings(), embedding_service=embedding_service)
    assert second.count() == 1
    results = second.search_with_scores("年假", k=1)
    assert results and results[0].chunk.doc_id == "persist"


# ---------------------------------------------------------------------------
# Embedding 后端
# ---------------------------------------------------------------------------
def test_hash_embeddings_deterministic(embedding_service) -> None:
    """hash 后端必须确定性：同文本永远同向量，且已归一化。"""
    service = embedding_service
    assert service.name == "hash"
    assert service.dimension == 512

    first = service.embed_documents(["年假政策说明"])[0]
    second = service.embed_documents(["年假政策说明"])[0]
    assert first == second

    norm = sum(value * value for value in first) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_hash_embeddings_literal_similarity(embedding_service) -> None:
    """字面越接近的文本，余弦相似度越高（保证检索在测试中可用）。"""
    service = embedding_service
    query = service.embed_query("年假")
    close = service.embed_documents(["年假政策：满一年五天"])[0]
    far = service.embed_documents(["报销标准：住宿每晚六百元"])[0]

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert cosine(query, close) > cosine(query, far)


def test_query_prefix_is_applied() -> None:
    """查询侧必须加前缀，文档侧不加——这是 bge 检索效果的关键。"""
    from rag.embeddings import HashEmbeddings

    prefix = "为这个句子生成表示以用于检索相关文章："
    with_prefix = HashEmbeddings(dim=64, prefix=prefix)
    without_prefix = HashEmbeddings(dim=64, prefix="")

    # 有前缀时，查询向量应当等价于"带前缀的文档向量"
    assert with_prefix.embed_query("年假") == with_prefix.embed_documents([f"{prefix}年假"])[0]
    # 无前缀时两者不同（说明前缀确实参与编码）
    assert without_prefix.embed_query("年假") != with_prefix.embed_query("年假")


def test_embedding_service_describe(embedding_service) -> None:
    """能力描述用于 /health，必须包含后端、维度与前缀信息。"""
    describe = embedding_service.describe()
    assert describe["backend"] == "hash"
    assert describe["dimension"] == 512
    assert describe["query_prefix"]
    assert describe["degraded"] is False


def test_explicit_sentence_transformers_backend_does_not_silently_degrade() -> None:
    """显式指定 sentence_transformers 时不允许静默降级。

    用一个**本地不存在的模型路径**触发加载失败：显式后端必须直接抛错，
    而不是悄悄换成 hash 后端（否则使用者会以为自己在用真实模型）。
    注意不使用线上模型名，避免测试触发真实下载。
    """
    from config.settings import Settings
    from rag.embeddings import EmbeddingService

    settings = Settings(
        _env_file=None,
        embedding_model=str(Path("不存在的本地模型目录")),
        embedding_cache_dir=str(Path("不存在的缓存目录")),
    )
    with pytest.raises(Exception):
        EmbeddingService(settings=settings, force_backend="sentence_transformers")


def test_auto_backend_degrades_to_hash_when_model_unavailable() -> None:
    """auto 模式下模型加载失败必须降级为 hash，并记录降级原因（降级可见）。"""
    from config.settings import Settings
    from rag.embeddings import EmbeddingService

    settings = Settings(
        _env_file=None,
        embedding_model=str(Path("不存在的本地模型目录")),
        embedding_cache_dir=str(Path("不存在的缓存目录")),
    )
    service = EmbeddingService(settings=settings, force_backend="auto")
    assert service.name == "hash"
    assert service.degraded is True
    assert "降级" in (service.degrade_reason or "")
    assert service.describe()["degraded"] is True
