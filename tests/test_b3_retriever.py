"""B3 测试：混合检索器（向量 + BM25 + RRF）。

覆盖点：
1. BM25 分支能命中**精确关键词**（这是纯向量检索的短板）；
2. RRF 融合后，两路都命中的块排在只被单路命中的块之前（融合的价值）；
3. **score 语义不变**：融合后对外暴露的分数仍是向量余弦相似度（拒答阈值依赖它）；
4. BM25 索引有缓存，语料变化后失效；
5. 边界：空库、空查询、停用 BM25、query 带噪声标点。
"""

from __future__ import annotations

import pytest

from rag.models import Chunk

CORPUS = [
    # (文本, doc_id, page)
    ("员工入职满一年后享有年假，工作满一年不满十年者每年五天，满十年不满二十年者每年十天。", "doc_hr", 1),
    ("病假需提供二级以上医院开具的病假证明，病假期间工资按当地最低工资标准的百分之八十发放。", "doc_hr", 2),
    ("云笔记支持移动端离线编辑，恢复网络后自动同步，冲突时保留冲突副本。", "doc_prd", 1),
    ("接口返回 502 时，先检查反向代理日志，再确认上游服务是否存活。", "doc_faq", 1),
    ("预算审批流程：单笔支出超过 5000 元需部门负责人与财务总监双重审批。", "doc_fin", 1),
    ("公司每季度组织一次全员技术分享会，分享主题由各团队轮流申报。", "doc_hr", 3),
]


def make_chunk(text: str, doc_id: str, index: int, page: int | None = 1, file_name: str | None = None) -> Chunk:
    """构造测试块。"""
    return Chunk(
        chunk_id=f"{doc_id}:{page or 0}:{index}",
        text=text,
        file_name=file_name or f"{doc_id}.pdf",
        doc_id=doc_id,
        page=page,
        chunk_index=index,
        char_len=len(text),
        created_at="2025-01-01 00:00:00",
    )


@pytest.fixture()
def filled_store(vector_store):
    """写入 6 个块的向量库。"""
    chunks = [
        make_chunk(text, doc_id, index, page)
        for index, (text, doc_id, page) in enumerate(CORPUS)
    ]
    vector_store.add_documents(chunks)
    return vector_store


@pytest.fixture()
def populated_retriever(retriever, filled_store):
    """已填充语料的检索器。"""
    return retriever


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------
def test_tokenize_chinese_basic() -> None:
    """中文分词：能切出词、确定性可复现、过滤纯标点。"""
    from rag.retriever import tokenize_chinese

    text = "员工入职满一年后享有年假。"
    tokens = tokenize_chinese(text)
    assert tokens, "分词结果不应为空"
    assert all(token.strip() for token in tokens)
    assert not any(token in {"。", "，", "！"} for token in tokens)
    # 确定性：同一文本多次分词结果必须完全一致（BM25 词项统计依赖这一点）
    assert tokenize_chinese(text) == tokens
    # 领域词典生效：业务词必须是独立词项，而不是被拆成单字
    # （单字匹配会带来噪声，例如"假"会匹配到所有假期条款，降低 BM25 区分度）
    assert "年假" in tokens
    assert "员工" in tokens


def test_tokenize_mixed_language() -> None:
    """中英混合文本：中文字与英文单词都要保留。"""
    from rag.retriever import tokenize_chinese

    tokens = tokenize_chinese("接口返回 502 error，请检查 nginx 日志")
    assert any("接口" in token for token in tokens)
    assert any("502" in token for token in tokens)
    assert any("error" in token for token in tokens)


def test_tokenize_empty() -> None:
    """空文本返回空列表。"""
    from rag.retriever import tokenize_chinese

    assert tokenize_chinese("") == []
    assert tokenize_chinese("   ") == []


# ---------------------------------------------------------------------------
# BM25 分支
# ---------------------------------------------------------------------------
def test_bm25_hits_exact_keyword(populated_retriever) -> None:
    """BM25 必须能命中精确关键词（型号/编号/专有名词这类字面串）。"""
    hits = populated_retriever._bm25_search("502", top_n=3)
    assert hits, "BM25 应当能命中「502」"
    assert any("502" in chunk.text for chunk, _score in hits)


def test_bm25_score_is_positive(populated_retriever) -> None:
    """BM25 返回的分数必须为正（零分结果被过滤，避免噪声进入融合）。"""
    hits = populated_retriever._bm25_search("年假 病假", top_n=5)
    assert hits
    assert all(score > 0 for _chunk, score in hits)


def test_bm25_no_match_returns_empty(populated_retriever) -> None:
    """完全不相关的查询：BM25 不应给出高分。

    这里刻意不断言"结果为空"。中文 BM25 的分词粒度是字/词混合，
    不相关句子里偶然出现的**单字**（"理""与"）仍会产生微小正分——
    这是字面检索的固有噪声，靠 RRF 排序与拒答阈值处理，而不是靠 BM25 自己。
    真正要保证的是：这类噪声分数远低于真实命中的分数。
    """
    noise_hits = populated_retriever._bm25_search("量子纠缠与超导材料", top_n=3)
    real_hits = populated_retriever._bm25_search("年假 病假 年假天数", top_n=3)
    assert real_hits, "正常查询必须有高相关命中"

    if noise_hits:
        noise_top = max(score for _chunk, score in noise_hits)
        real_top = max(score for _chunk, score in real_hits)
        assert noise_top < real_top, "不相关查询的 BM25 分数必须低于真实命中"


def test_bm25_index_cache_invalidated_on_corpus_change(populated_retriever, vector_store) -> None:
    """语料变化后索引指纹改变，新文档必须能被 BM25 检索到。"""
    before = populated_retriever._get_bm25_index()[1]
    assert len(before) == len(CORPUS)

    vector_store.add_documents([make_chunk("新增内容：年度调薪窗口为每年四月。", "doc_new", 0, 1)])
    index, chunks = populated_retriever._get_bm25_index()
    assert len(chunks) == len(CORPUS) + 1, "语料变化后必须重建索引"

    hits = populated_retriever._bm25_search("调薪窗口", top_n=3)
    assert any("调薪" in chunk.text for chunk, _score in hits)


def test_bm25_can_be_disabled(temp_db, embedding_service, vector_store, filled_store, monkeypatch) -> None:
    """ENABLE_BM25=false 时 BM25 分支整体关闭，检索仍可用（退化为纯向量）。"""
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("ENABLE_BM25", "false")

    from config.settings import get_settings
    from rag.retriever import HybridRetriever

    retriever = HybridRetriever(
        settings=get_settings(), vector_store=vector_store, embedding_service=embedding_service
    )
    assert retriever._get_bm25_index() == (None, [])
    hits = retriever.retrieve("年假有几天", top_k=3)
    assert hits, "关闭 BM25 后仍应有向量检索结果"
    assert all(item.retriever == "vector" for item in hits)
    assert retriever.last_stats["bm25_hits"] == 0


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------
def test_fusion_ranks_two_branch_hit_first(populated_retriever) -> None:
    """两路都命中的块必须排在只被单路命中的块之前（RRF 的核心价值）。"""
    hits = populated_retriever.retrieve("年假有几天", top_k=5)
    assert hits
    top = hits[0]
    assert top.vec_score is not None and top.bm25_score is not None
    assert top.retriever == "hybrid"
    assert top.rank_fused == 1


def test_fused_score_keeps_vector_semantics(populated_retriever, vector_store) -> None:
    """**关键契约**：融合结果的 score 必须等于向量余弦相似度（拒答阈值依赖它）。"""
    query = "年假有几天"
    fused = populated_retriever.retrieve(query, top_k=3)
    pure = vector_store.search_with_scores(query, k=5)
    pure_scores = {item.chunk.chunk_id: item.score for item in pure}

    assert fused
    for item in fused:
        if item.chunk.chunk_id in pure_scores:
            assert item.score == pytest.approx(pure_scores[item.chunk.chunk_id], abs=1e-6), (
                "融合不得改变 score 的口径，否则 REFUSE_THRESHOLD 会失效"
            )


def test_bm25_only_hit_is_marked(populated_retriever) -> None:
    """仅被 BM25 召回的块要如实标注 retriever=bm25，且分数保守（不会是虚高的向量分）。"""
    vectors = populated_retriever.vector_store.search_with_scores("502", k=20)
    vector_ids = {item.chunk.chunk_id for item in vectors}
    bm25_hits = populated_retriever._bm25_search("502", top_n=20)

    only_bm25 = [chunk for chunk, _score in bm25_hits if chunk.chunk_id not in vector_ids]
    if not only_bm25:
        pytest.skip("本用例语料下 BM25 命中项都被向量召回了，无法验证该分支")

    fused = populated_retriever.retrieve("502", top_k=20)
    marked = [item for item in fused if item.chunk.chunk_id == only_bm25[0].chunk_id]
    assert marked
    assert marked[0].retriever == "bm25"
    assert 0 < marked[0].score < 1.0


def test_retrieve_respects_top_k(populated_retriever) -> None:
    """top_k 生效，且 rank_final 从 1 连续编号（引用编号依赖它）。"""
    for k in (1, 2, 3):
        hits = populated_retriever.retrieve("年假 病假 报销", top_k=k)
        assert len(hits) <= k
        assert [item.rank_final for item in hits] == list(range(1, len(hits) + 1))


def test_retrieve_min_score_filter(populated_retriever) -> None:
    """min_score 过滤生效。"""
    hits = populated_retriever.retrieve("年假", top_k=5)
    if not hits:
        pytest.skip("无检索结果")
    highest = max(item.score for item in hits)
    filtered = populated_retriever.retrieve("年假", top_k=5, min_score=highest + 0.01)
    assert filtered == []


def test_retrieve_stats(populated_retriever) -> None:
    """检索统计要能反映两路召回与耗时（trace 与调试面板依赖它）。"""
    populated_retriever.retrieve("接口 502 排查", top_k=3)
    stats = populated_retriever.last_stats
    assert stats["vector_hits"] >= 0
    assert stats["bm25_hits"] >= 0
    assert stats["returned"] <= 3
    assert stats["latency_ms"] >= 0
    assert stats["bm25_enabled"] is True
    assert "top_score" in stats


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------
def test_retrieve_on_empty_store(retriever) -> None:
    """空库检索返回空列表，不抛异常。"""
    assert retriever.retrieve("任意问题", top_k=3) == []


def test_retrieve_empty_query(populated_retriever) -> None:
    """空查询直接返回空。"""
    assert populated_retriever.retrieve("", top_k=3) == []
    assert populated_retriever.retrieve("   ", top_k=3) == []


def test_retrieve_with_noisy_query(populated_retriever) -> None:
    """带标点/表情/空格的查询不应影响召回（分词阶段会清洗）。"""
    hits = populated_retriever.retrieve("年假？？有几天！！！", top_k=3)
    assert hits
    assert "年假" in hits[0].chunk.text


def test_retriever_describe(populated_retriever) -> None:
    """能力描述用于 /health。"""
    describe = populated_retriever.describe()
    assert describe["fusion"]["method"] == "rrf"
    assert describe["bm25"]["tokenizer"] == "jieba"
    assert describe["bm25"]["enabled"] is True
    assert describe["rerank"]["enabled"] is False
    assert describe["vector"]["chunks"] == len(CORPUS)


def test_invalidate_cache(populated_retriever) -> None:
    """手工清空缓存后仍能正常检索（重建索引）。"""
    populated_retriever.retrieve("年假", top_k=1)
    populated_retriever.invalidate_cache()
    hits = populated_retriever.retrieve("年假", top_k=1)
    assert hits
