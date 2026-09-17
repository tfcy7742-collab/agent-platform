"""B3 测试：三层拒答与引用校验（验收标准「文档外问题必须稳定拒答」的守护测试）。

三层拒答分别覆盖：

============  ============================================  ==========================
层            触发条件                                       本文件用例
============  ============================================  ==========================
L1 知识库为空  向量库里一个块都没有                             test_refuse_when_store_empty
L2 阈值短路    检索最高分 < REFUSE_THRESHOLD（不调用大模型）     test_refuse_below_threshold_*
L3 模型判定    模型返回 confident=false / 标准话术              test_refuse_when_llm_uncertain
L3 引用校验    回答引用了不存在的编号                           test_refuse_on_invalid_citation
============  ============================================  ==========================

另外覆盖：离线模式摘要、拒答话术常量唯一性、引用编号解析的边界情况。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from core.prompts import REFUSAL_MESSAGE
from rag.models import Chunk

CORPUS = [
    ("员工入职满一年后享有年假，工作满一年不满十年者每年五天，满十年不满二十年者每年十天。", "doc_hr", 1),
    ("病假需提供二级以上医院开具的病假证明，病假期间工资按当地最低工资标准的百分之八十发放。", "doc_hr", 2),
    ("云笔记支持移动端离线编辑，恢复网络后自动同步，冲突时保留冲突副本。", "doc_prd", 1),
]


@pytest.fixture()
def seeded_store(vector_store):
    """写入 3 个块，供拒答与引用测试使用。"""
    vector_store.add_documents(
        [
            Chunk(
                chunk_id=f"{doc_id}:{page}:{index}",
                text=text,
                file_name=f"{doc_id}.pdf",
                doc_id=doc_id,
                page=page,
                chunk_index=index,
                char_len=len(text),
                created_at="2025-01-01 00:00:00",
            )
            for index, (text, doc_id, page) in enumerate(CORPUS)
        ]
    )
    return vector_store


@pytest.fixture(autouse=True)
def _cleanup_llm_patch(rag_engine):
    """确保每个用例结束后停止 PropertyMock，避免污染其他测试。"""
    yield
    patcher = getattr(rag_engine, "_llm_patcher", None)
    if patcher is not None:
        patcher.stop()
        rag_engine._llm_patcher = None


# ---------------------------------------------------------------------------
# L1：知识库为空
# ---------------------------------------------------------------------------
def test_refuse_when_store_empty(rag_engine) -> None:
    """知识库为空时必须拒答，且不调用大模型。"""
    result = rag_engine.answer("年假有几天")
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.refuse_reason == "no_documents"
    assert result.sources == []
    assert result.total_tokens == 0            # 说明确实没有调用大模型


def test_refuse_empty_question(rag_engine, seeded_store) -> None:
    """空问题拒答（防御性，API 层已做校验）。"""
    result = rag_engine.answer("   ")
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.refuse_reason == "empty_question"


# ---------------------------------------------------------------------------
# L2：阈值短路（不调用大模型）
# ---------------------------------------------------------------------------
def test_refuse_below_threshold_short_circuits(rag_engine, seeded_store) -> None:
    """阈值设为 1.0 时，任何问题都会被短路拒答，且 token 消耗为 0。"""
    result = rag_engine.answer("年假有几天", threshold=1.0)
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.refuse_reason == "below_threshold"
    assert result.total_tokens == 0, "阈值短路必须发生在调用大模型之前"
    assert result.meta["threshold"] == 1.0
    # 仍然返回检索到的候选片段，便于人工核实"是不是真的没资料"
    assert result.sources


def test_refuse_below_threshold_for_out_of_domain(rag_engine, seeded_store) -> None:
    """文档外问题：用真实分布下的合理阈值应当被拒答。"""
    # hash 兜底后端的分数普遍偏低，这里用 0.5 模拟真实模型上的"安全阈值"
    result = rag_engine.answer("今天北京的天气怎么样", threshold=0.5)
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.refuse_reason == "below_threshold"


def test_threshold_zero_admits_retrieval(rag_engine, seeded_store) -> None:
    """阈值设为 0 时不做短路（用于对比测试，验证阈值确实起作用）。"""
    result = rag_engine.answer("年假有几天", threshold=0.0)
    assert result.refuse_reason != "below_threshold"


# ---------------------------------------------------------------------------
# L3-a：模型自述资料不足
# ---------------------------------------------------------------------------
def _patch_llm(engine, payload: Optional[Dict[str, Any]], ok: bool = True, error_type: Optional[str] = None):
    """替换引擎的 LLM 调用，构造确定的模型输出。

    做法：把 ``chat_json`` 换成固定返回，并让 ``available`` 属性恒为 True
    （否则引擎会因为"没配 Key"直接走离线分支，测不到 LLM 分支）。
    """
    from unittest.mock import PropertyMock, patch as mock_patch

    from core.llm import LLMResult

    def fake_chat_json(system_prompt: str, user_prompt: str, temperature=None):
        result = LLMResult(
            ok=ok,
            text="" if payload is None else str(payload),
            error=None if ok else "模拟失败",
            error_type=error_type,
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
            cost_est=0.0002,
            latency_ms=42,
        )
        return (payload if ok else None), result

    # 让 available 属性恒为 True（PropertyMock 可以直接覆盖 property）
    patcher = mock_patch.object(
        type(engine.llm), "available", new_callable=PropertyMock, return_value=True
    )
    patcher.start()
    engine._llm_patcher = patcher  # 由 fixture 收尾时 stop
    engine.llm.chat_json = fake_chat_json  # type: ignore[method-assign]


def test_refuse_when_llm_uncertain(rag_engine, seeded_store) -> None:
    """模型明确表示资料不足（confident=false）→ 拒答。"""
    _patch_llm(rag_engine, {"answer": "资料里没有提到。", "confident": False})

    result = rag_engine.answer("年假有几天", threshold=0.0)
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.refuse_reason == "llm_uncertain"
    assert result.total_tokens == 160          # 这次确实调用了模型


def test_refuse_when_llm_returns_refusal_text(rag_engine, seeded_store) -> None:
    """模型直接输出标准话术 → 判定为拒答（而不是当成正常答案）。"""
    _patch_llm(rag_engine, {"answer": REFUSAL_MESSAGE, "confident": True})

    result = rag_engine.answer("年假有几天", threshold=0.0)
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.refuse_reason == "llm_uncertain"


def test_refuse_when_llm_returns_empty_answer(rag_engine, seeded_store) -> None:
    """模型返回空答案 → 拒答（不能把空字符串抛给用户）。"""
    _patch_llm(rag_engine, {"answer": "   ", "confident": True})

    result = rag_engine.answer("年假有几天", threshold=0.0)
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE


# ---------------------------------------------------------------------------
# L3-b：引用校验
# ---------------------------------------------------------------------------
def test_refuse_on_invalid_citation(rag_engine, seeded_store) -> None:
    """回答引用了不存在的编号 → 拒答（这是防幻觉的关键一环）。"""
    _patch_llm(rag_engine, {"answer": "年假为五天【9】。", "confident": True})

    result = rag_engine.answer("年假有几天", top_k=3, threshold=0.0)
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.refuse_reason == "citation_invalid"
    assert result.meta["invalid_citations"] == [9]


def test_accept_valid_citation(rag_engine, seeded_store) -> None:
    """合法引用（编号在片段范围内）→ 正常返回答案。"""
    _patch_llm(rag_engine, {"answer": "年假为五天，满十年为十天【1】。", "confident": True, "citations": [1]})

    result = rag_engine.answer("年假有几天", top_k=3, threshold=0.0)
    assert result.refused is False
    assert "五天" in result.answer
    assert result.meta["cited"] == [1]


def test_accept_answer_without_citation(rag_engine, seeded_store) -> None:
    """模型没写引用**不算错误**（只有"引用不存在"才拒答）。"""
    _patch_llm(rag_engine, {"answer": "年假为五天，满十年为十天。", "confident": True})

    result = rag_engine.answer("年假有几天", top_k=3, threshold=0.0)
    assert result.refused is False
    assert result.meta["had_citation"] is False


def test_citation_check_can_be_disabled(rag_engine, seeded_store) -> None:
    """ENABLE_CITATION_CHECK=false 时不做引用校验（用于对比实验）。"""
    _patch_llm(rag_engine, {"answer": "年假为五天【9】。", "confident": True})
    object.__setattr__(rag_engine.settings, "enable_citation_check", False)
    try:
        result = rag_engine.answer("年假有几天", top_k=3, threshold=0.0)
    finally:
        object.__setattr__(rag_engine.settings, "enable_citation_check", True)
    assert result.refused is False


def test_llm_failure_degrades_to_excerpt(rag_engine, seeded_store) -> None:
    """大模型调用失败 → 降级为片段摘要（有依据），而不是直接报错或编造。"""
    _patch_llm(rag_engine, None, ok=False, error_type="llm_timeout")

    result = rag_engine.answer("年假有几天", threshold=0.0)
    assert result.refused is False
    assert result.llm_degraded is True
    assert result.meta["mode"] == "fallback_after_llm_error"
    assert result.meta["error_type"] == "llm_timeout"
    assert result.sources, "降级回答仍然必须带来源"


# ---------------------------------------------------------------------------
# 离线模式
# ---------------------------------------------------------------------------
def test_offline_mode_returns_excerpt_with_source(rag_engine, seeded_store) -> None:
    """离线模式：答案由片段摘取生成，必须带引用编号与来源。"""
    result = rag_engine.answer("年假有几天", threshold=0.0)
    assert result.refused is False
    assert result.llm_degraded is True
    assert result.meta["mode"] == "offline"
    assert result.sources
    assert "【" in result.answer           # 句末带引用编号


def test_offline_mode_never_fabricates(rag_engine, seeded_store) -> None:
    """离线模式下一个关键词都没命中时，必须如实拒答而不是拼凑内容。"""
    result = rag_engine.answer("量子纠缠退相干时间是多少", threshold=0.0)
    assert result.answer == REFUSAL_MESSAGE
    assert result.meta["cited"] == []


# ---------------------------------------------------------------------------
# 引用编号解析（纯函数）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("答案【1】", [1]),
        ("答案[1]", [1]),
        ("答案【1,2】", [1, 2]),
        ("答案【1-3】", [1, 2, 3]),
        ("答案【1】和【2】", [1, 2]),
        ("答案【1】【1】", [1]),          # 去重
        ("没有引用的答案", []),
        ("", []),
    ],
)
def test_extract_citation_indices(text: str, expected: List[int]) -> None:
    """引用编号解析：支持中英文括号、逗号分隔、区间与去重。"""
    from rag.answer import extract_citation_indices

    assert extract_citation_indices(text) == expected


@pytest.mark.parametrize(
    "text,hit_count,ok",
    [
        ("答案【1】", 3, True),
        ("答案【3】", 3, True),
        ("答案【4】", 3, False),
        ("答案【0】", 3, False),
        ("答案【1-5】", 3, False),        # 区间越界不能被静默忽略
        ("没有引用", 3, True),            # 没有引用不算错误
    ],
)
def test_validate_citations(text: str, hit_count: int, ok: bool) -> None:
    """引用校验：越界或 0 编号必须判为不合法。"""
    from rag.answer import validate_citations

    assert validate_citations(text, hit_count)["ok"] is ok


# ---------------------------------------------------------------------------
# 拒答话术常量
# ---------------------------------------------------------------------------
def test_refusal_message_is_exact() -> None:
    """拒答话术必须与验收标准逐字一致（改动会直接导致验收失败）。"""
    assert REFUSAL_MESSAGE == "根据现有资料，我无法回答这个问题"


def test_all_refusal_paths_use_same_message(rag_engine, seeded_store) -> None:
    """四种拒答原因返回的话术必须完全相同（前端才能用等值判断展示）。"""
    messages = set()

    messages.add(rag_engine.answer("年假", threshold=1.0).answer)          # below_threshold
    _patch_llm(rag_engine, {"answer": "x", "confident": False})
    messages.add(rag_engine.answer("年假", threshold=0.0).answer)          # llm_uncertain
    _patch_llm(rag_engine, {"answer": "年假五天【9】", "confident": True})
    messages.add(rag_engine.answer("年假", threshold=0.0).answer)          # citation_invalid

    assert messages == {REFUSAL_MESSAGE}
