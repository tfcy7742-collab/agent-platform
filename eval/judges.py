"""LLM 判分（judge）实现。

为什么单独一个模块
------------------
判分提示词与调用逻辑会被多处复用（评测 CLI、回归测试、将来的 CI），
集中放这里便于迭代与 A/B；同时把"判分失败"与"判为不正确"区分开——
判分失败时返回 ``None``（不计入准确率），避免因为接口抖动把指标拉低。

判分原则
--------
1. **关键事实正确即可**，不要求措辞与参考答案一致（模型输出天然多样）；
2. **该拒答的必须拒答**：参考答案是标准拒答话术时，模型也必须拒答才算对；
3. **出现编造或与参考答案矛盾的关键事实**，一律判为不正确。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from core.llm import LLMClient
from core.prompts import REFUSAL_MESSAGE

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """你是一个严格的答案评审员。请判断【模型回答】是否**正确回答了**【问题】。

判定标准：
1. 与【参考答案】相比，关键事实（数字、条件、结论）正确即可，不要求措辞一致；
2. 如果【参考答案】是「{refusal}」，说明资料中没有答案，
   此时模型回答必须也是拒绝回答（可以是这句话，也可以是明确表示资料中没有相关内容），
   才算正确；
3. 只要出现编造、或与参考答案矛盾的关键事实，一律判为不正确；
4. 回答里引用编号是否准确不在本次判分范围内（另有引用校验）。

只输出 JSON：{{"correct": true, "reason": "10 字以内的理由"}}
"""

JUDGE_USER_TEMPLATE = """【问题】
{question}

【参考答案】
{expected}

【模型回答】
{answer}
"""


def judge_answer(
    llm: LLMClient,
    question: str,
    expected: str,
    answer: str,
    answer_limit: int = 800,
) -> Tuple[Optional[bool], str]:
    """用大模型判断答案是否正确。

    Args:
        llm: LLM 客户端。
        question: 用户问题。
        expected: 参考答案（可能是标准拒答话术）。
        answer: 模型实际回答。
        answer_limit: 截断长度（避免超长回答把判分提示词撑爆）。

    Returns:
        ``(是否正确, 理由)``；判分失败时第一项为 ``None``。
    """
    if not llm.available:
        return None, "未配置大模型，跳过 LLM 判分"

    parsed, result = llm.chat_json(
        JUDGE_SYSTEM_PROMPT.format(refusal=REFUSAL_MESSAGE),
        JUDGE_USER_TEMPLATE.format(
            question=question,
            expected=expected or f"（资料中无答案，正确做法是回答「{REFUSAL_MESSAGE}」）",
            answer=(answer or "")[:answer_limit],
        ),
    )
    if not result.ok or not isinstance(parsed, dict):
        logger.info("判分失败（%s），该条不计入 LLM 判分准确率", result.error_type)
        return None, f"判分失败：{result.error_type or 'unknown'}"

    return bool(parsed.get("correct")), str(parsed.get("reason") or "")


def judge_summary(results: list) -> dict:
    """汇总判分结果（供报告展示）。"""
    judged = [item for item in results if getattr(item, "llm_judge", None) is not None]
    if not judged:
        return {"judged": 0, "correct": 0, "accuracy": None}
    correct = sum(1 for item in judged if item.llm_judge)
    return {
        "judged": len(judged),
        "correct": correct,
        "accuracy": round(correct / len(judged), 4),
    }


__all__ = ["JUDGE_SYSTEM_PROMPT", "JUDGE_USER_TEMPLATE", "judge_answer", "judge_summary"]
