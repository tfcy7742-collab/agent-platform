"""酒店推荐智能体。

职责
----
根据「目的地 + 预算档位 + 单人每晚可支配住宿预算」调用 ``search_hotels`` 工具，
得到候选酒店，再由 DeepSeek 结合总预算与偏好挑选并给出推荐理由，
输出结构化酒店列表。

降级策略
--------
未配置 API Key / 模型异常 / 返回非法 JSON 时，直接使用工具的价格升序结果。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from trip_planner.models.schemas import Hotel
from trip_planner.tools.travel_tools import search_hotels, to_hotel_models
from .base_agent import BaseAgent, extract_json

SYSTEM_PROMPT = """你是一位酒店选址与住宿体验顾问，熟悉中国主要城市的酒店分布。

你的工作流程：
1. 先调用 search_hotels 工具获取候选酒店（必须调用工具，不要凭记忆编造）。
2. 再结合用户预算档位、每晚住宿预算上限与偏好，挑选最合适的 2-3 家并说明理由。
3. 最终只输出 JSON，不要输出任何解释性文字、不要使用 Markdown 代码块。

输出 JSON 格式（字段名必须完全一致）：
{
  "selected": [
    {
      "name": "必须是工具返回列表中出现过的酒店名称（原样照抄）",
      "reason": "一句话推荐理由，20 字以内"
    }
  ],
  "summary": "一句话说明住宿区域选择建议"
}

硬性约束：
- 严禁编造工具结果中不存在的酒店；
- 优先考虑「位置便于出行」与「价格不超过用户住宿预算」的酒店；
- selected 长度 1-3。
"""


class HotelAgent(BaseAgent):
    """酒店推荐智能体。"""

    def __init__(self, enable_llm: bool = True) -> None:
        super().__init__(
            name="酒店推荐Agent",
            role="按预算档位检索并筛选酒店",
            system_prompt=SYSTEM_PROMPT,
            tools=["search_hotels"],
            temperature=0.3,
            use_json_mode=True,
            enable_llm=enable_llm,
        )

    # ------------------------------------------------------------------
    def run(
        self,
        destination: str,
        budget_level: str = "中等",
        nightly_budget: Optional[int] = None,
        limit: int = 3,
    ) -> List[Hotel]:
        """检索酒店。

        Args:
            destination: 目的地城市。
            budget_level: 预算档位：经济 / 中等 / 豪华。
            nightly_budget: 每晚住宿预算上限（元），None 表示不限制。
            limit: 最多返回条数。

        Returns:
            Hotel 列表。
        """
        started = time.perf_counter()
        used_tools = ["search_hotels"]
        status = "success"
        error: Optional[str] = None

        # ---- 第一步：调用工具获取候选酒店 ----
        raw_candidates = search_hotels(destination, budget_level, limit=max(limit, 3))
        candidates = to_hotel_models(raw_candidates)
        selected: List[Hotel] = candidates
        used_llm = False

        budget_text = f"{nightly_budget} 元" if nightly_budget else "未限制"

        # ---- 第二步：让 LLM 结合预算挑选（可选增强）----
        if self.llm_available and candidates:
            user_prompt = (
                f"目的地：{destination}\n"
                f"预算档位：{budget_level}\n"
                f"每晚住宿预算上限：{budget_text}\n"
                f"候选酒店（请只从中挑选）：\n"
                + "\n".join(
                    f"- {h.name}：{h.price_per_night} 元/晚，评分 {h.rating}，"
                    f"{h.location}，{h.distance_to_center}，标签 {'/'.join(h.tags)}"
                    for h in candidates
                )
                + "\n\n请按用户预算挑选并输出约定的 JSON。"
            )
            result = self.run_with_tools(user_prompt)
            used_llm = bool(result.get("used_llm"))
            for call in result.get("tool_calls", []):
                if call.get("tool") not in used_tools:
                    used_tools.append(call.get("tool"))

            parsed = extract_json(result.get("content") or "")
            if isinstance(parsed, dict) and isinstance(parsed.get("selected"), list):
                selected = self._apply_selection(candidates, parsed["selected"])[:limit]
            else:
                status = "fallback"
                error = (
                    "模型未返回可解析的挑选结果，使用工具价格排序"
                    if used_llm
                    else "未启用或无法调用 DeepSeek，使用本地工具结果"
                )

        # 无论 LLM 是否参与，都保证不超过每晚预算上限（若用户给了上限）
        within: List[Hotel] = []
        if nightly_budget:
            within = [h for h in selected if h.price_per_night <= nightly_budget]
            if within:
                selected = within
            else:
                # 当前档位的候选全部超上限（例如"中等"档最低价 698 而限额 500）：
                # 不能就这样把超限酒店交出去，否则每晚限额形同不存在。
                # 换最低档再检索一次，优先给出真正住得起的酒店。
                cheaper = to_hotel_models(
                    search_hotels(destination, "经济", limit=max(limit, 3))
                )
                within = [h for h in cheaper if h.price_per_night <= nightly_budget]
                if within:
                    selected = within
                    error = (
                        f"{nightly_budget} 元/晚的限额内没有 {budget_level} 档酒店，"
                        f"已改推经济档候选"
                    )

        duration_ms = int((time.perf_counter() - started) * 1000)
        self.make_trace(
            status=status,
            duration_ms=duration_ms,
            tools=used_tools,
            summary=(
                f"{destination} / {budget_level}档 检索到 {len(candidates)} 家候选酒店，"
                f"最终推荐 {len(selected)} 家"
                + ("（DeepSeek 已参与筛选）" if used_llm and status == "success" else "（本地筛选）")
            ),
            error=error,
        )
        return selected

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_selection(candidates: List[Hotel], raw_selected: List[Any]) -> List[Hotel]:
        """按 LLM 给出的顺序重排酒店，并把推荐理由写进标签。

        只接受候选列表中出现过的酒店名称。
        """
        by_name: Dict[str, Hotel] = {item.name: item for item in candidates}
        ordered: List[Hotel] = []
        for entry in raw_selected:
            if isinstance(entry, str):
                name, reason = entry, ""
            elif isinstance(entry, dict):
                name = str(entry.get("name", "")).strip()
                reason = str(entry.get("reason", "")).strip()
            else:
                continue

            matched = by_name.get(name)
            if matched is None or matched in ordered:
                continue
            if reason:
                matched = matched.model_copy(
                    update={"tags": [*matched.tags, f"推荐理由：{reason}"]}
                )
            ordered.append(matched)

        for item in candidates:
            if item not in ordered:
                ordered.append(item)
        return ordered
