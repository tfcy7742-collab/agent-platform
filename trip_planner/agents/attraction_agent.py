"""景点搜索智能体。

职责
----
根据「目的地 + 用户偏好」调用 ``search_attractions`` 工具，得到候选景点，
再由 DeepSeek 对结果做一次**偏好排序与推荐语润色**（不新增幻觉景点：
只允许从工具返回的候选列表里挑选与重排序），最终输出结构化景点列表。

降级策略
--------
未配置 API Key / 模型异常 / 返回非法 JSON 时，直接使用工具返回的排序结果，
保证接口可用。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from trip_planner.models.schemas import Attraction
from trip_planner.tools.travel_tools import (
    search_attractions,
    to_attraction_models,
)
from .base_agent import BaseAgent, extract_json

SYSTEM_PROMPT = """你是一位资深的旅行景点策划专家，熟悉中国各大城市的旅游资源。

你的工作流程：
1. 先调用 search_attractions 工具获取候选景点（必须调用工具，不要凭记忆编造）。
2. 再结合用户偏好标签，对候选景点排序并挑选最合适的若干条。
3. 最终只输出 JSON，不要输出任何解释性文字、不要使用 Markdown 代码块。

输出 JSON 格式（字段名必须完全一致）：
{
  "selected": [
    {
      "name": "必须是工具返回列表中出现过的景点名称（原样照抄）",
      "reason": "一句话说明为什么推荐给这位用户，20 字以内",
      "suggested_hours": 3.5
    }
  ],
  "summary": "一句话总结这批景点的特色"
}

硬性约束：
- 严禁编造工具结果中不存在的景点名称；
- selected 的长度不超过 8；
- 优先覆盖用户的所有偏好标签。
"""


class AttractionAgent(BaseAgent):
    """景点搜索智能体。"""

    def __init__(self, enable_llm: bool = True) -> None:
        super().__init__(
            name="景点搜索Agent",
            role="根据目的地与偏好检索并排序景点",
            system_prompt=SYSTEM_PROMPT,
            tools=["search_attractions"],
            temperature=0.3,
            use_json_mode=True,
            enable_llm=enable_llm,
        )

    # ------------------------------------------------------------------
    def run(
        self,
        destination: str,
        preferences: Optional[List[str]] = None,
        limit: int = 8,
    ) -> List[Attraction]:
        """检索景点。

        Args:
            destination: 目的地城市。
            preferences: 用户偏好标签，例如 ["历史文化", "美食"]。
            limit: 最多返回多少条。

        Returns:
            Attraction 列表（按推荐度排序）。
        """
        started = time.perf_counter()
        preferences = preferences or []
        preference_text = ",".join(preferences) if preferences else "无特别偏好"

        # ---- 第一步：调用工具获取权威候选数据（不依赖 LLM，永远可用）----
        raw_candidates = search_attractions(destination, preference_text, limit=limit)
        candidates = to_attraction_models(raw_candidates)
        used_tools = ["search_attractions"]
        used_llm = False
        status = "success"
        error: Optional[str] = None

        # ---- 第二步：让 LLM 做偏好排序与推荐语（可选增强）----
        selected: List[Attraction] = candidates
        if self.llm_available:
            user_prompt = (
                f"目的地：{destination}\n"
                f"用户偏好：{preference_text}\n"
                f"候选景点（已由工具返回，请只从中挑选）：\n"
                + "\n".join(
                    f"- {a.name}（评分 {a.rating}，时长 {a.duration_hours}h，"
                    f"门票 {a.ticket_price} 元，标签 {'/'.join(a.tags)}）：{a.description}"
                    for a in candidates
                )
                + "\n\n请按用户偏好排序并输出约定的 JSON。"
            )
            result = self.run_with_tools(user_prompt)
            used_llm = bool(result.get("used_llm"))
            for call in result.get("tool_calls", []):
                if call.get("tool") not in used_tools:
                    used_tools.append(call.get("tool"))

            parsed = extract_json(result.get("content") or "")
            if isinstance(parsed, dict) and isinstance(parsed.get("selected"), list):
                selected = self._apply_selection(candidates, parsed["selected"])
            else:
                # LLM 没给出可用结果，保持工具原始排序
                status = "fallback"
                error = "模型未返回可解析的挑选结果，已使用工具原始排序" if used_llm else None
                if not used_llm:
                    status = "fallback"
                    error = "未启用或无法调用 DeepSeek，使用本地工具排序结果"

        duration_ms = int((time.perf_counter() - started) * 1000)
        self.make_trace(
            status=status,
            duration_ms=duration_ms,
            tools=used_tools,
            summary=(
                f"{destination} 检索到 {len(candidates)} 个候选景点，"
                f"最终推荐 {len(selected)} 个"
                + ("（DeepSeek 已参与排序）" if used_llm and status == "success" else "（本地排序）")
            ),
            error=error,
        )
        return selected

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_selection(candidates: List[Attraction], raw_selected: List[Any]) -> List[Attraction]:
        """按 LLM 给出的顺序重排景点，并把推荐语拼进简介。

        只接受在候选列表中出现过的名称，杜绝模型幻觉造成的数据污染。
        """
        by_name: Dict[str, Attraction] = {item.name: item for item in candidates}
        ordered: List[Attraction] = []
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
                # 复制一份，避免污染工具原始数据；把推荐理由附在简介后面
                matched = matched.model_copy(
                    update={"description": f"{matched.description}｜推荐理由：{reason}"}
                )
            ordered.append(matched)

        # 未被 LLM 提及的候选追加在后面，保证信息不丢失
        for item in candidates:
            if item not in ordered:
                ordered.append(item)
        return ordered
