"""行程规划智能体。

职责
----
这是多智能体协作中的「最后一棒」：接收
    景点搜索 Agent + 天气查询 Agent + 酒店推荐 Agent
的输出，整合成一份结构化的最终行程计划（TripPlan 的行程部分 + 预算明细 + 注意事项）。

工作方式
--------
* **主路径**：把三个 Agent 的结构化结果整理成提示词交给 DeepSeek，
  要求其输出严格的 JSON（逐日行程 + 预算明细 + 注意事项 + 概述），
  再用 Pydantic 校验并转换为模型对象；
* **降级路径**：DeepSeek 不可用或输出非法时，调用 ``fallback_planner``
  中的本地规则生成行程，保证结果始终可用、字段始终完整；
* **预算二次校准**：无论哪条路径，最终都会用本地公式重新核算预算明细，
  保证「预算明细合计 = 预估总花费」这一硬约束不被模型破坏。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from trip_planner.models.schemas import (
    Attraction,
    BudgetBreakdown,
    DailyPlan,
    Hotel,
    TripPlan,
    TripRequest,
    WeatherInfo,
)
from trip_planner.tools.travel_tools import weather_text
from .base_agent import BaseAgent, extract_json
from .fallback_planner import (
    BudgetFit,
    budget_floor,
    budget_limit,
    budget_status_text,
    build_daily_plans,
    build_summary,
    build_tips,
    estimate_budget,
    fit_budget,
)

SYSTEM_PROMPT = """你是一位顶级的私人旅行定制规划师，擅长把景点、天气、酒店信息整合成可执行的逐日行程。

你会收到三个专业智能体的工作成果：
1. 景点搜索 Agent 给出的候选景点（含建议时长与门票价格）；
2. 天气查询 Agent 给出的逐日天气（含出行建议）；
3. 酒店推荐 Agent 给出的候选酒店（含每晚价格与位置）。

你的任务：输出一份逐日行程规划。只输出 JSON，不要输出任何解释性文字，不要使用 Markdown 代码块。

输出 JSON 格式（字段名必须完全一致）：
{
  "summary": "整体行程概述，120 字以内",
  "daily_plans": [
    {
      "day": 1,
      "date": "YYYY-MM-DD（必须来自给定的日期列表）",
      "theme": "当日主题，12 字以内",
      "morning": "上午安排，60-120 字，必须写出景点名称与具体动作",
      "afternoon": "下午安排，60-120 字，必须写出景点名称与具体动作",
      "evening": "晚间安排，40-100 字，含晚餐或夜间活动推荐",
      "accommodation": "当晚住宿建议（写酒店名称与选择理由）",
      "meals": ["早餐推荐", "午餐推荐", "晚餐推荐"],
      "transportation": "当日交通方式与建议",
      "estimated_cost": 480,
      "weather_note": "当日天气与应对建议",
      "tips": "当日小贴士"
    }
  ],
  "tips": ["注意事项 1", "注意事项 2"]
}

硬性约束（非常重要）：
- daily_plans 的长度必须等于给定天数，day 从 1 连续递增，date 必须与给定日期严格对应；
- 景点名称只允许使用「候选景点」列表中出现过的名称；
- 雨天（小雨/阵雨/雷阵雨）必须把户外景点换成室内展馆或街区；
- 每天上午、下午、晚间三段都不能为空，且同一天不要重复安排同一个景点；
- 行程要顺路，同一天的景点尽量在同一区域，避免全城折返；
- estimated_cost 为当日人均花费（含餐饮、交通、门票，不含往返大交通）；
- 全部使用简体中文。
"""


class PlannerAgent(BaseAgent):
    """行程规划智能体：整合前三个 Agent 的输出，生成最终行程。"""

    def __init__(self, enable_llm: bool = True) -> None:
        super().__init__(
            name="行程规划Agent",
            role="整合景点、天气、酒店信息生成最终行程",
            system_prompt=SYSTEM_PROMPT,
            tools=[],  # 该智能体不直接调用工具，而是消费其它 Agent 的产物
            temperature=0.5,
            use_json_mode=True,
            enable_llm=enable_llm,
        )

    # ------------------------------------------------------------------
    def run(
        self,
        request: TripRequest,
        attractions: List[Attraction],
        weather: List[WeatherInfo],
        hotels: List[Hotel],
    ) -> TripPlan:
        """生成最终旅行计划。

        Args:
            request: 用户请求。
            attractions: 景点搜索 Agent 的输出。
            weather: 天气查询 Agent 的输出。
            hotels: 酒店推荐 Agent 的输出。

        Returns:
            完整的 TripPlan。
        """
        started = time.perf_counter()
        status = "success"
        error: Optional[str] = None

        hotel_name = hotels[0].name if hotels else f"{request.destination}{request.budget_level}档酒店"
        nightly_price = hotels[0].price_per_night if hotels else 0
        weather_map: Dict[str, str] = {w.date: weather_text(w) for w in weather}
        weather_dict: Dict[str, str] = {w.date: w.suggestion for w in weather}

        # ---- 主路径：调用 DeepSeek 生成行程 ----
        daily_plans: List[DailyPlan] = []
        summary = ""
        tips: List[str] = []
        generated_by = "本地规则引擎（DeepSeek 未参与）"

        if self.llm_available:
            user_prompt = self._build_prompt(request, attractions, weather, hotels)
            raw_text = self.chat(user_prompt, json_mode=True)
            parsed = extract_json(raw_text or "")
            if isinstance(parsed, dict) and parsed.get("daily_plans"):
                daily_plans, summary, tips = self._parse_llm_output(
                    parsed, request, hotel_name, weather_dict
                )
                generated_by = f"DeepSeek {self.settings.model} + 多智能体协作"
                status = "success"
            else:
                status = "fallback"
                error = "模型未返回可解析的行程 JSON，已切换到本地规则引擎"
                generated_by = "本地规则引擎（DeepSeek 输出不可用）"

        # ---- 降级路径：本地规则生成 ----
        if not daily_plans:
            daily_plans = build_daily_plans(
                request,
                attractions,
                weather_texts=weather_dict,
                hotel_name=hotel_name,
                nightly_price=nightly_price,
            )
            summary = build_summary(request, daily_plans)
            tips = build_tips(request, attractions, list(weather_dict.values()))
            if self.llm_available and status == "success":
                status = "fallback"
                error = "DeepSeek 未启用，使用本地规则引擎"

        # ---- 预算：始终由本地公式校准，保证明细与合计自洽 ----
        attraction_total = sum(a.ticket_price for a in attractions[: request.days * 2])
        breakdown = estimate_budget(request, nightly_price=nightly_price, attraction_total=attraction_total)
        estimated_total = breakdown.total
        status_text = budget_status_text(estimated_total, request.budget)

        # ---- 预算硬约束：把总花费收敛到「预算 × 1.1」以内 ----
        fit = fit_budget(
            request,
            attractions,
            nightly_price=nightly_price,
            hotel_name=hotel_name,
            candidates=hotels,
        )
        dropped_names: List[str] = []
        extent_notes = ""
        if fit.adjusted:
            effective = fit.effective_request

            # 行程文案里出现了、但已被预算挤出去的付费景点 → 换成本地行程保证前后一致
            mentioned = {
                a.name
                for a in attractions
                if a.ticket_price > 0
                and any(a.name in (day.morning + day.afternoon + day.evening) for day in daily_plans)
            }
            priced_names = {a.name for a in attractions[: fit.priced_attraction_count]}
            dropped_names = sorted(mentioned - priced_names)

            if fit.regenerate_needed:
                daily_plans = build_daily_plans(
                    effective,
                    attractions[: fit.priced_attraction_count],
                    weather_texts=weather_dict,
                    hotel_name=fit.hotel_name,
                    nightly_price=fit.nightly_price,
                )

            breakdown = fit.breakdown
            estimated_total = breakdown.total
            request = effective
            hotel_name = fit.hotel_name or hotel_name
            nightly_price = fit.nightly_price
            extent_notes = self._budget_fit_notes(fit, dropped_names)
            status_text = budget_status_text(estimated_total, fit.request.budget) + extent_notes

        if not tips:
            tips = build_tips(request, attractions, list(weather_dict.values()))
        # 把预算结论作为第一条注意事项，信息更突出
        tips = [status_text, *tips]

        plan = TripPlan(
            destination=request.destination,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            days=request.days,
            travelers=request.travelers,
            budget=request.budget,
            budget_level=request.budget_level,
            preferences=request.preferences,
            summary=summary or build_summary(request, daily_plans),
            daily_plans=daily_plans,
            attractions=attractions,
            hotels=hotels,
            weather=weather,
            budget_breakdown=breakdown,
            estimated_total=estimated_total,
            budget_status=status_text,
            budget_limit=budget_limit(request.budget),
            budget_floor=budget_floor(request.budget),
            within_budget=fit.within_budget,
            within_floor=fit.within_floor,
            budget_fit=self._budget_fit_payload(fit, extent_notes),
            tips=tips,
            generated_by=generated_by,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        duration_ms = int((time.perf_counter() - started) * 1000)
        self.make_trace(
            status=status,
            duration_ms=duration_ms,
            tools=[],
            summary=(
                f"生成 {len(daily_plans)} 天行程，预算合计 {estimated_total} 元，"
                f"生成方式：{generated_by}"
            ),
            error=error,
        )
        return plan

    # ------------------------------------------------------------------
    # 预算收敛结果整理
    # ------------------------------------------------------------------
    @staticmethod
    def _budget_fit_notes(fit: BudgetFit, dropped_names: List[str]) -> str:
        """把收敛结论补全为面向用户的说明文本。"""
        notes = fit.notes
        if dropped_names:
            notes += f"（已从行程中移出：{'、'.join(dropped_names)}）"
        return notes

    @staticmethod
    def _budget_fit_payload(fit: BudgetFit, notes: str) -> Dict[str, Any]:
        """输出预算收敛的结构化明细，便于前端展示与调试。"""
        return {
            "applied": fit.adjusted,
            "within_budget": fit.within_budget,
            "within_floor": fit.within_floor,
            "budget_limit": budget_limit(fit.request.budget),
            "budget_floor": budget_floor(fit.request.budget),
            "max_spendable": fit.max_spendable,
            "period_ratio": round(fit.period_ratio, 4),
            "level_before": fit.request.budget_level,
            "level_after": fit.effective_request.budget_level,
            "level_downgraded": fit.level_downgraded,
            "hotel_name": fit.hotel_name,
            "nightly_price": fit.nightly_price,
            "priced_attraction_count": fit.priced_attraction_count,
            "dropped_attractions": fit.dropped_attractions,
            "min_feasible_total": fit.min_feasible_total,
            "notes": notes,
        }

    # ------------------------------------------------------------------
    # 提示词构造
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(
        request: TripRequest,
        attractions: List[Attraction],
        weather: List[WeatherInfo],
        hotels: List[Hotel],
    ) -> str:
        """把三个 Agent 的输出拼装成规划提示词。"""
        date_lines = [
            f"第 {i + 1} 天：{d}" for i, d in enumerate(request.date_strings())
        ]
        attraction_lines = [
            f"- {a.name}｜{a.location or request.destination}｜建议 {a.duration_hours:g}h｜"
            f"门票 {a.ticket_price} 元｜评分 {a.rating}｜标签 {'/'.join(a.tags)}｜{a.description}"
            for a in attractions
        ]
        weather_lines = [
            f"- {w.date}（{w.weekday}）{w.condition} {w.temp_range} {w.wind}｜建议：{w.suggestion}"
            for w in weather
        ]
        hotel_lines = [
            f"- {h.name}｜{h.price_per_night} 元/晚｜评分 {h.rating}｜{h.location}｜"
            f"{h.distance_to_center}｜标签 {'/'.join(h.tags)}"
            for h in hotels
        ]

        return (
            "【用户需求】\n"
            f"目的地：{request.destination}\n"
            f"出发日期：{request.start_date.isoformat()}，共 {request.days} 天"
            f"（{request.start_date.isoformat()} 至 {request.end_date.isoformat()}）\n"
            f"出行人数：{request.travelers} 人\n"
            f"总预算：{request.budget} 元（档位：{request.budget_level}）\n"
            f"偏好标签：{'、'.join(request.preferences) if request.preferences else '无特别偏好'}\n"
            f"补充说明：{request.notes or '无'}\n\n"
            "【日期列表（date 字段必须从这里取）】\n"
            + "\n".join(date_lines)
            + "\n\n【景点搜索 Agent 的候选景点】\n"
            + ("\n".join(attraction_lines) or "- （无候选景点，请按常识安排城市地标与街区）")
            + "\n\n【天气查询 Agent 的逐日天气】\n"
            + ("\n".join(weather_lines) or "- （无天气数据）")
            + "\n\n【酒店推荐 Agent 的候选酒店】\n"
            + ("\n".join(hotel_lines) or "- （无酒店数据）")
            + f"\n\n请按约定 JSON 格式输出 {request.days} 天的完整行程。"
        )

    # ------------------------------------------------------------------
    # LLM 输出解析
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_llm_output(
        parsed: Dict[str, Any],
        request: TripRequest,
        hotel_name: str,
        weather_dict: Dict[str, str],
    ) -> tuple:
        """把 LLM 的 JSON 输出转换为 DailyPlan 列表。

        做三件事保证健壮性：
        1. 天数不足时用本地规则补齐，天数超出时截断；
        2. 缺失字段（日期、住宿、天气提示）自动回填；
        3. 上午/下午/晚间为空时给出兜底文案。

        Returns:
            ``(daily_plans, summary, tips)``
        """
        raw_days = parsed.get("daily_plans") or []
        date_strings = request.date_strings()
        daily_plans: List[DailyPlan] = []

        for index, raw in enumerate(raw_days[: request.days]):
            if not isinstance(raw, dict):
                continue
            date_str = str(raw.get("date") or date_strings[index])
            if date_str not in date_strings:
                date_str = date_strings[index]

            meals = raw.get("meals")
            if isinstance(meals, str):
                meals = [meals]
            elif not isinstance(meals, list):
                meals = []

            try:
                cost = int(raw.get("estimated_cost") or 0)
            except (TypeError, ValueError):
                cost = 0

            daily_plans.append(
                DailyPlan(
                    day=int(raw.get("day") or index + 1),
                    date=date_str,
                    theme=str(raw.get("theme") or f"第 {index + 1} 天行程"),
                    morning=str(raw.get("morning") or "上午自由活动，按体力灵活安排。"),
                    afternoon=str(raw.get("afternoon") or "下午自由活动，可就近游览街区。"),
                    evening=str(raw.get("evening") or "品尝当地特色晚餐，早些休息。"),
                    accommodation=str(raw.get("accommodation") or hotel_name),
                    meals=[str(m) for m in meals],
                    transportation=str(raw.get("transportation") or "地铁 + 步行为主，必要时打车。"),
                    estimated_cost=max(cost, 0),
                    weather_note=str(raw.get("weather_note") or weather_dict.get(date_str, "")),
                    tips=str(raw.get("tips") or "行程节奏适中，可按当天状态灵活调整。"),
                )
            )

        # 天数不足：用本地规则补齐剩余天数
        if len(daily_plans) < request.days:
            existing_dates = {p.date for p in daily_plans}
            supplement = build_daily_plans(
                request,
                [],  # 补齐时不再依赖景点列表
                weather_texts=weather_dict,
                hotel_name=hotel_name,
            )
            for plan in supplement:
                if plan.date not in existing_dates:
                    plan = plan.model_copy(update={"day": len(daily_plans) + 1})
                    daily_plans.append(plan)
                if len(daily_plans) >= request.days:
                    break

        # 重新编号，保证 day 连续
        daily_plans = [
            plan.model_copy(update={"day": i + 1}) for i, plan in enumerate(daily_plans)
        ]

        summary = str(parsed.get("summary") or "")
        raw_tips = parsed.get("tips")
        if isinstance(raw_tips, str):
            tips = [raw_tips]
        elif isinstance(raw_tips, list):
            tips = [str(t) for t in raw_tips if str(t).strip()]
        else:
            tips = []

        return daily_plans, summary, tips
