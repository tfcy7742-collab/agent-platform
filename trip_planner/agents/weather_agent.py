"""天气查询智能体。

职责
----
根据「目的地 + 旅行日期」调用 ``get_weather`` 工具，得到每日天气，
再由 DeepSeek 为每一天生成一句**针对性的行程调整建议**（例如雷阵雨建议
把户外长城改到室内博物馆），最终输出结构化天气列表。

降级策略
--------
未配置 API Key / 模型异常时，直接使用工具内置的通用出行建议。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from trip_planner.models.schemas import WeatherInfo
from trip_planner.tools.travel_tools import get_weather, to_weather_models
from .base_agent import BaseAgent, extract_json

SYSTEM_PROMPT = """你是一位旅行天气顾问，擅长根据天气调整旅行计划。

你的工作流程：
1. 先调用 get_weather 工具获取目的地逐日天气（必须调用工具）。
2. 再为每一天给出一条中文出行建议，说明「这样的天气适合安排什么、要注意什么」。
3. 最终只输出 JSON，不要输出任何解释性文字、不要使用 Markdown 代码块。

输出 JSON 格式（字段名必须完全一致）：
{
  "daily": [
    {
      "date": "YYYY-MM-DD（必须与工具返回的日期完全一致）",
      "suggestion": "30 字以内的出行与行程调整建议"
    }
  ],
  "summary": "整体天气概况与打包提醒，50 字以内"
}

硬性约束：
- 严禁编造工具没有返回的日期或天气数据；
- 建议要具体到行程动作（如"带伞""改室内""注意防晒"），不要空话。
"""


class WeatherAgent(BaseAgent):
    """天气查询智能体。"""

    def __init__(self, enable_llm: bool = True) -> None:
        super().__init__(
            name="天气查询Agent",
            role="查询旅行期间逐日天气并给出出行建议",
            system_prompt=SYSTEM_PROMPT,
            tools=["get_weather"],
            temperature=0.2,
            use_json_mode=True,
            enable_llm=enable_llm,
        )

    # ------------------------------------------------------------------
    def run(self, destination: str, dates: List[str]) -> List[WeatherInfo]:
        """查询逐日天气。

        Args:
            destination: 目的地城市。
            dates: 日期字符串列表（YYYY-MM-DD）。

        Returns:
            WeatherInfo 列表，与 dates 一一对应（个别非法日期会被跳过）。
        """
        started = time.perf_counter()
        used_tools = ["get_weather"]
        status = "success"
        error: Optional[str] = None

        # ---- 第一步：调用工具（权威数据来源，永远可用）----
        raw_result: Dict[str, Any] = get_weather(destination, dates)
        weather_list = to_weather_models(raw_result.get("forecast", []))
        used_llm = False

        # ---- 第二步：让 LLM 生成针对性建议（可选增强）----
        if self.llm_available and weather_list:
            user_prompt = (
                f"目的地：{destination}\n"
                f"旅行日期：{', '.join(dates)}\n"
                f"工具返回的天气：\n"
                + "\n".join(
                    f"- {w.date}（{w.weekday}）{w.condition} {w.temp_range} {w.wind}"
                    for w in weather_list
                )
                + "\n\n请为每一天生成出行建议并输出约定的 JSON。"
            )
            result = self.run_with_tools(user_prompt)
            used_llm = bool(result.get("used_llm"))
            for call in result.get("tool_calls", []):
                if call.get("tool") not in used_tools:
                    used_tools.append(call.get("tool"))

            parsed = extract_json(result.get("content") or "")
            if isinstance(parsed, dict) and isinstance(parsed.get("daily"), list):
                weather_list = self._apply_suggestions(weather_list, parsed["daily"])
            else:
                status = "fallback"
                error = (
                    "模型未返回可解析的建议，沿用工具内置建议"
                    if used_llm
                    else "未启用或无法调用 DeepSeek，使用本地工具结果"
                )
        elif not weather_list:
            status = "fallback"
            error = "未查询到任何日期的天气数据"

        duration_ms = int((time.perf_counter() - started) * 1000)
        self.make_trace(
            status=status,
            duration_ms=duration_ms,
            tools=used_tools,
            summary=(
                f"{destination} 查询到 {len(weather_list)} 天天气"
                + ("（DeepSeek 已生成定制建议）" if used_llm and status == "success" else "（工具内置建议）")
            ),
            error=error,
        )
        return weather_list

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_suggestions(
        weather_list: List[WeatherInfo], raw_daily: List[Any]
    ) -> List[WeatherInfo]:
        """把 LLM 给出的每日建议写回天气模型（按日期匹配，匹配不上则保持原样）。"""
        by_date: Dict[str, str] = {}
        for entry in raw_daily:
            if not isinstance(entry, dict):
                continue
            date_key = str(entry.get("date", "")).strip()
            suggestion = str(entry.get("suggestion", "")).strip()
            if date_key and suggestion:
                by_date[date_key] = suggestion

        result: List[WeatherInfo] = []
        for item in weather_list:
            new_suggestion = by_date.get(item.date)
            if new_suggestion:
                item = item.model_copy(update={"suggestion": new_suggestion})
            result.append(item)
        return result
