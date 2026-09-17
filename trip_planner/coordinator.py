"""多智能体协调者（协调者-工作者模式）。

``MultiAgentTripPlanner`` 是整个系统的「大脑」，负责：

1. 接收用户输入（目的地、天数、预算、偏好）；
2. **并行**调度三个信息采集型智能体（景点 / 天气 / 酒店），它们之间无依赖，
   使用线程池并发执行以降低总耗时；
3. 把三个 Agent 的结构化输出**传递**给「行程规划 Agent」；
4. 汇总所有执行轨迹并返回最终的 ``TripPlan``。

架构图::

                      ┌──────────────────────────────┐
     用户 TripRequest │  MultiAgentTripPlanner       │
        ─────────────▶│  （协调者 Coordinator）       │
                      └───────────────┬──────────────┘
                                      │  并行派发（ThreadPool）
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ 景点搜索 Agent    │   │ 天气查询 Agent    │   │ 酒店推荐 Agent    │
    │ search_attractions│  │ get_weather       │   │ search_hotels    │
    └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
             │  List[Attraction]    │  List[WeatherInfo]   │  List[Hotel]
             └───────────────────────┼──────────────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │ 行程规划 Agent            │
                        │ 整合 → TripPlan           │
                        └──────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from .agents import AttractionAgent, HotelAgent, PlannerAgent, WeatherAgent
from .models.schemas import AgentTrace, TripPlan, TripRequest
from .tools.travel_tools import get_supported_destinations

logger = logging.getLogger(__name__)


class MultiAgentTripPlanner:
    """多智能体旅行规划协调者。

    Args:
        enable_llm: 是否启用 DeepSeek。设为 False 时全部走本地规则，
                    适合离线演示或单元测试。
        max_workers: 并行调用智能体的线程数。

    Example:
        >>> planner = MultiAgentTripPlanner()
        >>> plan = planner.plan(TripRequest(destination="北京", start_date=date(2025, 5, 1), days=3))
        >>> plan.daily_plans[0].morning
    """

    def __init__(self, enable_llm: bool = True, max_workers: int = 3) -> None:
        self.enable_llm = enable_llm
        self.max_workers = max_workers

        # 四个专用智能体
        self.attraction_agent = AttractionAgent(enable_llm=enable_llm)
        self.weather_agent = WeatherAgent(enable_llm=enable_llm)
        self.hotel_agent = HotelAgent(enable_llm=enable_llm)
        self.planner_agent = PlannerAgent(enable_llm=enable_llm)

        # 最近一次执行的轨迹（便于调试与前端展示）
        self.traces: List[AgentTrace] = []
        self.last_duration_ms: int = 0

    # ------------------------------------------------------------------
    # 预算档位推断
    # ------------------------------------------------------------------
    @staticmethod
    def derive_budget_level(request: TripRequest) -> str:
        """根据「总预算 + 天数 + 人数」推断合理的住宿档位。

        规则：人均每晚可支配预算（住宿占 40%）——
            < 400 元 → 经济；400 ~ 1000 元 → 中等；> 1000 元 → 豪华。

        若用户显式选择了档位且该档位与预算不冲突，则尊重用户选择；
        若预算明显无法支撑用户档位（例如"豪华"但人均每晚不足 400 元），
        自动下调档位，避免推荐出与预算严重不符的酒店。
        """
        nights = max(request.days - 1, 1)
        nightly_disposable = request.budget * 0.4 / nights / request.travelers
        if nightly_disposable < 400:
            derived = "经济"
        elif nightly_disposable <= 1000:
            derived = "中等"
        else:
            derived = "豪华"

        order = {"经济": 0, "中等": 1, "豪华": 2}
        # 用户档位高于预算可支撑的档位时，以预算为准
        if order[derived] < order.get(request.budget_level, 1):
            logger.info(
                "预算档位下调：用户选择 %s，按预算推断为 %s", request.budget_level, derived
            )
            return derived
        return request.budget_level

    @staticmethod
    def nightly_budget_limit(request: TripRequest) -> int:
        """人均每晚住宿预算上限（住宿按总预算的 40% 估算）。"""
        nights = max(request.days - 1, 1)
        return max(int(request.budget * 0.4 / nights / request.travelers), 100)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def _safe_result(self, future, agent, default):
        """读取线程池结果；失败时记录 failed 轨迹并返回默认值。"""
        try:
            return future.result()
        except Exception as exc:  # pragma: no cover - 防御性分支
            logger.exception("[%s] 执行失败：%s", agent.name, exc)
            self.traces.append(
                agent.make_trace(
                    status="failed",
                    duration_ms=0,
                    tools=agent.tools,
                    summary="智能体执行异常，已跳过并使用降级数据",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return default

    def plan(self, request: TripRequest) -> TripPlan:
        """同步执行完整的规划流程。

        Args:
            request: 用户旅行需求。

        Returns:
            包含每日行程、景点、天气、酒店、预算明细的 TripPlan。
        """
        overall_start = time.perf_counter()
        self.traces = []
        budget_level = self.derive_budget_level(request)
        nightly_limit = self.nightly_budget_limit(request)

        logger.info(
            "开始规划：%s %s 天 / %s 人 / 预算 %s 元 / 档位 %s",
            request.destination, request.days, request.travelers, request.budget, budget_level,
        )

        # ---- 第一阶段：并行调度三个采集型智能体 ----
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="agent") as pool:
            attraction_future = pool.submit(
                self.attraction_agent.run, request.destination, request.preferences, 8
            )
            weather_future = pool.submit(
                self.weather_agent.run, request.destination, request.date_strings()
            )
            hotel_future = pool.submit(
                self.hotel_agent.run, request.destination, budget_level, nightly_limit, 3
            )

            # 单个智能体失败不应该拖垮整个流程：捕获异常后使用空结果继续,
            # 由「行程规划 Agent」的本地规则引擎补全行程。
            attractions = self._safe_result(attraction_future, self.attraction_agent, [])
            weather = self._safe_result(weather_future, self.weather_agent, [])
            hotels = self._safe_result(hotel_future, self.hotel_agent, [])

        for agent in (self.attraction_agent, self.weather_agent, self.hotel_agent):
            existing = {trace.agent for trace in self.traces}
            if agent.last_trace is not None and agent.name not in existing:
                self.traces.append(agent.last_trace)

        # ---- 第二阶段：把三个 Agent 的输出交给行程规划 Agent ----
        plan = self.planner_agent.run(request, attractions, weather, hotels)
        if self.planner_agent.last_trace is not None:
            self.traces.append(self.planner_agent.last_trace)

        # ---- 汇总可观测信息 ----
        self.last_duration_ms = int((time.perf_counter() - overall_start) * 1000)
        plan.agent_traces = list(self.traces)

        # 把协调者层面的结论追加进注意事项
        plan.tips = [
            *plan.tips,
            f"本次规划由 {len(self.traces)} 个智能体协作完成，总耗时 {self.last_duration_ms} ms。",
        ]

        logger.info(
            "规划完成：总耗时 %s ms，生成 %s 天行程，预估花费 %s 元",
            self.last_duration_ms, len(plan.daily_plans), plan.estimated_total,
        )
        return plan

    async def plan_async(self, request: TripRequest) -> TripPlan:
        """异步版本的规划入口。

        FastAPI 是异步框架，但本项目中的智能体走的是同步 HTTP 客户端，
        因此用 ``run_in_executor`` 把同步逻辑挪到线程池，避免阻塞事件循环。
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.plan, request)

    # ------------------------------------------------------------------
    # 辅助信息
    # ------------------------------------------------------------------
    @property
    def supported_destinations(self) -> List[str]:
        """内置模拟数据覆盖的目的地。"""
        return get_supported_destinations()

    def describe_agents(self) -> List[Dict[str, str]]:
        """返回所有智能体的元信息，供 /api/agents 接口展示。"""
        agents = [self.attraction_agent, self.weather_agent, self.hotel_agent, self.planner_agent]
        return [
            {
                "name": agent.name,
                "role": agent.role,
                "tools": "、".join(agent.tools) if agent.tools else "（消费其它 Agent 的输出）",
                "llm_enabled": "是" if agent.llm_available else "否（本地规则）",
            }
            for agent in agents
        ]

    def health(self) -> Dict[str, object]:
        """健康检查信息。"""
        from .config import get_settings

        settings = get_settings()
        return {
            "status": "ok",
            "llm_available": settings.has_api_key and self.enable_llm,
            "model": settings.model,
            "base_url": settings.base_url,
            "api_key": settings.masked_key,
            "destinations": self.supported_destinations,
            "version": "1.0.0",
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
