"""智能体包。

四个专用智能体 + 一个可扩展的基类：

======================  ==========================================
智能体                    职责
======================  ==========================================
AttractionAgent         调用 search_attractions，检索并排序景点
WeatherAgent            调用 get_weather，查询逐日天气
HotelAgent              调用 search_hotels，按预算筛选酒店
PlannerAgent            整合前三者输出，生成最终行程
======================  ==========================================

协同方式：由 ``backend/coordinator.py`` 中的 ``MultiAgentTripPlanner``
按「协调者-工作者」模式依次/并行调度。
"""

from .attraction_agent import AttractionAgent  # noqa: F401
from .base_agent import BaseAgent, extract_json  # noqa: F401
from .hotel_agent import HotelAgent  # noqa: F401
from .planner_agent import PlannerAgent  # noqa: F401
from .weather_agent import WeatherAgent  # noqa: F401

__all__ = [
    "AttractionAgent",
    "BaseAgent",
    "HotelAgent",
    "PlannerAgent",
    "WeatherAgent",
    "extract_json",
]
