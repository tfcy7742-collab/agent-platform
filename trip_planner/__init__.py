"""旅行规划子系统（从 trip-planner 项目迁移而来）。

对外入口：
    from trip_planner import MultiAgentTripPlanner, TripRequest

它在本平台中作为 ``trip_planner`` 工具被 Agent 调用：
用户说「帮我规划北京三日游」时，Planner 会路由到这个子系统，
由内部 4 个智能体（景点/天气/酒店/行程规划）协作产出完整行程。
"""

from .coordinator import MultiAgentTripPlanner
from .models.schemas import TripPlan, TripRequest

__all__ = ["MultiAgentTripPlanner", "TripPlan", "TripRequest"]
