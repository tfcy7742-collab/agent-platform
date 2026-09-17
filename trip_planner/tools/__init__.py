"""旅行工具包（模拟数据实现，可直接替换为真实 API）。"""

from .travel_tools import (  # noqa: F401
    TOOL_REGISTRY,
    TOOL_SPECS,
    get_supported_destinations,
    get_weather,
    get_weather_tool_spec,
    search_attractions,
    search_attractions_tool_spec,
    search_hotels,
    search_hotels_tool_spec,
    to_attraction_models,
    to_hotel_models,
    to_weather_models,
    weather_text,
)

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_SPECS",
    "get_supported_destinations",
    "get_weather",
    "get_weather_tool_spec",
    "search_attractions",
    "search_attractions_tool_spec",
    "search_hotels",
    "search_hotels_tool_spec",
    "to_attraction_models",
    "to_hotel_models",
    "to_weather_models",
    "weather_text",
]
