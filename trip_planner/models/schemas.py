"""Pydantic 数据模型定义。

本模块定义了整个旅行规划系统中流转的所有数据结构：
1. 用户输入模型：TripRequest
2. 智能体工具返回的数据模型：Attraction / WeatherInfo / Hotel
3. 最终输出模型：DailyPlan / TripPlan / BudgetBreakdown
4. 观测模型：AgentTrace（记录每个智能体的执行耗时与状态）

所有模型都使用中文 description，FastAPI 会据此自动生成中文 OpenAPI 文档。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# 常量：合法偏好标签（前端多选框与后端校验共用同一份定义）
# ---------------------------------------------------------------------------
PREFERENCE_OPTIONS: List[str] = [
    "历史文化",
    "自然风光",
    "美食",
    "购物",
    "亲子",
    "摄影",
    "夜生活",
    "休闲度假",
]

# 预算档位
BUDGET_LEVELS: List[str] = ["经济", "中等", "豪华"]


# ---------------------------------------------------------------------------
# 1. 用户请求
# ---------------------------------------------------------------------------
class TripRequest(BaseModel):
    """用户提交的旅行需求。"""

    destination: str = Field(..., min_length=1, max_length=32, description="目的地城市，例如：北京")
    start_date: date = Field(..., description="出发日期，格式 YYYY-MM-DD")
    days: int = Field(3, ge=1, le=15, description="旅行天数，1~15 天")
    budget: int = Field(5000, ge=500, le=200000, description="总预算（人民币元）")
    budget_level: str = Field("中等", description="预算档位：经济 / 中等 / 豪华")
    preferences: List[str] = Field(default_factory=list, description="偏好标签列表")
    travelers: int = Field(1, ge=1, le=20, description="出行人数")
    notes: Optional[str] = Field(None, max_length=500, description="用户补充说明（可选）")

    @field_validator("destination")
    @classmethod
    def _strip_destination(cls, value: str) -> str:
        """去掉目的地首尾空格，避免 " 北京 " 这类输入导致匹配失败。"""
        value = value.strip()
        if not value:
            raise ValueError("目的地不能为空")
        return value

    @field_validator("budget_level")
    @classmethod
    def _check_budget_level(cls, value: str) -> str:
        """校验预算档位必须是预定枚举值之一。"""
        if value not in BUDGET_LEVELS:
            raise ValueError(f"预算档位必须是 {BUDGET_LEVELS} 之一，当前为 {value!r}")
        return value

    @field_validator("preferences")
    @classmethod
    def _check_preferences(cls, value: List[str]) -> List[str]:
        """过滤非法偏好标签并去重，保持顺序稳定。"""
        result: List[str] = []
        for item in value:
            item = item.strip()
            if not item:
                continue
            if item not in PREFERENCE_OPTIONS:
                raise ValueError(f"不支持的偏好标签：{item}，可选：{PREFERENCE_OPTIONS}")
            if item not in result:
                result.append(item)
        return result

    @model_validator(mode="after")
    def _check_end_date(self) -> "TripRequest":
        """校验结束日期不能超出合理范围（仅做提示性校验）。"""
        if self.days > 1:
            end = self.start_date + timedelta(days=self.days - 1)
            if end.year > self.start_date.year + 2:
                raise ValueError("行程跨度过于离谱，请检查天数与出发日期")
        return self

    @property
    def end_date(self) -> date:
        """行程结束日期（含当天）。"""
        return self.start_date + timedelta(days=self.days - 1)

    @property
    def date_list(self) -> List[date]:
        """行程覆盖的日期列表。"""
        return [self.start_date + timedelta(days=i) for i in range(self.days)]

    def date_strings(self) -> List[str]:
        """返回 YYYY-MM-DD 格式的日期字符串列表，用于调用天气工具。"""
        return [d.isoformat() for d in self.date_list]


# ---------------------------------------------------------------------------
# 2. 智能体工具产出的数据模型
# ---------------------------------------------------------------------------
class Attraction(BaseModel):
    """景点信息（由景点搜索 Agent 产出）。"""

    name: str = Field(..., description="景点名称")
    description: str = Field(..., description="景点简介")
    duration_hours: float = Field(2.0, ge=0, description="建议游览时长（小时）")
    ticket_price: int = Field(0, ge=0, description="门票价格（元，0 表示免费）")
    rating: float = Field(4.5, ge=0, le=5, description="评分（0~5）")
    tags: List[str] = Field(default_factory=list, description="景点标签")
    best_time: str = Field("全年", description="最佳游览时间")
    location: str = Field("", description="所在区域")


class WeatherInfo(BaseModel):
    """单日天气信息（由天气查询 Agent 产出）。"""

    date: str = Field(..., description="日期，格式 YYYY-MM-DD")
    weekday: str = Field("", description="星期几")
    condition: str = Field(..., description="天气状况，如：晴 / 多云 / 小雨")
    temp_min: int = Field(..., description="最低温度（摄氏度）")
    temp_max: int = Field(..., description="最高温度（摄氏度）")
    wind: str = Field("微风", description="风力")
    suggestion: str = Field("", description="出行建议")

    @property
    def temp_range(self) -> str:
        """温度区间文本，例如 12~24℃。"""
        return f"{self.temp_min}~{self.temp_max}℃"


class Hotel(BaseModel):
    """酒店信息（由酒店推荐 Agent 产出）。"""

    name: str = Field(..., description="酒店名称")
    price_per_night: int = Field(..., ge=0, description="每晚价格（元）")
    rating: float = Field(4.5, ge=0, le=5, description="评分（0~5）")
    location: str = Field("", description="位置 / 所在商圈")
    level: str = Field("中等", description="酒店档次：经济 / 中等 / 豪华")
    tags: List[str] = Field(default_factory=list, description="酒店标签")
    distance_to_center: str = Field("", description="距市中心距离描述")


# ---------------------------------------------------------------------------
# 3. 行程规划 Agent 的产出模型
# ---------------------------------------------------------------------------
class BudgetBreakdown(BaseModel):
    """预算明细（分项估算）。"""

    accommodation: int = Field(0, ge=0, description="住宿费用（元）")
    food: int = Field(0, ge=0, description="餐饮费用（元）")
    transportation: int = Field(0, ge=0, description="市内交通费用（元）")
    tickets: int = Field(0, ge=0, description="门票费用（元）")
    intercity: int = Field(0, ge=0, description="往返大交通费用（元）")
    shopping: int = Field(0, ge=0, description="购物与其它费用（元）")

    @property
    def total(self) -> int:
        """预算合计。"""
        return (
            self.accommodation
            + self.food
            + self.transportation
            + self.tickets
            + self.intercity
            + self.shopping
        )

    def as_dict(self) -> dict:
        """以字典形式返回，便于前端表格渲染。"""
        return {
            "住宿": self.accommodation,
            "餐饮": self.food,
            "市内交通": self.transportation,
            "景点门票": self.tickets,
            "往返大交通": self.intercity,
            "购物及其它": self.shopping,
            "合计": self.total,
        }


class DailyPlan(BaseModel):
    """单日行程安排。"""

    day: int = Field(..., ge=1, description="第几天")
    date: str = Field(..., description="日期 YYYY-MM-DD")
    theme: str = Field("", description="当日主题，例如：故宫与中轴线")
    morning: str = Field("", description="上午安排")
    afternoon: str = Field("", description="下午安排")
    evening: str = Field("", description="晚间安排")
    accommodation: str = Field("", description="住宿建议")
    meals: List[str] = Field(default_factory=list, description="推荐餐饮")
    transportation: str = Field("", description="交通方式建议")
    estimated_cost: int = Field(0, ge=0, description="当日预估花费（元）")
    weather_note: str = Field("", description="当日天气提示")
    tips: str = Field("", description="当日小贴士")


class TripPlan(BaseModel):
    """最终旅行计划（系统输出）。"""

    destination: str = Field(..., description="目的地")
    start_date: str = Field(..., description="出发日期")
    end_date: str = Field(..., description="结束日期")
    days: int = Field(..., description="天数")
    travelers: int = Field(1, description="出行人数")
    budget: int = Field(..., description="用户预算（元）")
    budget_level: str = Field("中等", description="预算档位")
    preferences: List[str] = Field(default_factory=list, description="用户偏好")
    summary: str = Field("", description="整体行程概述")
    daily_plans: List[DailyPlan] = Field(default_factory=list, description="每日行程")
    attractions: List[Attraction] = Field(default_factory=list, description="推荐景点")
    hotels: List[Hotel] = Field(default_factory=list, description="推荐酒店")
    weather: List[WeatherInfo] = Field(default_factory=list, description="每日天气")
    budget_breakdown: BudgetBreakdown = Field(
        default_factory=BudgetBreakdown, description="预算明细"
    )
    estimated_total: int = Field(0, ge=0, description="预估总花费（元）")
    budget_status: str = Field("", description="预算对比结论")
    budget_limit: int = Field(0, ge=0, description="预算硬上限（预算 × 1.1）")
    within_budget: bool = Field(True, description="预估总花费是否已控制在预算上限内")
    budget_fit: Dict[str, Any] = Field(
        default_factory=dict, description="预算收敛明细（是否调整、放弃了多少景点等）"
    )
    budget_floor: int = Field(0, ge=0, description="预算消费下限（预算 × 0.8）")
    within_floor: bool = Field(True, description="预估总花费是否达到预算消费下限")
    tips: List[str] = Field(default_factory=list, description="注意事项")
    generated_by: str = Field("", description="生成方式：DeepSeek LLM / 本地规则")
    generated_at: str = Field("", description="生成时间")
    agent_traces: List["AgentTrace"] = Field(
        default_factory=list, description="各智能体执行轨迹（可观测性）"
    )


# ---------------------------------------------------------------------------
# 4. 可观测性模型
# ---------------------------------------------------------------------------
class AgentTrace(BaseModel):
    """单个智能体的一次执行记录。"""

    agent: str = Field(..., description="智能体名称")
    role: str = Field("", description="智能体职责")
    status: str = Field("success", description="执行状态：success / fallback / failed")
    duration_ms: int = Field(0, ge=0, description="耗时（毫秒）")
    tools: List[str] = Field(default_factory=list, description="本次调用的工具")
    summary: str = Field("", description="执行结果摘要")
    error: Optional[str] = Field(None, description="失败原因（如有）")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field("ok", description="服务状态")
    llm_available: bool = Field(False, description="是否配置了 DeepSeek API Key")
    model: str = Field("", description="使用的模型名称")
    destinations: List[str] = Field(default_factory=list, description="内置模拟数据覆盖的城市")
    version: str = Field("1.0.0", description="服务版本")


# 解决 TripPlan -> AgentTrace 的前向引用
TripPlan.model_rebuild()
