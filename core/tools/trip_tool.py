"""trip_planner 工具：把多智能体旅行规划子系统包装成 Agent 可调用的工具。

这个工具是"平台能编排异构能力"的直接证据：
``knowledge_search`` 背后是"检索 + 生成"管道，``trip_planner`` 背后是
**4 个智能体并行协作**的任务型子系统。Planner 用同一套协议调用两者。

两条输入路径（都能用，互不冲突）
--------------------------------
1. **结构化参数**（推荐）：界面上的行程表单直接传
   ``destination / start_date / days / travelers / budget / budget_level / preferences``。
   参数由用户显式选定，不依赖模型或正则去猜，结果最可控。
2. **自然语言兜底**：用户只在对话里说"帮我规划北京三日游，预算 8000，两个人"时，
   工具自己从文本里解析出目的地、天数、人数、预算与偏好。

输出为什么不截断
----------------
第一版把 ``display`` 里的每日安排截成了"前 3 天 / 每天 40 字"，导致大模型最终
只能复述一个极简版本，用户反馈"比原项目单薄很多"。
这是适配层的偷懒，不是子系统能力不足——**子系统返回的每一天都是完整内容**。
现在 display 保留全部天数与完整时段描述，``data`` 里也带齐结构化明细，
让上层既能直接展示、也能喂给大模型。
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# 中文数字 → 阿拉伯数字（覆盖 1~15 天，够用且不会误伤"十"以外的表达）
CHINESE_NUMBERS: Dict[str, int] = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
    "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
}

# 偏好关键词 → 规范标签（与 trip_planner.models.schemas.PREFERENCE_OPTIONS 对齐）
PREFERENCE_KEYWORDS: Dict[str, List[str]] = {
    "历史文化": ["历史", "文化", "古迹", "古都", "人文", "博物馆"],
    "自然风光": ["自然", "风光", "山水", "风景", "海边", "爬山"],
    "美食": ["美食", "小吃", "吃", "餐厅", "火锅", "菜"],
    "购物": ["购物", "逛街", "买", "商场", "免税"],
    "亲子": ["亲子", "孩子", "小孩", "儿童", "溜娃"],
    "摄影": ["摄影", "拍照", "出片", "相机"],
    "夜生活": ["夜生活", "夜市", "酒吧", "夜景"],
    "休闲度假": ["休闲", "度假", "放松", "躺平", "慢"],
}

# 预算档位关键词
BUDGET_KEYWORDS: Dict[str, List[str]] = {
    "经济": ["经济", "省钱", "穷游", "便宜", "预算紧", "性价比", "学生"],
    "豪华": ["豪华", "高端", "五星", "奢侈", "不差钱", "奢华", "顶配"],
    "中等": ["中等", "舒适", "正常预算", "一般"],
}

# 预算金额：如"预算 8000"、"8000 元"
BUDGET_PATTERN = re.compile(r"(?:预算|花费|大概|大约)?\s*(\d{3,6})\s*(?:元|块|rmb|人民币)?", re.IGNORECASE)
DAYS_PATTERN = re.compile(r"(\d{1,2})\s*(?:天|日)")
CHINESE_DAYS_PATTERN = re.compile(r"([一二两三四五六七八九十]{1,3})\s*(?:天|日)")
# 人数：阿拉伯数字 + "人/位/个人"，或中文数字 + "人"
PEOPLE_PATTERN = re.compile(r"(\d{1,2})\s*(?:个)?\s*人")
CHINESE_PEOPLE_PATTERN = re.compile(r"([一二两三四五六七八九十]{1,3})\s*(?:个)?\s*人")

# 兜底解析城市名时要排除的词。
#
# 为什么需要：``_guess_destination`` 在文本里找不到词典城市时会用
# "XX + 旅游/行程"的前缀正则去猜地名，而中文里这类结构常常没有地名
# （"帮我规划一次旅行"）。实测把"一次"当成了城市名，进而生成了一份
# "一次 3 天行程"——用户看到会觉得系统坏了。
DESTINATION_BLOCKLIST = {
    # 数量 / 指代
    "一次", "这次", "那次", "上次", "下次", "一场", "一趟", "一个", "一种",
    "两天", "三天", "几天", "多天", "一日", "两日", "三日", "数日",
    # 泛化词 / 动作
    "国内", "国外", "周边", "附近", "本地", "外地", "当地", "我们", "你们",
    "帮我", "给我", "替我", "想要", "想去", "准备", "打算", "计划", "安排",
    "自己", "一起", "全家", "朋友", "同事", "家人", "孩子",
    # 与"游/行"组合后仍不构成地名的词
    "旅游", "旅行", "游玩", "行程", "出行", "度假", "自由", "深度", "亲子",
}

# 常见城市词典（用于从自由文本里识别目的地）。
#
# 为什么需要：子系统的内置数据只覆盖 5 个城市，但用户可能说任何城市。
# 这 5 个城市有专属数据，其余城市由子系统的兜底逻辑生成合理行程。
# 用"词典匹配"而不是正则猜词，是为了避免把"一次""周边"这类词当地名。
KNOWN_CITIES: List[str] = [
    "北京", "上海", "天津", "重庆", "广州", "深圳",
    "成都", "西安", "杭州", "南京", "苏州", "厦门", "青岛", "大连", "三亚",
    "丽江", "大理", "昆明", "桂林", "张家界", "黄山", "婺源", "拉萨", "西宁",
    "银川", "兰州", "乌鲁木齐", "哈尔滨", "长春", "沈阳", "呼和浩特", "太原",
    "石家庄", "济南", "郑州", "武汉", "长沙", "南昌", "福州", "合肥", "宁波",
    "无锡", "温州", "珠海", "佛山", "东莞", "汕头", "洛阳", "开封", "敦煌",
    "西双版纳", "香格里拉", "九寨沟", "稻城", "呼伦贝尔", "喀什", "伊犁",
    "秦皇岛", "承德", "平遥", "凤凰", "乌镇", "周庄", "千岛湖", "普陀山",
]

# 预算档位 → 人均每天参考预算（用户只给档位、不给金额时用来推算总额）
LEVEL_DAILY_BUDGET: Dict[str, int] = {"经济": 500, "中等": 1200, "豪华": 3000}


# ---------------------------------------------------------------------------
# 文本解析（自然语言兜底路径）
# ---------------------------------------------------------------------------
def parse_days(text: str) -> Optional[int]:
    """从文本里解析天数（支持 "3天" 与 "三天"）。"""
    if not text:
        return None
    match = DAYS_PATTERN.search(text)
    if match:
        days = int(match.group(1))
        return days if 1 <= days <= 15 else None
    match = CHINESE_DAYS_PATTERN.search(text)
    if match:
        days = CHINESE_NUMBERS.get(match.group(1))
        return days if days and 1 <= days <= 15 else None
    return None


def parse_travelers(text: str) -> Optional[int]:
    """从文本里解析出行人数（支持 "2人" "两个人" "3 位"）。

    注意：必须在解析天数之后使用——"2 人"与"2 天"共用数字，靠单位区分。
    """
    if not text:
        return None
    for pattern in (PEOPLE_PATTERN, CHINESE_PEOPLE_PATTERN):
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(1)
        value = int(raw) if raw.isdigit() else CHINESE_NUMBERS.get(raw)
        if value and 1 <= value <= 20:
            return value
    return None


def parse_preferences(text: str, limit: int = 3) -> List[str]:
    """从文本里解析偏好标签（最多 ``limit`` 个，保持稳定顺序）。"""
    if not text:
        return []
    found: List[str] = []
    for label, keywords in PREFERENCE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            found.append(label)
        if len(found) >= limit:
            break
    return found


def parse_budget_level(text: str) -> Optional[str]:
    """从文本里解析预算档位；没提及返回 ``None``（由调用方决定默认值）。"""
    if not text:
        return None
    for level, keywords in BUDGET_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return level
    return None


def strip_dates(text: str) -> str:
    """去掉文本里的日期片段。

    为什么必须做：日期里的数字会被预算解析误吃——实测 ``"2026-10-01 出发"``
    被解析成"预算 2026 元"。预算解析前先把日期抠掉即可根治。
    """
    if not text:
        return ""
    cleaned = re.sub(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?", " ", text)
    cleaned = re.sub(r"\d{1,2}\s*月\s*\d{1,2}\s*日", " ", cleaned)
    return cleaned


def parse_budget(text: str) -> Optional[int]:
    """从文本里解析预算金额（金额太小或太大时忽略，避免把天数当成预算）。

    解析前会先剔除日期片段，避免 "2026-10-01" 里的年份被当成预算。
    """
    text = strip_dates(text)
    if not text:
        return None
    for match in BUDGET_PATTERN.finditer(text):
        amount = int(match.group(1))
        # 800 ~ 200000 之间的数字才当成预算（"3天"是 1 位数，不会被误判）
        if 800 <= amount <= 200000:
            return amount
    return None


def parse_start_date(text: str) -> Optional[date]:
    """从文本里解析出发日期（``YYYY-MM-DD`` 或 ``YYYY年M月D日``）。"""
    if not text:
        return None
    for pattern in (r"(\d{4})-(\d{1,2})-(\d{1,2})", r"(\d{4})年(\d{1,2})月(\d{1,2})日"):
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
    return None


class TripPlannerTool(BaseTool):
    """多智能体旅行规划工具。"""

    name = "trip_planner"
    description = (
        "生成一份**完整的**旅行行程规划：逐日的上午/下午/晚间具体安排、景点与门票、"
        "逐日天气、酒店建议、预算明细与注意事项。"
        "适用于：用户想让你规划一次旅行/出行/旅游行程。"
        "参数尽量填全（目的地、天数、人数、预算、偏好），填得越全结果越贴合需求；"
        "只给 question 时工具会自己从文本里解析。"
        "不适用于：查询公司制度或产品文档（那应该用 knowledge_search）。"
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用户的原始需求描述，例如「帮我规划北京三日游，喜欢历史文化，两个人」；"
                "结构化参数缺失时，工具会从这里解析",
            },
            "destination": {"type": "string", "description": "目的地城市，例如：北京"},
            "start_date": {
                "type": "string",
                "description": "出发日期，格式 YYYY-MM-DD；不填则默认一周后",
            },
            "days": {
                "type": "integer",
                "description": "旅行天数，1~15",
                "minimum": 1,
                "maximum": 15,
            },
            "travelers": {
                "type": "integer",
                "description": "出行人数，1~20，默认 2",
                "minimum": 1,
                "maximum": 20,
            },
            "budget": {
                "type": "integer",
                "description": "总预算（元），500~200000",
                "minimum": 500,
                "maximum": 200000,
            },
            "budget_level": {
                "type": "string",
                "description": "预算档位（住宿与餐饮标准）",
                "enum": ["经济", "中等", "豪华"],
            },
            "preferences": {
                "type": "array",
                "description": "偏好标签列表，例如 [\"历史文化\",\"美食\"]",
            },
            "notes": {
                "type": "string",
                "description": "补充要求，例如「带老人，节奏慢一点」",
            },
        },
        "required": [],
    }
    timeout_s = 300.0     # 内部要跑 4 个智能体，开启大模型时可能多次调用模型，需要宽松上限
    est_cost = "high"     # 内部可能调用多次大模型
    est_latency = "high"
    retryable = False     # 重试等于重跑 4 个智能体，代价过高

    def __init__(self, planner: Any = None, enable_llm: Optional[bool] = None) -> None:
        super().__init__()
        self._planner = planner
        self._enable_llm = enable_llm
        # 允许通过配置覆盖超时（模型慢或快时可调）
        try:
            from config.settings import get_settings

            self.timeout_s = float(get_settings().trip_planner_timeout_s)
        except Exception:  # noqa: BLE001 - 配置不可用时用类默认值
            pass

    # ------------------------------------------------------------------
    @property
    def planner(self) -> Any:
        """延迟构造协调者（导入 trip_planner 会拉起 pydantic 模型）。"""
        if self._planner is None:
            from trip_planner import MultiAgentTripPlanner

            enable = self._enable_llm
            if enable is None:
                # 默认跟随平台模式：平台离线时，旅行规划也用本地规则引擎，
                # 避免"平台说自己在离线模式、却偷偷调了大模型"这种不一致。
                try:
                    from config.settings import get_settings

                    enable = get_settings().llm_online
                except Exception:  # noqa: BLE001
                    enable = False
            self._planner = MultiAgentTripPlanner(enable_llm=bool(enable))
            logger.info("旅行规划子系统就绪：enable_llm=%s", enable)
        return self._planner

    # ------------------------------------------------------------------
    def _run(
        self,
        question: str = "",
        destination: Optional[str] = None,
        start_date: Optional[str] = None,
        days: Optional[int] = None,
        travelers: Optional[int] = None,
        budget: Optional[int] = None,
        budget_level: Optional[str] = None,
        preferences: Optional[List[str]] = None,
        notes: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        """解析参数并调用旅行规划子系统。

        **参数优先级：结构化参数 > 文本解析 > 默认值**。
        这样"表单填写"与"对话里说需求"两条路径都能用，且表单永远优先。
        """
        text = question or ""

        # ---- 目的地：显式参数 > 从文本里找城市 ----
        resolved_destination = destination or self._guess_destination(text)
        if not resolved_destination:
            return ToolResult(
                ok=False,
                error="没有识别出目的地城市，请告诉我想去哪儿（例如：北京、成都）",
                error_type="tool_invalid_args",
                display="缺少目的地：请说明想去的城市",
            )

        # ---- 其余参数：显式 > 文本 > 默认 ----
        resolved_days = int(days) if days else (parse_days(text) or 3)
        resolved_days = max(1, min(resolved_days, 15))

        resolved_travelers = int(travelers) if travelers else (parse_travelers(text) or 2)
        resolved_travelers = max(1, min(resolved_travelers, 20))

        resolved_level = budget_level or parse_budget_level(text) or "中等"

        resolved_budget = int(budget) if budget else parse_budget(text)
        budget_source = "用户指定"
        if not resolved_budget:
            resolved_budget = LEVEL_DAILY_BUDGET.get(resolved_level, 1200) * resolved_days * resolved_travelers
            budget_source = "按档位推算"
        resolved_budget = max(500, min(resolved_budget, 200000))

        resolved_preferences = list(preferences) if preferences else parse_preferences(text)
        resolved_start = self._parse_date_value(start_date) or parse_start_date(text) or (
            date.today() + timedelta(days=7)
        )

        # ---- 构造请求并执行 ----
        from trip_planner import TripRequest

        try:
            request = TripRequest(
                destination=resolved_destination,
                start_date=resolved_start,
                days=resolved_days,
                budget=resolved_budget,
                budget_level=resolved_level,
                preferences=resolved_preferences,
                travelers=resolved_travelers,
                notes=(notes or "").strip()[:500] or None,
            )
        except Exception as exc:  # noqa: BLE001 - 参数非法时给出可读提示
            return ToolResult(
                ok=False,
                error=f"行程参数不合法：{exc}",
                error_type="tool_invalid_args",
                display=f"行程参数不合法：{exc}",
            )

        plan = self.planner.plan(request)

        # ---- 组装返回值（**不截断**：完整内容既给模型也给界面）----
        daily_lines: List[str] = []
        for item in plan.daily_plans:
            daily_lines.append(
                f"【第 {item.day} 天】{item.date}｜{item.theme}\n"
                f"  上午：{item.morning}\n"
                f"  下午：{item.afternoon}\n"
                f"  晚间：{item.evening}\n"
                f"  住宿：{item.accommodation}\n"
                f"  交通：{item.transportation}\n"
                + (f"  天气：{item.weather_note}\n" if item.weather_note else "")
                + (f"  当日花费：约 {item.estimated_cost} 元\n" if item.estimated_cost else "")
                + (f"  贴士：{item.tips}" if item.tips else "")
            )

        display = (
            f"已生成 {plan.destination} {plan.days} 天行程（{plan.travelers} 人，"
            f"{plan.budget_level}预算，总预算 {plan.budget} 元，"
            f"预算区间 {plan.budget_floor} ~ {plan.budget_limit} 元，"
            f"预估总花费 {plan.estimated_total} 元，"
            f"{'已控制在预算区间内' if plan.within_budget and plan.within_floor else ('超出预算上限' if not plan.within_budget else '未达消费下限')}）\n"
            f"{plan.summary}\n\n"
            + "\n\n".join(daily_lines)
            + f"\n\n预算明细：{plan.budget_breakdown.as_dict()}"
            + f"\n{plan.budget_status}"
        )

        return ToolResult(
            ok=True,
            data={
                "destination": plan.destination,
                "start_date": plan.start_date,
                "end_date": plan.end_date,
                "days": plan.days,
                "travelers": plan.travelers,
                "budget": plan.budget,
                "budget_level": plan.budget_level,
                "preferences": plan.preferences,
                "summary": plan.summary,
                "daily_plans": [
                    {
                        "day": item.day,
                        "date": item.date,
                        "theme": item.theme,
                        "morning": item.morning,
                        "afternoon": item.afternoon,
                        "evening": item.evening,
                        "accommodation": item.accommodation,
                        "meals": item.meals,
                        "transportation": item.transportation,
                        "estimated_cost": item.estimated_cost,
                        "weather_note": item.weather_note,
                        "tips": item.tips,
                    }
                    for item in plan.daily_plans
                ],
                "attractions": [
                    {
                        "name": item.name,
                        "description": item.description,
                        "duration_hours": item.duration_hours,
                        "ticket_price": item.ticket_price,
                        "rating": item.rating,
                        "tags": item.tags,
                        "location": item.location,
                    }
                    for item in plan.attractions
                ],
                "hotels": [
                    {
                        "name": item.name,
                        "price_per_night": item.price_per_night,
                        "rating": item.rating,
                        "location": item.location,
                        "level": item.level,
                        "tags": item.tags,
                        "distance_to_center": item.distance_to_center,
                    }
                    for item in plan.hotels
                ],
                "weather": [
                    {
                        "date": item.date,
                        "weekday": item.weekday,
                        "condition": item.condition,
                        "temp_min": item.temp_min,
                        "temp_max": item.temp_max,
                        "wind": item.wind,
                        "suggestion": item.suggestion,
                    }
                    for item in plan.weather
                ],
                "budget_breakdown": plan.budget_breakdown.as_dict(),
                "estimated_total": plan.estimated_total,
                "budget_status": plan.budget_status,
                # 预算区间约束（预算 × 0.8 ≤ 总花费 ≤ 预算 × 1.1）的收敛结果
                "budget_limit": plan.budget_limit,
                "budget_floor": plan.budget_floor,
                "within_budget": plan.within_budget,
                "within_floor": plan.within_floor,
                "budget_fit": plan.budget_fit,
                "tips": plan.tips,
                "generated_by": plan.generated_by,
            },
            display=display,
            degraded=not plan.generated_by.startswith("DeepSeek"),
            meta={
                "agent_traces": [
                    {
                        "agent": trace.agent,
                        "status": trace.status,
                        "duration_ms": trace.duration_ms,
                        "summary": trace.summary,
                    }
                    for trace in plan.agent_traces
                ],
                "sub_agents": len(plan.agent_traces),
                "estimated_total": plan.estimated_total,
                "budget_limit": plan.budget_limit,
                "budget_floor": plan.budget_floor,
                "within_budget": plan.within_budget,
                "within_floor": plan.within_floor,
                "budget_fit": plan.budget_fit,
                # 把"实际用了哪些参数"如实回传，便于用户核对（也便于排查解析错误）
                "parsed": {
                    "destination": resolved_destination,
                    "start_date": resolved_start.isoformat(),
                    "days": resolved_days,
                    "travelers": resolved_travelers,
                    "budget": resolved_budget,
                    "budget_source": budget_source,
                    "budget_level": resolved_level,
                    "preferences": resolved_preferences,
                    "notes": request.notes,
                },
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_date_value(value: Optional[str]) -> Optional[date]:
        """解析界面传来的日期字符串（兼容 YYYY-MM-DD 与 ISO 时间戳）。"""
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "")).date()
        except ValueError:
            pass
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"):
            try:
                return datetime.strptime(text, pattern).date()
            except ValueError:
                continue
        return None

    def _guess_destination(self, text: str) -> Optional[str]:
        """从文本里找目的地城市。

        解析优先级（从可靠到宽松）：

        1. **内置城市**（有完整数据）：北京/上海/成都/西安/杭州；
        2. **常见城市词典**（约 60 个国内热门目的地）——没有专属数据，
           但子系统的兜底逻辑能生成合理行程，比"猜错一个不存在的城市"好；
        3. **带行政区后缀的地名**（"大理市""余杭区"）；
        4. **"去/到/往 + 地名"结构**；
        5. 最后才是"XX + 三日游/旅游"结构（最容易误判，有排除表与虚词过滤）。

        拿不准时**返回 None**，由工具给出"请说明目的地"的提示——
        宁可多问一句，也不要编出一个城市然后生成一份假行程。
        """
        if not text:
            return None

        # 1) 内置城市优先（子系统的 5 个城市有专属数据）
        try:
            from trip_planner.tools.travel_tools import get_supported_destinations

            builtin = get_supported_destinations()
        except Exception:  # noqa: BLE001 - 子系统不可用时退化为通用匹配
            builtin = ["北京", "上海", "成都", "西安", "杭州"]

        matched = [(text.find(city), city) for city in builtin if city in text]
        if matched:
            matched.sort()
            return matched[0][1]

        # 2) 常见城市词典（取最先出现的那个）
        common = [(text.find(city), city) for city in KNOWN_CITIES if city in text]
        if common:
            common.sort()
            return common[0][1]

        # 3) 带行政区后缀的地名（贪婪匹配，把后缀一起带上）
        suffix_match = re.search(r"([\u4e00-\u9fff]{2,6}?(?:市|县|区|州|镇))", text)
        if suffix_match:
            candidate = suffix_match.group(1)
            if candidate not in DESTINATION_BLOCKLIST and not self._has_stopword(candidate):
                return candidate

        # 4) "去/到/往 + 地名"结构（"想去拉萨玩五天"）
        verb_match = re.search(
            r"(?:去|到|往|在)\s*([\u4e00-\u9fff]{2,4}?)"
            r"(?:玩|旅游|旅行|游玩|出差|度假|住|待|呆|看|吃|的|，|,|。|、|$)",
            text,
        )
        if verb_match:
            candidate = verb_match.group(1)
            if candidate not in DESTINATION_BLOCKLIST and not self._has_stopword(candidate):
                return candidate

        # 5) "XX + 三日游/旅游/行程"（风险最高，放最后）
        match = re.search(
            r"([\u4e00-\u9fff]{2,4}?)(?:三日|两日|一日|\d+\s*[天日]|旅游|旅行|行程|游玩|攻略)",
            text,
        )
        if not match:
            return None

        candidate = match.group(1)
        for filler in ("帮我", "给我", "替我", "我要", "想去", "计划", "规划", "安排", "准备"):
            candidate = candidate.replace(filler, "")
        candidate = candidate.strip()

        if not candidate or candidate in DESTINATION_BLOCKLIST or self._has_stopword(candidate):
            return None
        return candidate

    @staticmethod
    def _has_stopword(text: str) -> bool:
        """候选地名里含有虚词（"的""了""一"…）时，几乎不可能是地名。"""
        try:
            from rag.retriever import STOPWORDS
        except Exception:  # noqa: BLE001
            STOPWORDS = set()  # type: ignore[assignment]
        return any(char in STOPWORDS for char in text)


__all__ = [
    "CHINESE_NUMBERS",
    "KNOWN_CITIES",
    "LEVEL_DAILY_BUDGET",
    "PREFERENCE_KEYWORDS",
    "TripPlannerTool",
    "parse_budget",
    "parse_budget_level",
    "parse_days",
    "parse_preferences",
    "parse_start_date",
    "parse_travelers",
    "strip_dates",
]
