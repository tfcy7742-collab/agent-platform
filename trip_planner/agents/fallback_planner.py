"""本地规则行程生成器（降级方案）。

当 DeepSeek 不可用（未配置 API Key、网络异常、模型返回非法 JSON）时，
本模块用**纯 Python 规则**生成一份结构与质量都完整的旅行计划，
确保系统在任何环境下都能返回可用的结果。

规则说明
--------
* 每天固定拆分为 上午 / 下午 / 晚间 三段，外加餐饮、住宿、交通建议；
* 优先使用「景点搜索 Agent」返回的真实候选景点（含门票、时长）；
* 候选景点不足时，使用内置的展示型活动池补齐，保证每日三段都不为空；
* 日花费 = 餐饮与市内交通 + 当日景点门票；
* 预算分项按人数、天数、预算档位确定性推算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from trip_planner.models.schemas import Attraction, BudgetBreakdown, DailyPlan, TripRequest

# ---------------------------------------------------------------------------
# 展示型活动池：行程小时数不足时用于补齐"上午/下午/晚间"三段
# ---------------------------------------------------------------------------
ACTIVITY_POOL: Dict[str, List[Dict[str, str]]] = {
    "北京": [
        {"name": "老北京胡同早餐", "kind": "美食", "time": "morning", "note": "豆汁焦圈、炒肝配包子，8:00 前后人少不用排队。"},
        {"name": "城市漫步・中轴线骑行", "kind": "休闲度假", "time": "afternoon", "note": "沿中轴线骑行，串联钟鼓楼与景山，节奏自由。"},
        {"name": "王府井 / 前门大街夜逛", "kind": "购物", "time": "evening", "note": "老字号点心与伴手礼集中，注意比价。"},
        {"name": "老舍茶馆听相声、看变脸", "kind": "历史文化", "time": "evening", "note": "提前 1 天订票，前两排体验最好。"},
        {"name": "三里屯 / 国贸夜景", "kind": "夜生活", "time": "evening", "note": "城市天际线夜景，餐吧选择丰富。"},
    ],
    "上海": [
        {"name": "弄堂早餐・四大金刚", "kind": "美食", "time": "morning", "note": "大饼油条豆浆粢饭团，本地人才懂的打开方式。"},
        {"name": "梧桐区 City Walk", "kind": "摄影", "time": "afternoon", "note": "武康路—安福路一带，老洋房与咖啡馆密度极高。"},
        {"name": "黄浦江夜游船", "kind": "夜生活", "time": "evening", "note": "从水上同时看外滩与陆家嘴，建议选 19:00 后班次。"},
        {"name": "南京路步行街购物", "kind": "购物", "time": "afternoon", "note": "老字号与旗舰店集中，晚间灯光更好看。"},
    ],
    "成都": [
        {"name": "人民公园盖碗茶与采耳", "kind": "休闲度假", "time": "morning", "note": "鹤鸣茶社坐一上午，是成都的正确打开方式。"},
        {"name": "川剧变脸专场", "kind": "历史文化", "time": "evening", "note": "建议提前订票，含变脸、吐火、手影戏。"},
        {"name": "玉林路小酒馆与串串", "kind": "夜生活", "time": "evening", "note": "成都夜宵代表，微辣锅底更贴近本地口味。"},
        {"name": "太古里商圈逛街", "kind": "购物", "time": "afternoon", "note": "IFS 熊猫爬楼是必拍机位。"},
    ],
    "西安": [
        {"name": "永兴坊摔碗酒与小吃", "kind": "美食", "time": "evening", "note": "一站式吃遍陕西各地小吃。"},
        {"name": "城墙根晨练与早市", "kind": "休闲度假", "time": "morning", "note": "感受本地生活节奏，顺路吃一碗胡辣汤。"},
        {"name": "大唐不夜城夜游", "kind": "夜生活", "time": "evening", "note": "灯光演艺与街头表演密集，人多注意随身物品。"},
    ],
    "杭州": [
        {"name": "西湖边晨跑 / 骑行", "kind": "自然风光", "time": "morning", "note": "清晨苏堤人少，光线最好。"},
        {"name": "龙井村茶山徒步", "kind": "自然风光", "time": "afternoon", "note": "可体验采茶炒茶，注意防晒。"},
        {"name": "武林夜市 / 湖滨步行街", "kind": "夜生活", "time": "evening", "note": "夜市小吃与商圈购物结合。"},
    ],
}

# 通用活动池（未知城市兜底）
GENERIC_ACTIVITY_POOL: List[Dict[str, str]] = [
    {"name": "本地特色早餐", "kind": "美食", "time": "morning", "note": "找一家人多的本地早餐店，通常是性价比最高的选择。"},
    {"name": "城市街区漫步", "kind": "摄影", "time": "afternoon", "note": "放慢节奏，感受当地生活气息。"},
    {"name": "本地夜市与小吃街", "kind": "夜生活", "time": "evening", "note": "夜宵是了解一座城市最快的方式。"},
]

# 预算档位对应的默认人均每晚酒店价格（用于无法从酒店 Agent 拿到数据时）
DEFAULT_NIGHTLY_PRICE: Dict[str, int] = {"经济": 280, "中等": 700, "豪华": 1900}

# 预算档位对应的每日餐饮 + 市内交通基准（每人每天）
DAILY_MEAL_BASE: Dict[str, int] = {"经济": 120, "中等": 220, "豪华": 480}
DAILY_TRANSPORT_BASE: Dict[str, int] = {"经济": 40, "中等": 80, "豪华": 160}

# 预算档位对应的往返大交通基准（每人）
INTERCITY_BASE: Dict[str, int] = {"经济": 600, "中等": 1200, "豪华": 2600}
INTERCITY_HIGH_SPEED_CITIES = {"北京", "上海", "成都", "西安", "杭州", "南京", "广州", "深圳", "重庆", "武汉"}

#: 硬约束：预估总花费不得超过「用户预算 × 该系数」，即最多超支 10%
BUDGET_LIMIT_RATIO = 1.1

#: 硬约束下限：预估总花费不得低于「用户预算 × 该系数」，即至少花掉 80%
BUDGET_FLOOR_RATIO = 0.8

#: 区间搜索时的房价步长（比整百步长更细，便于精确落进"预算区间"）
FINE_PRICE_STEP = 10

#: 档位由高到低，预算收敛时按此顺序降级
LEVEL_ORDER: List[str] = ["豪华", "中等", "经济"]

#: 每次切换档位时，每晚房价的下调步长（元）；降到 0 表示改用该档位默认价
PRICE_STEP = 100

# ---------------------------------------------------------------------------
# 行程生成
# ---------------------------------------------------------------------------
def _pick_activity(pool: List[Dict[str, str]], slot: str, used: set) -> Optional[Dict[str, str]]:
    """从活动池中挑一条指定时段且未使用过的活动。"""
    for item in pool:
        if item["time"] == slot and item["name"] not in used:
            used.add(item["name"])
            return item
    for item in pool:  # 允许跨时段复用，保证不为空
        if item["name"] not in used:
            used.add(item["name"])
            return item
    return None


def _format_attraction_activity(attraction: Attraction, destination: str) -> str:
    """把景点格式化成一句行程描述。"""
    ticket = "免费" if attraction.ticket_price == 0 else f"门票 {attraction.ticket_price} 元"
    return (
        f"游览「{attraction.name}」（{attraction.location or destination}，"
        f"建议 {attraction.duration_hours:g} 小时，{ticket}，评分 {attraction.rating}）"
        f"——{attraction.description}"
    )


def _format_pool_activity(item: Dict[str, str], destination: str) -> str:
    """把活动池条目格式化成一句行程描述。"""
    return f"{item['name']}（{item['kind']}）——{item['note']}"


def build_daily_plans(
    request: TripRequest,
    attractions: List[Attraction],
    weather_texts: Optional[Dict[str, str]] = None,
    hotel_name: str = "",
    nightly_price: int = 0,
) -> List[DailyPlan]:
    """用本地规则生成逐日行程。

    Args:
        request: 用户请求。
        attractions: 景点搜索 Agent 返回的候选景点。
        weather_texts: ``{日期: 天气一行文本}``。
        hotel_name: 推荐住宿名称（会在每天"住宿"字段中体现）。
        nightly_price: 每晚房价，用于计算当日花费。

    Returns:
        DailyPlan 列表，长度为 ``request.days``。
    """
    weather_texts = weather_texts or {}
    pool = ACTIVITY_POOL.get(request.destination, GENERIC_ACTIVITY_POOL)
    used_activities: set = set()
    attraction_cursor = 0
    daily_plans: List[DailyPlan] = []
    dates = request.date_strings()

    # 把门票与景点按"天"分组：每天 2 个景点（上午 1 个、下午 1 个）
    for index in range(request.days):
        date_str = dates[index]
        # 首日/末日自动加上交通说明
        is_first, is_last = index == 0, index == request.days - 1
        morning_attraction = attractions[attraction_cursor] if attraction_cursor < len(attractions) else None
        attraction_cursor += 1
        afternoon_attraction = attractions[attraction_cursor] if attraction_cursor < len(attractions) else None
        attraction_cursor += 1

        ticket_cost = sum(
            a.ticket_price for a in (morning_attraction, afternoon_attraction) if a is not None
        )

        # ---- 上午 ----
        if morning_attraction is not None:
            morning = _format_attraction_activity(morning_attraction, request.destination)
            if is_first:
                morning = f"抵达 {request.destination} 后先办理入住或寄存行李，随后{morning}"
        else:
            picked = _pick_activity(pool, "morning", used_activities)
            morning = (
                _format_pool_activity(picked, request.destination)
                if picked
                else f"{request.destination}市区自由漫步，感受当地生活节奏"
            )

        # ---- 下午 ----
        if afternoon_attraction is not None:
            afternoon = _format_attraction_activity(afternoon_attraction, request.destination)
        else:
            picked = _pick_activity(pool, "afternoon", used_activities)
            afternoon = (
                _format_pool_activity(picked, request.destination)
                if picked
                else f"逛 {request.destination} 特色街区，选购伴手礼"
            )

        # ---- 晚间 ----
        evening_item = _pick_activity(pool, "evening", used_activities)
        if evening_item is not None:
            evening = _format_pool_activity(evening_item, request.destination)
        else:
            evening = f"在 {request.destination} 品尝当地特色晚餐，早点休息为次日行程储备体力"

        # ---- 主题 ----
        names = [a.name for a in (morning_attraction, afternoon_attraction) if a is not None]
        if names:
            theme = " + ".join(names)
        elif is_first:
            theme = f"{request.destination}初印象"
        elif is_last:
            theme = "收尾与返程"
        else:
            theme = f"{request.destination}城市深度体验"
        if is_first:
            theme = f"抵达日｜{theme}"
        if is_last:
            theme = f"{theme}｜返程日"

        # ---- 餐饮 / 交通 / 花费 ----
        meal_base = DAILY_MEAL_BASE.get(request.budget_level, 220)
        transport_base = DAILY_TRANSPORT_BASE.get(request.budget_level, 80)
        meals = [f"早餐：酒店或附近本地早餐店（约 {int(meal_base * 0.2 * request.travelers)} 元）"]
        meals.append(
            f"{'午餐' if is_first else '午餐'}：{names[0] if names else request.destination}附近特色餐厅"
            f"（约 {int(meal_base * 0.35 * request.travelers)} 元）"
        )
        meals.append(
            f"晚餐：{evening_item['name'] if evening_item else '本地口碑餐厅'}"
            f"（约 {int(meal_base * 0.45 * request.travelers)} 元）"
        )
        transportation = (
            f"地铁 + 步行 + 打车组合，人均约 {transport_base} 元"
            if not is_first
            else f"机场/车站往返市区 + 市内地铁，人均约 {int(transport_base * 1.8)} 元"
        )

        daily_cost = int(
            (meal_base + transport_base * (1.8 if is_first else 1.0)) * request.travelers
            + ticket_cost * request.travelers
            + (nightly_price if not is_last else 0)
        )

        weather_note = weather_texts.get(date_str, "")
        tips_parts: List[str] = []
        if is_first:
            tips_parts.append("首日行程安排较松，优先解决落地交通与入住。")
        if is_last:
            tips_parts.append("返程日请预留 2 小时前往机场/高铁站，行李可寄存酒店。")
        if "雨" in weather_note:
            tips_parts.append("当天有降水，随身带伞并把室内展馆作为备选。")
        if "晴" in weather_note:
            tips_parts.append("紫外线较强，注意防晒补水。")
        if morning_attraction and morning_attraction.ticket_price > 0:
            tips_parts.append(f"{morning_attraction.name}建议提前在官方渠道预约门票。")

        daily_plans.append(
            DailyPlan(
                day=index + 1,
                date=date_str,
                theme=theme,
                morning=morning,
                afternoon=afternoon,
                evening=evening,
                accommodation=hotel_name or f"{request.destination}{request.budget_level}档酒店",
                meals=meals,
                transportation=transportation,
                estimated_cost=daily_cost,
                weather_note=weather_note,
                tips=" ".join(tips_parts) or "行程节奏适中，可按当天体力灵活调整。",
            )
        )

    return daily_plans

# ---------------------------------------------------------------------------
# 预算估算
# ---------------------------------------------------------------------------
# 预算估算
# ---------------------------------------------------------------------------
def estimate_budget(
    request: TripRequest,
    nightly_price: int = 0,
    attraction_total: int = 0,
    period_ratio: float = 1.0,
) -> BudgetBreakdown:
    """按人数、天数与预算档位推算预算明细。

    Args:
        request: 用户请求。
        nightly_price: 单人每晚房价（0 表示用档位默认值）。
        attraction_total: 行程中涉及的景点门票合计（每人）。
        period_ratio: 可压缩项（住宿/餐饮/市内交通/购物）的缩放系数，
            仅在预算收敛时小于 1（见 ``fit_budget``）。

    Returns:
        BudgetBreakdown。
    """
    level = request.budget_level
    travelers = request.travelers
    days = request.days
    nights = max(days - 1, 1)

    price = nightly_price or DEFAULT_NIGHTLY_PRICE.get(level, 700)

    meal_base = DAILY_MEAL_BASE.get(level, 220)
    transport_base = DAILY_TRANSPORT_BASE.get(level, 80)
    intercity_base = INTERCITY_BASE.get(level, 1200)

    # 往返大交通：非一线城市枢纽时按 8 折估算
    intercity_unit = int(intercity_base * (1.0 if request.destination in INTERCITY_HIGH_SPEED_CITIES else 0.8))

    breakdown = BudgetBreakdown(
        accommodation=price * nights * travelers,
        food=int(meal_base * days * travelers),
        transportation=int(transport_base * days * travelers),
        tickets=attraction_total * travelers,
        intercity=intercity_unit * travelers,
        shopping=int(meal_base * days * travelers * 0.3),
    )
    return _scale_accommodation(breakdown, request, period_ratio)


def budget_limit(budget: int) -> int:
    """允许的总花费上限 = 预算 × 1.1（严格上限，预估合计不得突破）。"""
    return int(budget * BUDGET_LIMIT_RATIO)


def budget_floor(budget: int) -> int:
    """要求的总花费下限 = 预算 × 0.8（花太少也算不达标）。"""
    return int(budget * BUDGET_FLOOR_RATIO)


def _scale_accommodation(
    breakdown: BudgetBreakdown, request: TripRequest, period_ratio: float
) -> BudgetBreakdown:
    """按 ``period_ratio`` 收缩"可压缩项"，使合计尽量落在预算上限内。

    可压缩项 = 住宿 + 餐饮 + 市内交通 + 购物；
    **门票与往返大交通保持真实值**——门票要么不去这个景点，要么按票价花钱，
    大交通更是不可压缩，缩放它们会让预算明细失真。

    先按住宿单独压缩（性价比最高：住宿通常是弹性最大的一项），
    仍不够时再等比压缩其余各项。
    """
    if period_ratio >= 1.0:
        return breakdown

    nights = max(request.days - 1, 1)
    travelers = request.travelers
    ratio = max(period_ratio, 0.0)

    accommodation = int(breakdown.accommodation * ratio)
    if ratio > 0:
        # 保证人均每晚不出现 0 元这种不真实的结果
        accommodation = max(accommodation, nights * travelers)
    remainder = breakdown.total - breakdown.accommodation - accommodation

    food = breakdown.food
    transportation = breakdown.transportation
    shopping = breakdown.shopping
    compressible = food + transportation + shopping

    if remainder > 0 and compressible > 0:
        squeeze = min(remainder / compressible, 1.0)
        food = int(food * (1 - squeeze))
        transportation = int(transportation * (1 - squeeze))
        shopping = int(shopping * (1 - squeeze))

    return BudgetBreakdown(
        accommodation=accommodation,
        food=food,
        transportation=transportation,
        tickets=breakdown.tickets,
        intercity=breakdown.intercity,
        shopping=shopping,
    )


@dataclass
class BudgetFit:
    """预算收敛结果。"""

    request: TripRequest
    breakdown: BudgetBreakdown
    attraction_total: int
    nightly_price: int
    priced_attraction_count: int
    #: 调整档位后的有效请求（档位可能被下调，budget_level 随之下调）
    effective_request: TripRequest
    #: 命中的缩放系数（1.0 表示无需压缩）
    period_ratio: float = 1.0
    #: 是否成功把总花费压进「预算 × 1.1」
    within_budget: bool = True
    #: 是否达到「预算 × 0.8」的消费下限
    within_floor: bool = True
    #: 未达下限时的"最多可花"金额；达下限时为 0
    max_spendable: int = 0
    #: 是否对用户指定的预算做了收敛（未指定预算 / 本来就够花时为 False）
    adjusted: bool = False
    #: 用户显式选择了档位、但为了预算被下调
    level_downgraded: bool = False
    #: 为了压预算放弃的付费景点数量
    dropped_attractions: int = 0
    #: 收敛时实际采用（或建议采用）的酒店名称
    hotel_name: str = ""
    #: 逐日行程是否需要按收敛结果重新生成
    regenerate_needed: bool = False
    #: 无论怎么压缩都超出预算时的理论最低花费
    min_feasible_total: int = 0
    #: 面向用户的说明文本
    notes: str = ""

    @property
    def total(self) -> int:
        return self.breakdown.total


def fit_budget(
    request: TripRequest,
    attractions: List[Attraction],
    nightly_price: int = 0,
    hotel_name: str = "",
    candidates: Optional[List[Any]] = None,
    tol: float = 1e-6,
) -> BudgetFit:
    """把预估总花费收敛到「预算 × 1.1」以内（硬约束）。

    为什么需要它：``estimate_budget`` 是"按档位查单价表"正向算出来的，
    它从不读 ``request.budget``，所以用户预算只是个被汇报的数字，
    算出来超了就只在结论里写一句"建议节约"。本函数补上这个闭环。

    收敛旋钮只有两个（也是 ``estimate_budget`` 仅有的两个真实价格输入）：

    1. **降住宿档 / 降房价**：按 豪华 → 中等 → 经济 逐档下调，档内再按
       ``PRICE_STEP`` 逐步下调每晚房价，直到命中目标花费；
    2. **放弃付费景点**：门票按真实票价计入，压不下来时优先放弃最贵的景点
       （至少保留 1 个，"一个景点都不去"不是一份可用行程）。

    收敛是**先算后量**：定价阶段就检查"命中后会不会花太多"，避免出现
    "预算明细按便宜酒店算、逐日行程却写着贵酒店"的不一致。

    Args:
        request: 用户请求（``budget`` 即用户预算）。
        attractions: 候选景点，按推荐优先级排序（越靠前越优先保留）。
        nightly_price: 已选酒店的每晚房价（0 表示未选，用档位默认价）。
        hotel_name: 已选酒店名称，收敛后会返回建议替换的酒店名。
        candidates: 候选酒店列表（含 ``name`` / ``price_per_night``），
            用于在降级后挑一家真实存在且更便宜的酒店。
        tol: 浮点比较容差。

    Returns:
        BudgetFit。``within_budget=False`` 表示连最低配置都超预算。
    """
    target = budget_limit(request.budget)
    floor = budget_floor(request.budget)
    travelers = request.travelers
    level = request.budget_level

    def build_fit(
        price: int,
        tickets_per_person: int,
        level_name: str,
        count: int,
        result: BudgetBreakdown,
        within_floor: bool,
        max_spendable: int = 0,
    ) -> BudgetFit:
        """按给定配置组装收敛结果（含酒店名与说明文本）。"""
        fitted_name, regenerate = _fit_hotel(candidates, hotel_name, price, nightly_price)
        effective = request.model_copy(update={"budget_level": level_name})
        return BudgetFit(
            request=request,
            breakdown=result,
            attraction_total=tickets_per_person,
            nightly_price=price,
            priced_attraction_count=count,
            effective_request=effective,
            within_budget=True,
            within_floor=within_floor,
            max_spendable=max_spendable,
            adjusted=True,
            level_downgraded=(level_name != level),
            dropped_attractions=max(original_count - count, 0),
            hotel_name=fitted_name,
            regenerate_needed=regenerate,
            notes=_adjust_notes(
                request,
                baseline.total,
                result.total,
                price,
                fitted_name,
                original_count - count,
                level_name != level,
                within_floor,
                floor,
                max_spendable,
            ),
        )

    # 初始配置：在上限内就原样保留；已经满足区间时不做任何调整
    baseline_tickets = sum(a.ticket_price for a in attractions[: request.days * 2])
    baseline = estimate_budget(
        request, nightly_price=nightly_price, attraction_total=baseline_tickets
    )
    baseline_count = min(len(attractions), request.days * 2)
    if baseline.total <= target + tol:
        if baseline.total >= floor - tol:
            return BudgetFit(
                request=request,
                breakdown=baseline,
                attraction_total=baseline_tickets,
                nightly_price=nightly_price,
                priced_attraction_count=baseline_count,
                effective_request=request,
                hotel_name=hotel_name,
                within_budget=True,
                within_floor=True,
                adjusted=False,
            )
        # 没到消费下限：配置本身可以（不超上限），只是花得不够
        return BudgetFit(
            request=request,
            breakdown=baseline,
            attraction_total=baseline_tickets,
            nightly_price=nightly_price,
            priced_attraction_count=baseline_count,
            effective_request=request,
            hotel_name=hotel_name,
            within_budget=True,
            within_floor=False,
            max_spendable=baseline.total,
            adjusted=False,
            notes=_floor_notes(request, baseline.total, floor, baseline.total),
        )

    def counts_for(level_name: str) -> List[int]:
        """返回该档位下所有"降几档"的取值，从高到低覆盖到最低档。

        为什么不是固定的 ``[0, 1]``：豪华档若只降一档会停在中等档，
        中等档本身也可能超预算，于是明明"降到经济档就能满足"的组合
        会被误判成"怎么压都超预算"。这里直接把当前档到经济档的每一级
        都枚举出来，保证搜索空间完整。
        """
        start = LEVEL_ORDER.index(level_name)
        return list(range(0, len(LEVEL_ORDER) - start))

    def attempt(price: int, tickets_per_person: int, level_name: str) -> BudgetBreakdown:
        """用给定房价/门票试算一次总花费。"""
        trial = request.model_copy(update={"budget_level": level_name})
        return estimate_budget(
            trial,
            nightly_price=price,
            attraction_total=tickets_per_person,
            period_ratio=1.0,
        )

    # 景点选择方案：优先"按推荐顺序取前 N 个"，N 不足时退化为"取最便宜的 N 个"。
    original_count = min(len(attractions), request.days * 2)
    n_min = 1 if original_count else 0
    cheapest_first = sorted(attractions, key=lambda a: a.ticket_price)

    def select(count: int) -> List[Attraction]:
        if count <= original_count:
            return attractions[:count]
        return cheapest_first[:count]

    # 房价下限：不能低于当地真实最低房价，否则会算出"住不到的价格"。
    # 没有候选酒店数据时才退化为 0（即允许用档位默认价）。
    price_floor = 0
    if candidates:
        try:
            price_floor = max(int(min(h.price_per_night for h in candidates)), 0)
        except (AttributeError, ValueError):  # pragma: no cover - 防御性分支
            price_floor = 0

    # ---- 第一轮：找"既达消费下限、又不超上限"的最优配置 ----
    # 优先景点多、其次住宿好；为此把候选酒店的**真实房价**也放进备选，
    # 否则"最便宜的那家就已经超过可承受区间上限"时会被误判成"花不出去"。
    price_candidates = {p for p in DEFAULT_NIGHTLY_PRICE.values() if price_floor <= p <= target}
    if candidates:
        try:
            price_candidates.update(
                int(h.price_per_night)
                for h in candidates
                if price_floor <= int(h.price_per_night) <= target
            )
        except (AttributeError, ValueError):  # pragma: no cover - 防御性分支
            pass
    ladder = sorted(price_candidates, reverse=True)

    best_below_floor: Optional[tuple] = None

    for n in range(original_count, n_min - 1, -1):
        picked = select(n)
        tickets_per_person = sum(a.ticket_price for a in picked)

        for steps in counts_for(level):
            idx = min(LEVEL_ORDER.index(level) + steps, len(LEVEL_ORDER) - 1)
            trial_level = LEVEL_ORDER[idx]

            for price in ladder:
                result = attempt(price, tickets_per_person, trial_level)
                if result.total > target + tol:
                    continue
                if result.total >= floor - tol:
                    # 从高到低扫，第一个命中的就是该配置下最贵的可行项
                    return build_fit(
                        price, tickets_per_person, trial_level, n, result,
                        within_floor=True,
                    )
                if best_below_floor is None or result.total > best_below_floor[0]:
                    best_below_floor = (result.total, price, tickets_per_person, trial_level, n, result)

    # ---- 第二轮：区间无解时保上限（用户已确认优先级），如实汇报"最多只能花到多少" ----
    for n in range(original_count, n_min - 1, -1):
        picked = select(n)
        tickets_per_person = sum(a.ticket_price for a in picked)

        # 逐档下调：0=当前档，逐级降到经济档为止
        for steps in counts_for(level):
            idx = min(LEVEL_ORDER.index(level) + steps, len(LEVEL_ORDER) - 1)
            trial_level = LEVEL_ORDER[idx]
            base_price = DEFAULT_NIGHTLY_PRICE.get(trial_level, 700)

            # 起点不能低于房价下限（经济档默认价 280 可能低于当地最低房价 289）
            start_price = max(base_price, price_floor)
            for price in range(start_price, price_floor - 1, -PRICE_STEP):
                result = attempt(price, tickets_per_person, trial_level)
                if result.total <= target + tol:
                    best = (
                        best_below_floor[0]
                        if best_below_floor is not None
                        else result.total
                    )
                    return build_fit(
                        price, tickets_per_person, trial_level, n, result,
                        within_floor=False, max_spendable=best,
                    )

    # ---- 第三轮：连"最便宜配置"都超上限 → 如实汇报理论最低花费 ----
    cheapest_level = "经济"
    cheapest = attempt(price_floor, 0, cheapest_level)
    shortfall = cheapest.total - target

    return BudgetFit(
        request=request,
        breakdown=cheapest,
        attraction_total=0,
        nightly_price=price_floor,
        priced_attraction_count=0,
        effective_request=request.model_copy(update={"budget_level": cheapest_level}),
        period_ratio=1.0,
        within_budget=False,
        within_floor=False,
        adjusted=True,
        level_downgraded=(cheapest_level != level),
        dropped_attractions=original_count,
        hotel_name=_cheapest_hotel_name(candidates, hotel_name),
        regenerate_needed=False,
        min_feasible_total=cheapest.total,
        notes=_infeasible_notes(request, cheapest.total, target, shortfall),
    )


def _cheapest_hotel_name(candidates: Optional[List[Any]], original_name: str) -> str:
    """取候选里最便宜的酒店名（收敛到最低档时用来替换原酒店）。"""
    if candidates:
        try:
            return min(candidates, key=lambda h: h.price_per_night).name
        except (AttributeError, ValueError):  # pragma: no cover - 防御性分支
            pass
    return original_name


def _fit_hotel(
    candidates: Optional[List[Any]],
    original_name: str,
    price: int,
    original_price: int,
) -> tuple:
    """按收敛房价挑一家真实存在的酒店。

    Returns:
        ``(酒店名, 是否需要重写逐日行程)``。房价没变或没有候选数据时沿用原名。
    """
    if not candidates or price >= original_price:
        return original_name, price != original_price
    affordable = [h for h in candidates if h.price_per_night <= price]
    if affordable:
        return max(affordable, key=lambda h: h.price_per_night).name, True
    cheapest = _cheapest_hotel_name(candidates, original_name)
    return cheapest, cheapest != original_name


def _adjust_notes(
    request: TripRequest,
    before_total: int,
    after_total: int,
    nightly_price: int,
    hotel_name: str,
    dropped: int,
    level_downgraded: bool,
    within_floor: bool = True,
    floor: int = 0,
    max_spendable: int = 0,
) -> str:
    """生成"已为预算收敛"的说明文本。"""
    actions: List[str] = []
    if level_downgraded:
        actions.append("下调住宿档位")
    if nightly_price > 0:
        actions.append(f"住宿控制在 {nightly_price} 元/晚以内")
    if hotel_name:
        actions.append(f"改荐 {hotel_name}")
    if dropped > 0:
        actions.append(f"放弃 {dropped} 个付费景点")

    action_text = "、".join(actions) if actions else "压缩弹性支出"
    notes = (
        f"（原方案预估 {before_total} 元，超出预算上限；"
        f"已自动{action_text}，收敛后 {after_total} 元）"
    )
    if not within_floor and floor > 0:
        notes += _floor_notes(request, after_total, floor, max_spendable)
    return notes


def _floor_notes(
    request: TripRequest, total: int, floor: int, max_spendable: int
) -> str:
    """生成"未达消费下限"的说明文本（保上限，如实说明）。"""
    return (
        f"（未达消费下限：要求至少花掉 {floor} 元，而 {request.destination} "
        f"{request.days} 天 {request.travelers} 人的当前配置最多只能花到 {max_spendable} 元。"
        f"上限优先，未做超额消费；如需把预算用足，可升级住宿档位、增加付费景点或延长天数）"
    )


def _infeasible_notes(
    request: TripRequest, min_total: int, target: int, shortfall: int
) -> str:
    """生成"压不进预算"时的说明文本（明确告知而不是静默超标）。"""
    return (
        f"【无法压入预算】{request.destination} {request.days} 天 {request.travelers} 人的理论最低"
        f"花费约 {min_total} 元（经济档住宿 + 免费景点 + 往返大交通与餐饮的固定支出），"
        f"仍高于预算上限 {target} 元，缺口 {shortfall} 元。"
        f"往返大交通与餐饮属刚性支出，无法通过压缩住宿与门票消除。"
        f"如预算固定，建议减少天数、改选更近的目的地或调整出行日期。"
    )


def budget_status_text(total: int, budget: int) -> str:
    """生成预算对比结论文本。

    判定口径是**区间**：``[预算 × 0.8, 预算 × 1.1]``。
    ``budget`` 是用户预算，文案里把下限与上限都写出来，避免"到底算不算达标"产生歧义。
    """
    if budget <= 0:
        return f"预估总花费约 {total} 元。"
    cap = budget_limit(budget)
    floor = budget_floor(budget)

    if total > cap:
        diff = total - cap
        if diff <= budget * 0.1:
            return f"预估总花费 {total} 元，略超预算上限 {cap} 元（{diff} 元），建议压缩购物或把一餐改为小吃。"
        if diff <= budget * 0.3:
            return f"预估总花费 {total} 元，超出预算上限 {cap} 元（{diff} 元），建议下调住宿档位或减少付费景点。"
        return f"预估总花费 {total} 元，明显超出预算上限 {cap} 元（{diff} 元），建议缩短天数或选择经济档住宿。"

    if total < floor:
        return (
            f"【未达消费下限】预估总花费 {total} 元，低于要求的 {floor} 元"
            f"（预算 {budget} 元的下限），当前行程配置无法把预算用足，但仍未超过上限 {cap} 元。"
        )

    return (
        f"【已按预算区间收敛】预估总花费 {total} 元，落在预算区间 "
        f"{floor} ~ {cap} 元内（预算 {budget} 元），距离上限还有 {cap - total} 元余量。"
    )



def build_tips(
    request: TripRequest,
    attractions: List[Attraction],
    weather_notes: Optional[List[str]] = None,
) -> List[str]:
    """生成注意事项列表。"""
    tips: List[str] = []
    tips.append(f"{request.destination}热门景点普遍需要提前 1-7 天在官方渠道实名预约，请尽早订票。")
    if request.preferences:
        tips.append(f"本次行程重点照顾了你的偏好：{'、'.join(request.preferences)}，可按体力取舍付费景点。")
    if any(a.ticket_price >= 100 for a in attractions):
        tips.append("行程中含高价门票景点，学生/老人凭证件通常可享半价，记得携带有效证件。")
    if request.days >= 5:
        tips.append("行程超过 5 天，建议中间安排半天机动时间，避免连续高强度游览导致疲劳。")
    if request.budget_level == "经济":
        tips.append("经济档建议优先选择地铁沿线酒店，通勤时间与打车费用都能明显下降。")
    if request.budget_level == "豪华":
        tips.append("豪华档建议提前预订带行政酒廊或接送服务的酒店，可显著提升出行效率。")
    tips.append("出发前请确认身份证、充电宝（≤100Wh）、常用药品与当地天气对应的衣物。")
    if weather_notes:
        rainy = [note for note in weather_notes if "雨" in note]
        if rainy:
            tips.append(f"行程中有 {len(rainy)} 天可能出现降水，务必携带折叠伞与防水鞋套。")
    tips.append("以上价格为模拟估算数据，仅用于行程规划演示，实际价格请以各平台实时报价为准。")
    return tips


def build_summary(request: TripRequest, daily_plans: List[DailyPlan]) -> str:
    """生成整体行程概述。"""
    pref = "、".join(request.preferences) if request.preferences else "综合体验"
    first_theme = daily_plans[0].theme if daily_plans else ""
    return (
        f"这是一份为 {request.travelers} 人定制的 {request.destination} {request.days} 天行程，"
        f"偏好侧重「{pref}」，预算档位为{request.budget_level}（总预算 {request.budget} 元）。"
        f"行程以 {first_theme} 开场，串联 {request.destination} 的代表性景点，"
        f"每天按上午/下午/晚间三段安排，并已结合天气与住宿位置优化路线，"
        f"可直接作为出行参考。"
    )
