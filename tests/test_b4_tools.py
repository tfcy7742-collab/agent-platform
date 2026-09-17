"""B4 测试：工具协议、参数校验、注册表与两个内置工具。

覆盖点：
1. ``validate_args`` 的必填/类型/枚举/区间校验与默认值补全（含"模型给字符串数字"的宽松转换）；
2. ``BaseTool.run`` 的模板方法：参数错误、超时熔断、内部异常都被包装成 ``ToolResult``；
3. 注册表：注册、查找（含中文别名）、启用禁用、统一调用入口、目录导出；
4. ``knowledge_search`` 工具：拒答是"正常结果"而不是错误；
5. ``trip_planner`` 工具：自然语言参数解析（"北京三日游，喜欢历史文化，预算 8000"）。
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest

from core.tools.base import BaseTool, ToolErrorType, ToolResult, validate_args


# ---------------------------------------------------------------------------
# 测试用工具
# ---------------------------------------------------------------------------
class EchoTool(BaseTool):
    """回显工具（用于协议测试）。"""

    name = "echo"
    description = "回显输入文本"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "count": {"type": "integer", "default": 1, "minimum": 1, "maximum": 5},
            "mode": {"type": "string", "enum": ["plain", "upper"], "default": "plain"},
        },
        "required": ["text"],
    }
    timeout_s = 5.0

    def _run(self, text: str, count: int = 1, mode: str = "plain") -> ToolResult:
        value = text.upper() if mode == "upper" else text
        return ToolResult(ok=True, data=value * count, display=value * count)


class SlowTool(BaseTool):
    """故意很慢的工具（用于超时测试）。"""

    name = "slow"
    description = "故意超时的工具"
    parameters = {"type": "object", "properties": {"seconds": {"type": "number", "default": 1.0}}}
    timeout_s = 0.3
    retryable = False

    def _run(self, seconds: float = 1.0) -> ToolResult:
        time.sleep(seconds)
        return ToolResult(ok=True, data="done")


class BrokenTool(BaseTool):
    """总是抛异常的工具。"""

    name = "broken"
    description = "总是失败的工具"
    parameters = {"type": "object", "properties": {}}
    timeout_s = 5.0

    def _run(self, **_: Any) -> ToolResult:
        raise RuntimeError("内部炸了")


class FlakyTool(BaseTool):
    """前 N 次失败（模拟瞬时故障）。"""

    name = "flaky"
    description = "前几次失败的工具"
    parameters = {"type": "object", "properties": {"fail_times": {"type": "integer", "default": 1}}}
    timeout_s = 5.0
    retryable = True

    def __init__(self, error_type: str = ToolErrorType.TIMEOUT) -> None:
        super().__init__()
        self.remaining_failures = 0
        self.error_type = error_type
        self.calls = 0

    def _run(self, fail_times: int = 1) -> ToolResult:
        self.calls += 1
        if self.calls <= fail_times:
            return ToolResult(ok=False, error="瞬时失败", error_type=self.error_type)
        return ToolResult(ok=True, data="终于在第三次成功")


class DangerousTool(BaseTool):
    """需要人工确认的工具。"""

    name = "dangerous"
    description = "有副作用的危险工具"
    parameters = {"type": "object", "properties": {"target": {"type": "string"}}}
    requires_confirmation = True
    timeout_s = 5.0

    def _run(self, target: str = "") -> ToolResult:
        return ToolResult(ok=True, data=f"已对 {target} 执行了写操作")


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------
def test_validate_args_fills_defaults() -> None:
    """未提供的可选参数要补上默认值。"""
    schema = EchoTool.parameters
    result = validate_args(schema, {"text": "hi"})
    assert result == {"text": "hi", "count": 1, "mode": "plain"}


def test_validate_args_requires_mandatory() -> None:
    """缺必填参数要报错，且错误信息里点名缺哪个。"""
    with pytest.raises(ValueError, match="text"):
        validate_args(EchoTool.parameters, {})


def test_validate_args_rejects_empty_string_for_required() -> None:
    """必填参数传空字符串等同于没传。"""
    with pytest.raises(ValueError, match="text"):
        validate_args(EchoTool.parameters, {"text": "   "})


def test_validate_args_type_check() -> None:
    """类型不符要报错。"""
    with pytest.raises(ValueError, match="count"):
        validate_args(EchoTool.parameters, {"text": "a", "count": [1]})


def test_validate_args_coerces_numeric_string() -> None:
    """模型经常把数字写成字符串，应当宽松转换而不是直接报错。"""
    result = validate_args(EchoTool.parameters, {"text": "a", "count": "3"})
    assert result["count"] == 3
    assert isinstance(result["count"], int)


def test_validate_args_rejects_bool_for_integer() -> None:
    """布尔值是 int 的子类，但当成数字传进来属于类型错误。"""
    with pytest.raises(ValueError, match="count"):
        validate_args(EchoTool.parameters, {"text": "a", "count": True})


def test_validate_args_enum_and_range() -> None:
    """枚举与区间校验。"""
    with pytest.raises(ValueError, match="mode"):
        validate_args(EchoTool.parameters, {"text": "a", "mode": "weird"})
    with pytest.raises(ValueError, match="count"):
        validate_args(EchoTool.parameters, {"text": "a", "count": 99})


# ---------------------------------------------------------------------------
# BaseTool.run 模板方法
# ---------------------------------------------------------------------------
def test_tool_run_success() -> None:
    """正常执行：返回 ToolResult，且补齐耗时与统计。"""
    tool = EchoTool()
    result = tool.run(text="hi", mode="upper", count=2)
    assert result.ok is True
    assert result.data == "HIHI"
    assert result.latency_ms >= 0
    assert tool.call_count == 1
    assert tool.error_count == 0


def test_tool_run_invalid_args_is_wrapped() -> None:
    """参数错误要被包装成 ToolResult（不抛给上层），错误类型可回灌给模型。"""
    tool = EchoTool()
    result = tool.run()
    assert result.ok is False
    assert result.error_type == ToolErrorType.INVALID_ARGS
    assert "text" in (result.error or "")


def test_tool_run_timeout_is_circuit_broken() -> None:
    """超时要被熔断，并标记 degraded（供 trace 与前端展示）。"""
    tool = SlowTool()
    result = tool.run(seconds=2.0)
    assert result.ok is False
    assert result.error_type == ToolErrorType.TIMEOUT
    assert result.degraded is True
    assert "超时" in (result.error or "")


def test_tool_run_internal_error_is_wrapped() -> None:
    """工具内部异常要被包装成 tool_error，而不是 500。"""
    tool = BrokenTool()
    result = tool.run()
    assert result.ok is False
    assert result.error_type == ToolErrorType.INTERNAL
    assert "内部炸了" in (result.error or "")


def test_tool_disabled() -> None:
    """禁用后调用直接返回结构化错误。"""
    tool = EchoTool()
    tool.enabled = False
    result = tool.run(text="hi")
    assert result.ok is False
    assert result.error_type == ToolErrorType.DISABLED


def test_tool_to_openai_format() -> None:
    """导出的 function calling 描述要符合 OpenAI/DeepSeek 格式。"""
    spec = EchoTool().to_openai_tool()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "echo"
    assert spec["function"]["parameters"] is EchoTool.parameters


def test_tool_result_to_llm_text() -> None:
    """给模型的文本要紧凑，失败时给出分类与原因。"""
    success = ToolResult(ok=True, data={"a": 1}, display="简短说明")
    assert success.to_llm_text() == "简短说明"

    failure = ToolResult(ok=False, error="炸了", error_type="tool_error")
    assert "tool_error" in failure.to_llm_text()
    assert "炸了" in failure.to_llm_text()


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
@pytest.fixture()
def registry():
    """干净的注册表（不加载真实内置工具）。"""
    from core.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


def test_registry_register_and_get(registry) -> None:
    """注册与查找。"""
    assert registry.get("echo") is not None
    assert registry.get("echo").name == "echo"
    assert registry.get("不存在") is None
    assert registry.names() == ["echo"]


def test_registry_alias_lookup(registry) -> None:
    """中文别名要能解析（Planner 在中文语境下常用中文叫法）。"""
    from core.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(EchoTool())
    assert reg.get("知识库检索") is None      # 别名指向的工具没注册

    from core.tools.registry import TOOL_ALIASES

    assert TOOL_ALIASES["知识库检索"] == "knowledge_search"
    assert TOOL_ALIASES["旅行规划"] == "trip_planner"


def test_registry_execute_success(registry) -> None:
    """统一调用入口。"""
    result = registry.execute("echo", {"text": "hi"})
    assert result.ok is True
    assert result.data == "hi"


def test_registry_execute_unknown_tool(registry) -> None:
    """调用不存在的工具返回结构化错误，并列出可用工具。"""
    result = registry.execute("nope", {})
    assert result.ok is False
    assert result.error_type == ToolErrorType.NOT_FOUND
    assert "echo" in (result.error or "")


def test_registry_enable_disable(registry) -> None:
    """启用/禁用会影响 catalog 与执行。"""
    assert registry.set_enabled("echo", False) is True
    assert registry.catalog(enabled_only=True) == []
    assert len(registry.catalog(enabled_only=False)) == 1
    assert registry.execute("echo", {"text": "x"}).error_type == ToolErrorType.DISABLED
    assert registry.set_enabled("不存在", True) is False


def test_registry_catalog_has_required_fields() -> None:
    """工具目录必须包含 Planner 决策所需的全部字段。"""
    from core.tools.registry import bootstrap_tools

    registry = bootstrap_tools(force=True)
    catalog = registry.catalog()
    assert len(catalog) >= 2
    names = {item["name"] for item in catalog}
    assert {"knowledge_search", "trip_planner"} <= names
    for item in catalog:
        assert item["description"], f"{item['name']} 缺少能力说明（会影响路由准确率）"
        assert item["parameters"]["type"] == "object"
        assert item["est_cost"] in {"low", "medium", "high"}
        assert item["est_latency"] in {"low", "medium", "high"}
    registry.set_enabled("knowledge_search", True)


def test_registry_prompt_catalog_mentions_limits() -> None:
    """给 Planner 的工具说明里要带成本/耗时，便于做权衡。"""
    from core.tools.registry import bootstrap_tools

    registry = bootstrap_tools(force=True)
    text = registry.describe_for_prompt()
    assert "knowledge_search" in text
    assert "trip_planner" in text
    assert "成本" in text
    assert "耗时" in text


# ---------------------------------------------------------------------------
# knowledge_search 工具
# ---------------------------------------------------------------------------
def test_knowledge_tool_refusal_is_normal_result(rag_engine) -> None:
    """知识库为空时：``ok=True`` 但 data.refused=True（拒答不是错误）。"""
    from core.tools.knowledge_tool import KnowledgeSearchTool

    tool = KnowledgeSearchTool(engine=rag_engine)
    result = tool.run(query="年假有几天")
    assert result.ok is True                      # 工具本身成功执行了
    assert result.data["refused"] is True         # 但明确表示资料中没有
    assert result.data["refuse_reason"] == "no_documents"
    assert "无法回答" in result.data["answer"]
    assert result.meta["refused"] is True


def test_knowledge_tool_returns_sources(rag_engine, vector_store) -> None:
    """有资料时返回结构化来源（文件名/页码/chunk_id/分数）。"""
    from rag.models import Chunk

    vector_store.add_documents(
        [
            Chunk(
                chunk_id="doc_hr:1:0",
                text="员工入职满一年后享有年假，满一年不满十年者每年五天。",
                file_name="员工手册.pdf",
                doc_id="doc_hr",
                page=1,
                chunk_index=0,
            )
        ]
    )

    from core.tools.knowledge_tool import KnowledgeSearchTool

    tool = KnowledgeSearchTool(engine=rag_engine)
    result = tool.run(query="年假有几天", threshold=0.0)
    assert result.ok is True
    assert result.data["refused"] is False
    assert result.data["sources"]
    source = result.data["sources"][0]
    assert source["file_name"] == "员工手册.pdf"
    assert source["page"] == 1
    assert source["chunk_id"]
    assert source["score"] > 0
    # 给模型看的文本要包含答案，便于 Planner 归纳
    assert "年假" in result.to_llm_text()


def test_knowledge_tool_requires_query() -> None:
    """缺少 query 参数 → 参数错误。"""
    from core.tools.knowledge_tool import KnowledgeSearchTool

    result = KnowledgeSearchTool().run()
    assert result.ok is False
    assert result.error_type == ToolErrorType.INVALID_ARGS


# ---------------------------------------------------------------------------
# trip_planner 工具的文本解析
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("帮我规划北京三日游", 3),
        ("安排 5 天行程", 5),
        ("两天一夜", 2),
        ("玩七天", 7),
        ("随便看看", None),
        ("计划 99 天", None),      # 超出上限视为未识别
    ],
)
def test_parse_days(text: str, expected) -> None:
    """天数解析：中英文数字、带/不带空格都要认。"""
    from core.tools.trip_tool import parse_days

    assert parse_days(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("预算 8000 元", 8000),
        ("大概 12000 块", 12000),
        ("3 天", None),            # 天数不应被当成预算
        ("随便", None),
    ],
)
def test_parse_budget(text: str, expected) -> None:
    """预算解析：小额数字（天数）不能被误判为预算。"""
    from core.tools.trip_tool import parse_budget

    assert parse_budget(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("喜欢历史文化", ["历史文化"]),
        ("想吃喝玩乐顺便拍照", ["美食", "摄影"]),
        ("带着孩子去，喜欢自然风光", ["自然风光", "亲子"]),
        ("随便逛逛", []),
    ],
)
def test_parse_preferences(text: str, expected) -> None:
    """偏好解析：按关键词命中，最多 3 个且顺序稳定。"""
    from core.tools.trip_tool import parse_preferences

    assert parse_preferences(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("穷游", "经济"), ("五星酒店", "豪华"), ("舒适一点", "中等"), ("随便", None), ("", None)],
)
def test_parse_budget_level(text: str, expected) -> None:
    """预算档位解析：未提及档位时返回 None，由调用方决定默认值。

    契约变化说明：早期版本在这里直接返回默认值"中等"，导致工具分不清
    "用户说了中等"与"用户什么都没说"，也无法实现"结构化参数 > 文本解析 > 默认值"
    的优先级。现在解析函数只负责识别，默认值统一在调用方决定。
    """
    from core.tools.trip_tool import parse_budget_level

    assert parse_budget_level(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("两个人出游", 2),
        ("3人出行", 3),
        ("我们一共 5 个人", 5),
        ("十个人", 10),
        ("北京三日游", None),
        ("", None),
        ("20人以上", 20),
        ("50人", None),        # 超出上限视为未识别
    ],
)
def test_parse_travelers(text: str, expected) -> None:
    """人数解析（新增能力：原来根本没有这个函数，人数永远走默认值 2）。"""
    from core.tools.trip_tool import parse_travelers

    assert parse_travelers(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026-10-01 出发", None),              # 日期不能被当成预算（实际踩过的 bug）
        ("2026-10-01 出发，预算 8000", 8000),
        ("10月1日去北京，大概 12000 块", 12000),
        ("2026年10月1日，预算 15000 元", 15000),
        ("穷游", None),
    ],
)
def test_parse_budget_ignores_dates(text: str, expected) -> None:
    """预算解析必须先剔除日期片段。

    实测 bug：``"2026-10-01 出发"`` 曾被解析成"预算 2026 元"，
    因为日期里的年份符合"3~6 位数字"的预算规则。
    """
    from core.tools.trip_tool import parse_budget

    assert parse_budget(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026-10-01 出发", "2026-10-01"),
        ("2026年12月25日走", "2026-12-25"),
        ("10月1日去北京", None),                # 缺年份时不做猜测
        ("随便", None),
    ],
)
def test_parse_start_date(text: str, expected) -> None:
    """出发日期解析（新增能力：原来不支持指定日期，永远默认"一周后"）。"""
    from datetime import date

    from core.tools.trip_tool import parse_start_date

    result = parse_start_date(text)
    assert (result.isoformat() if isinstance(result, date) else None) == expected


def test_structured_params_override_text() -> None:
    """结构化参数优先于文本解析。

    这是"表单通道"能可靠工作的前提：用户在表单里选了什么就用什么，
    不能被问题文本里的其他数字或关键词带偏。
    """
    from core.tools.trip_tool import TripPlannerTool

    tool = TripPlannerTool(enable_llm=False)
    result = tool.run(
        question="帮我规划北京三日游，两个人，预算 3000",
        destination="杭州",       # 与文本里的"北京"冲突，应当以结构化参数为准
        days=5,
        travelers=4,
        budget=20000,
        budget_level="豪华",
        preferences=["美食"],
        start_date="2026-11-11",
    )

    assert result.ok is True
    parsed = result.meta["parsed"]
    assert parsed["destination"] == "杭州"
    assert parsed["days"] == 5
    assert parsed["travelers"] == 4
    assert parsed["budget"] == 20000
    assert parsed["budget_level"] == "豪华"
    assert parsed["preferences"] == ["美食"]
    assert parsed["start_date"] == "2026-11-11"
    assert parsed["budget_source"] == "用户指定"


def test_plan_result_is_not_truncated() -> None:
    """工具返回的行程必须完整（不截断）。

    这是用户反馈"比原项目单薄"的直接原因：早期版本把 display 截成
    "前 3 天 / 每天 40 字"，大模型只能复述一个极简版。
    """
    from core.tools.trip_tool import TripPlannerTool

    tool = TripPlannerTool(enable_llm=False)
    result = tool.run(destination="北京", days=5, travelers=2, budget=12000, budget_level="中等")

    assert result.ok is True
    data = result.data
    assert len(data["daily_plans"]) == 5, "5 天行程必须有 5 天的完整数据"

    # display 里要能看到全部 5 天，且每天的时段描述完整（不出现"…"截断）
    for day in range(1, 6):
        assert f"【第 {day} 天】" in result.display, f"display 缺少第 {day} 天"
    assert "…" not in result.display.split("预算明细")[0], "display 不应包含省略号截断"

    # 结构化字段要齐全（供界面渲染）
    assert data["daily_plans"][0]["meals"], "每日餐饮应结构化返回"
    assert data["attractions"] and isinstance(data["attractions"][0], dict)
    assert data["hotels"] and isinstance(data["hotels"][0], dict)
    assert data["weather"] and isinstance(data["weather"][0], dict)
    assert data["budget_breakdown"].get("合计", 0) > 0


def test_guess_destination() -> None:
    """目的地识别：优先匹配内置城市，避免猜出没有数据的城市。"""
    from core.tools.trip_tool import TripPlannerTool

    tool = TripPlannerTool(enable_llm=False)
    assert tool._guess_destination("帮我规划北京三日游") == "北京"
    assert tool._guess_destination("想去成都吃火锅") == "成都"
    # 内置城市优先于自定义解析
    assert tool._guess_destination("从上海出发去杭州玩三天") == "上海"
    assert tool._guess_destination("随便聊聊") is None


def test_guess_destination_requires_city_dictionary() -> None:
    """未收录的城市不会被强行当成地名：宁可报错也不编城市名。

    这是实际踩过的坑——早期版本用正则猜前缀，把"帮我规划一次旅行"里的
    "一次"当成了城市，生成了一份"一次 3 天行程"。
    """
    from core.tools.trip_tool import TripPlannerTool

    tool = TripPlannerTool(enable_llm=False)
    assert tool._guess_destination("帮我规划一次旅行") is None
    assert tool._guess_destination("给我安排一次周边游") is None
    assert tool._guess_destination("随便聊聊") is None
    # 词典里的城市（即使子系统没有专属数据）要能识别出来
    assert tool._guess_destination("想去婺源看油菜花") == "婺源"
    assert tool._guess_destination("想去拉萨玩五天") == "拉萨"


def test_trip_tool_missing_destination() -> None:
    """识别不出目的地时给出明确提示（而不是猜一个城市）。"""
    from core.tools.trip_tool import TripPlannerTool

    result = TripPlannerTool(enable_llm=False).run(question="帮我规划一次旅行")
    assert result.ok is False
    assert "目的地" in (result.error or "")


def test_trip_tool_end_to_end_offline() -> None:
    """旅行规划工具端到端（本地规则引擎，不调用大模型）。"""
    from core.tools.trip_tool import TripPlannerTool

    tool = TripPlannerTool(enable_llm=False)
    result = tool.run(question="帮我规划北京三日游，喜欢历史文化，预算 8000 元")

    assert result.ok is True
    data = result.data
    assert data["destination"] == "北京"
    assert data["days"] == 3
    assert data["budget"] if "budget" in data else True
    assert data["daily_plans"] and len(data["daily_plans"]) == 3
    assert data["attractions"]
    assert data["hotels"]
    assert data["estimated_total"] > 0
    # 每日三段必须完整（这是 trip-planner 子系统的既有保证）
    for plan in data["daily_plans"]:
        assert plan["morning"] and plan["afternoon"] and plan["evening"]
    # 子智能体轨迹要透传出来（体现"工具内部也是多智能体"）
    assert result.meta["sub_agents"] == 4
    assert "北京" in result.display
    # 解析出来的参数要如实回传，便于排查"是不是解析错了"
    assert result.meta["parsed"]["destination"] == "北京"
    assert result.meta["parsed"]["days"] == 3
