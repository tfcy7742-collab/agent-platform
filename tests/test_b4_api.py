"""B4 测试：Agent 模式的问答接口、工具目录与执行轨迹接口。

覆盖点：
1. ``POST /api/chat`` 的 ``mode=agent``：Planner 自主路由到不同工具；
2. ``mode=rag`` 与 ``mode=agent`` 的行为差异（回归基线 vs 自主编排）；
3. ``GET /api/tools``：目录字段完整（Planner 决策依赖这些字段）；
4. ``POST /api/tools/{name}/invoke``：绕过 Planner 直接调用工具；
5. ``POST /api/tools/{name}/toggle``：运行期启停；
6. ``GET /api/traces`` 与 ``/api/traces/{run_id}``：能回看决策与工具调用。
"""

from __future__ import annotations

import io

from core.prompts import REFUSAL_MESSAGE

DOC_TEXT = "\n\n".join(
    [
        "第一条 年假规定：员工入职满一年后享有年假，工作满一年不满十年者每年五天。",
        "第二条 报销标准：一线城市住宿标准为每晚六百元，其他城市为每晚四百元。",
    ]
)


def upload(client, name: str = "员工手册.txt", content: str = DOC_TEXT):
    """上传一份文档。"""
    return client.post(
        "/api/documents",
        files=[("files", (name, io.BytesIO(content.encode("utf-8")), "text/plain"))],
        data={"force": "false"},
    )


# ---------------------------------------------------------------------------
# 工具目录
# ---------------------------------------------------------------------------
def test_tools_endpoint_lists_builtin_tools(client) -> None:
    """工具目录必须包含两个内置工具，且字段完整。"""
    body = client.get("/api/tools").json()
    assert body["ok"] is True
    names = {item["name"] for item in body["tools"]}
    assert {"knowledge_search", "trip_planner"} <= names

    for item in body["tools"]:
        assert item["description"], f"{item['name']} 缺少能力说明"
        assert item["parameters"]["type"] == "object"
        assert item["est_cost"] in {"low", "medium", "high"}
        assert item["est_latency"] in {"low", "medium", "high"}
        assert "retryable" in item
        assert "requires_confirmation" in item


def test_tools_endpoint_reports_costs(client) -> None:
    """成本/耗时档位要能体现差异（Planner 据此做权衡）。"""
    tools = {item["name"]: item for item in client.get("/api/tools").json()["tools"]}
    assert tools["knowledge_search"]["est_cost"] == "medium"
    assert tools["trip_planner"]["est_cost"] == "high"
    assert tools["trip_planner"]["retryable"] is False, "旅行规划重试等于重跑 4 个智能体"


def test_tools_stats_endpoint(client) -> None:
    """统计接口可用。"""
    body = client.get("/api/tools/stats").json()
    assert body["ok"] is True
    assert isinstance(body["stats"], list)
    assert "pending_confirmations" in body


def test_toggle_tool(client) -> None:
    """运行期禁用工具后：enabled_only 视图里消失、调用返回禁用错误。"""
    response = client.post("/api/tools/knowledge_search/toggle", params={"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False

    # 注意：/api/tools 默认返回**全部**工具（含已禁用但标记 enabled=false 的），
    # 这是刻意的——前端需要展示"有哪些工具、当前哪些被关掉了"。
    # 想要"只看可用工具"要传 enabled_only=true（Planner 用的就是这个视图）。
    all_tools = {item["name"]: item for item in client.get("/api/tools").json()["tools"]}
    assert all_tools["knowledge_search"]["enabled"] is False

    enabled_tools = {
        item["name"]
        for item in client.get("/api/tools", params={"enabled_only": True}).json()["tools"]
    }
    assert "knowledge_search" not in enabled_tools
    assert "trip_planner" in enabled_tools

    invoked = client.post("/api/tools/knowledge_search/invoke", json={"args": {"query": "年假"}}).json()
    assert invoked["ok"] is False
    assert invoked["result"]["error_type"] == "tool_disabled"

    # 恢复（避免影响其他用例）
    client.post("/api/tools/knowledge_search/toggle", params={"enabled": True})
    enabled_again = {
        item["name"]
        for item in client.get("/api/tools", params={"enabled_only": True}).json()["tools"]
    }
    assert "knowledge_search" in enabled_again


def test_toggle_unknown_tool(client) -> None:
    """禁用不存在的工具 → 404。"""
    response = client.post("/api/tools/nope/toggle", params={"enabled": False})
    assert response.status_code == 404


def test_invoke_tool_directly(client) -> None:
    """直接调用工具（调试用）：知识库为空时返回"拒答"这种正常结果。"""
    body = client.post(
        "/api/tools/knowledge_search/invoke", json={"args": {"query": "年假有几天"}}
    ).json()
    assert body["ok"] is True                    # 工具执行成功
    assert body["result"]["data"]["refused"] is True   # 但资料中没有
    assert body["attempts"] == 1


def test_invoke_tool_with_bad_args(client) -> None:
    """参数不对时返回结构化错误，便于调试。"""
    body = client.post("/api/tools/knowledge_search/invoke", json={"args": {}}).json()
    assert body["ok"] is False
    assert body["result"]["error_type"] == "tool_invalid_args"


# ---------------------------------------------------------------------------
# Agent 模式问答
# ---------------------------------------------------------------------------
def test_agent_mode_routes_to_knowledge_base(client) -> None:
    """文档类问题 → Agent 自主选择 knowledge_search，并在回答里如实体现结果。

    两点说明：

    * 本用例的核心是**验证路由**（是否选中了知识库工具、是否把来源带回来），
      所以断言集中在 ``tools_used`` / ``sources`` / ``tool_calls`` 上；
    * 测试环境用 hash 兜底 Embedding（非语义模型），分数偏低会被默认阈值拦成拒答，
      因此这里**不断言"答案里有具体内容"**——拒答也是合法结果，而且拒答话术必须是标准的。
    """
    upload(client)

    body = client.post(
        "/api/chat",
        json={"question": "年假有几天", "mode": "agent", "session_id": "sess_agent_1"},
    ).json()

    assert body["ok"] is True
    assert body["mode"] == "agent"
    assert body["tools_used"] == ["knowledge_search"]
    assert body["steps"] >= 1
    assert body["tool_calls"] and body["tool_calls"][0]["tool"] == "knowledge_search"
    assert body["tool_calls"][0]["ok"] is True, "工具本身应当执行成功"

    if body["refused"]:
        # 拒答必须是标准话术，且给出原因
        assert body["answer"] == REFUSAL_MESSAGE
        assert body["refuse_reason"]
    else:
        assert body["sources"], "非拒答时必须带来源"
    assert body["trace_id"].startswith("run_")


def test_agent_mode_knowledge_tool_returns_answer_when_threshold_relaxed(client) -> None:
    """把阈值放宽（直接调用工具）后，知识库能给出答案与来源。

    这条用例把"检索质量"与"路由正确性"分开验证：Agent 负责选对工具，
    工具自身的召回质量由 B3 的检索测试与评测集覆盖。
    """
    upload(client)
    body = client.post(
        "/api/tools/knowledge_search/invoke",
        json={"args": {"query": "年假有几天", "threshold": 0}},
    ).json()

    assert body["ok"] is True
    assert body["result"]["data"]["refused"] is False
    assert body["result"]["data"]["sources"]
    assert body["result"]["data"]["answer"]


def test_agent_mode_routes_to_trip_planner(client) -> None:
    """旅行需求 → Agent 自主选择 trip_planner（同一套协议）。"""
    body = client.post(
        "/api/chat",
        json={"question": "帮我规划北京三日游，喜欢历史文化", "mode": "agent"},
    ).json()

    assert body["ok"] is True
    assert body["tools_used"] == ["trip_planner"]
    assert "北京" in body["answer"]
    assert body["tool_calls"][0]["tool"] == "trip_planner"


def test_agent_mode_refuses_unknown_question(client) -> None:
    """文档外问题 → 标准拒答话术（Agent 模式下同样成立）。"""
    upload(client)
    body = client.post(
        "/api/chat",
        json={"question": "2022 年世界杯冠军是谁", "mode": "agent"},
    ).json()
    assert body["refused"] is True
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["status"] == "refused"


def test_agent_mode_returns_event_stream(client) -> None:
    """响应里带事件流，前端可据此回放决策过程（B5 变成实时 SSE）。"""
    upload(client)
    body = client.post(
        "/api/chat", json={"question": "年假有几天", "mode": "agent"}
    ).json()

    events = body["events"]
    assert events, "Agent 模式必须返回事件流"
    types = [event["type"] for event in events]
    assert types[0] == "plan"
    assert "tool_start" in types
    assert "tool_end" in types
    assert "final" in types
    assert types[-1] == "done"
    # 事件带序号与时间戳（前端排序与展示依赖）
    assert [event["seq"] for event in events] == list(range(len(events)))
    assert all(event["ts"] > 0 for event in events)
    assert all(event["run_id"] == body["trace_id"] for event in events)


def test_agent_mode_respects_max_steps(client) -> None:
    """步数预算可通过请求覆盖，且响应里如实报告步数。"""
    upload(client)
    body = client.post(
        "/api/chat",
        json={"question": "年假有几天", "mode": "agent", "max_steps": 1},
    ).json()
    assert body["steps"] <= 1
    assert body["meta"]["step_budget"] == 1


def test_rag_and_agent_modes_both_work(client) -> None:
    """两种模式各有分工：rag 走单工具检索管道，agent 走自主编排。

    rag 模式支持显式 ``threshold``（会一路透传到检索层），agent 模式下阈值由
    Planner 决定（默认取配置值），因此这里对两者的断言不同——这正是"两条路径"的
    差异所在，也解释了为什么保留 rag 模式作为可预测的回归基线。
    """
    upload(client)

    rag = client.post(
        "/api/chat", json={"question": "年假有几天", "mode": "rag", "threshold": 0}
    ).json()
    agent = client.post(
        "/api/chat", json={"question": "年假有几天", "mode": "agent"}
    ).json()

    # rag：单工具、有检索统计、阈值可控
    assert rag["mode"] == "rag"
    assert rag["tools_used"] == []
    assert rag["retrieval"], "rag 模式返回检索统计"
    assert rag["answer"] and rag["answer"] != REFUSAL_MESSAGE
    assert rag["sources"]

    # agent：经过工具层，工具调用被记录
    assert agent["mode"] == "agent"
    assert agent["tools_used"] == ["knowledge_search"]
    assert agent["tool_calls"]
    assert agent["answer"]


def test_default_mode_is_rag(client) -> None:
    """默认模式保持 raq（不破坏既有接口行为）。"""
    upload(client)
    body = client.post("/api/chat", json={"question": "年假有几天"}).json()
    assert body["mode"] == "rag"


# ---------------------------------------------------------------------------
# 轨迹
# ---------------------------------------------------------------------------
def test_traces_list_and_detail(client) -> None:
    """轨迹列表与详情：能看到 Planner 决策与工具调用。"""
    upload(client)
    response = client.post(
        "/api/chat", json={"question": "年假有几天", "mode": "agent", "session_id": "sess_trace"}
    ).json()
    run_id = response["trace_id"]

    listing = client.get("/api/traces", params={"limit": 10}).json()
    assert listing["ok"] is True
    assert listing["total"] >= 1
    assert any(item["run_id"] == run_id for item in listing["traces"])

    detail = client.get(f"/api/traces/{run_id}").json()
    assert detail["ok"] is True
    assert detail["question"] == "年假有几天"
    assert detail["tools_used"] == ["knowledge_search"]
    steps = detail["steps_detail"]
    assert steps
    assert any(step["type"] == "tool" and step["tool_name"] == "knowledge_search" for step in steps)


def test_traces_filter_by_session(client) -> None:
    """按会话过滤轨迹。"""
    upload(client)
    client.post("/api/chat", json={"question": "年假有几天", "mode": "agent", "session_id": "sess_A"})
    client.post("/api/chat", json={"question": "报销标准是多少", "mode": "agent", "session_id": "sess_B"})

    body = client.get("/api/traces", params={"session_id": "sess_A"}).json()
    assert body["total"] >= 1
    assert all(item["session_id"] == "sess_A" for item in body["traces"])


def test_trace_detail_404(client) -> None:
    """不存在的轨迹 → 404。"""
    assert client.get("/api/traces/run_不存在").status_code == 404


def test_openapi_lists_b4_endpoints(client) -> None:
    """OpenAPI 文档包含 B4 新增接口。"""
    schema = client.get("/openapi.json").json()
    assert "/api/tools" in schema["paths"]
    assert "/api/tools/{name}/invoke" in schema["paths"]
    assert "/api/tools/{name}/toggle" in schema["paths"]
    assert "/api/traces" in schema["paths"]
    assert "/api/traces/{run_id}" in schema["paths"]
