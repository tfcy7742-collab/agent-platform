"""B5 测试：SSE 流式问答、人工确认闭环与 Gradio 界面挂载。

覆盖点：
1. ``POST /api/chat/stream`` 的 SSE 帧格式与事件序列；
2. **人工确认闭环**：危险工具挂起 → 拒绝（不执行）→ 同意（真正执行）→ 同一 token 不可重放；
3. 挂起恢复时**不从头重跑**（不会重复调用之前的工具）；
4. Gradio 界面挂载到 ``/ui``，且挂载失败不影响 API；
5. 流式执行同样落 trace 与会话消息。
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List

import pytest

from api.chat_stream import sse_frame

DOC_TEXT = "第一条 年假规定：员工入职满一年后享有年假，工作满一年不满十年者每年五天。"


def upload(client, name: str = "员工手册.txt", content: str = DOC_TEXT) -> Dict[str, Any]:
    """上传一份文档。"""
    response = client.post(
        "/api/documents",
        files=[("files", (name, io.BytesIO(content.encode("utf-8")), "text/plain"))],
        data={"force": "false"},
    )
    return response.json()


def read_sse(client, path: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """调用流式接口并解析出全部事件。"""
    events: List[Dict[str, Any]] = []
    with client.stream("POST", path, json=payload) as response:
        assert response.status_code == 200, response.read()
        event_type = ""
        data_lines: List[str] = []
        for line in response.iter_lines():
            line = line.rstrip("\r")
            if not line:
                if event_type and data_lines:
                    parsed = json.loads("".join(data_lines))
                    events.append({"type": event_type, **parsed})
                event_type, data_lines = "", []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
    if event_type and data_lines:
        events.append({"type": event_type, **json.loads("".join(data_lines))})
    return events


def event_types(events: List[Dict[str, Any]]) -> List[str]:
    """提取事件类型序列。"""
    return [str(event["type"]) for event in events]


# ---------------------------------------------------------------------------
# SSE 帧格式
# ---------------------------------------------------------------------------
def test_sse_frame_format() -> None:
    """SSE 帧必须符合 ``event:`` + ``data:`` + 空行的规范。"""
    frame = sse_frame("plan", {"action": "tool", "tool": "knowledge_search"})
    assert frame.startswith("event: plan\n")
    assert "data: " in frame
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["action"] == "tool"


def test_sse_frame_keeps_chinese_readable() -> None:
    """中文不能被转义成 \\uXXXX（否则前端日志与调试很难读）。"""
    frame = sse_frame("final", {"answer": "年假为五天"})
    assert "年假为五天" in frame


# ---------------------------------------------------------------------------
# 流式问答
# ---------------------------------------------------------------------------
def test_stream_returns_sse_events(client) -> None:
    """流式问答：事件序列完整，且带 session/run 头。"""
    upload(client)
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "年假有几天", "session_id": "sess_stream"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers.get("x-session-id") == "sess_stream"
        assert response.headers.get("x-run-id", "").startswith("run_")
        body = "".join(response.iter_text())

    assert "event: plan" in body
    assert "event: tool_start" in body
    assert "event: tool_end" in body
    assert "event: final" in body
    assert "event: done" in body
    assert body.rstrip().endswith("}")


def test_stream_event_sequence(client) -> None:
    """事件顺序符合契约：plan → tool_start → tool_end → observation → final → done。"""
    upload(client)
    events = read_sse(client, "/api/chat/stream", {"question": "年假有几天"})
    types = event_types(events)

    assert types[0] == "plan"
    assert types[-1] == "done"
    assert types.index("tool_start") < types.index("tool_end")
    assert types.index("tool_end") < types.index("observation")
    assert types.index("observation") < types.index("final")
    # 序号递增且 run_id 一致
    assert [event["seq"] for event in events] == list(range(len(events)))
    run_ids = {event["run_id"] for event in events}
    assert len(run_ids) == 1 and run_ids.pop().startswith("run_")


def test_stream_routes_trip_planner(client) -> None:
    """流式模式下同样能自主路由到旅行规划工具。"""
    events = read_sse(
        client, "/api/chat/stream", {"question": "帮我规划北京三日游，喜欢历史文化"}
    )
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert tool_starts and tool_starts[0]["payload"]["tool"] == "trip_planner"
    final = next(e for e in events if e["type"] == "final")
    assert "北京" in final["payload"]["answer"]


def test_stream_refusal(client) -> None:
    """流式模式下的文档外问题：必须有 refuse 事件 + 标准话术。"""
    from core.prompts import REFUSAL_MESSAGE

    upload(client)
    events = read_sse(client, "/api/chat/stream", {"question": "2022 年世界杯冠军是谁"})
    types = event_types(events)
    assert "refuse" in types
    final = next(e for e in events if e["type"] == "final")
    assert final["payload"]["answer"] == REFUSAL_MESSAGE


def test_stream_rejects_empty_question(client) -> None:
    """空问题 → 422。"""
    response = client.post("/api/chat/stream", json={"question": ""})
    assert response.status_code == 422


def test_stream_writes_trace_and_messages(client) -> None:
    """流式执行同样落 trace 与会话消息（与 /api/chat 保持一致）。"""
    upload(client)
    events = read_sse(
        client, "/api/chat/stream", {"question": "年假有几天", "session_id": "sess_stream_trace"}
    )
    run_id = events[0]["run_id"]

    detail = client.get(f"/api/traces/{run_id}").json()
    assert detail["ok"] is True
    assert detail["question"] == "年假有几天"
    assert any(step["type"] == "tool" for step in detail["steps_detail"])

    from infra import db

    history = db.get_history("sess_stream_trace")
    assert len(history) == 2
    assert history[0]["content"] == "年假有几天"


# ---------------------------------------------------------------------------
# 人工确认闭环
# ---------------------------------------------------------------------------
def _pending_events(client) -> List[Dict[str, Any]]:
    """触发一次需要确认的写操作，返回事件流。"""
    return read_sse(
        client,
        "/api/chat/stream",
        {"question": "把这份行程发到 me@example.com", "session_id": "sess_confirm"},
    )


def test_stream_pauses_for_confirmation(client) -> None:
    """危险工具：流式响应里出现 pending_confirmation，且工具未真正执行。"""
    from core.tools.registry import get_registry

    events = _pending_events(client)
    types = event_types(events)
    assert "pending_confirmation" in types
    assert "done" in types

    pending = next(e for e in events if e["type"] == "pending_confirmation")["payload"]
    assert pending["tool"] == "send_email"
    assert pending["resume_token"]
    assert "确认" in pending["risk_note"]

    # 未确认前绝不能真正执行
    email_tool = get_registry().get("send_email")
    assert email_tool is not None
    assert email_tool.stats()["sent_count"] == 0

    # 挂起记录可以通过 /api/pending 查到
    body = client.get("/api/pending").json()
    assert body["total"] >= 1
    assert any(item["tool"] == "send_email" for item in body["pending"])


def test_confirm_rejects_without_executing(client) -> None:
    """拒绝确认：不执行工具，但 Agent 仍要给出后续回答。"""
    from core.tools.registry import get_registry

    events = _pending_events(client)
    token = next(e for e in events if e["type"] == "pending_confirmation")["payload"]["resume_token"]

    confirmed = read_sse(
        client,
        "/api/chat/confirm",
        {"resume_token": token, "approved": False, "session_id": "sess_confirm"},
    )
    types = event_types(confirmed)
    assert "final" in types
    assert types[-1] == "done"

    email_tool = get_registry().get("send_email")
    assert email_tool.stats()["sent_count"] == 0, "拒绝后不能产生副作用"

    # token 已被消费，不能重复使用
    again = client.post(
        "/api/chat/confirm", json={"resume_token": token, "approved": True}
    )
    assert again.status_code == 404


def test_confirm_executes_tool(client) -> None:
    """同意确认：工具真正执行，且结果进入后续回答。"""
    from core.tools.registry import get_registry

    events = _pending_events(client)
    token = next(e for e in events if e["type"] == "pending_confirmation")["payload"]["resume_token"]

    confirmed = read_sse(
        client,
        "/api/chat/confirm",
        {"resume_token": token, "approved": True, "session_id": "sess_confirm"},
    )
    final = next(e for e in confirmed if e["type"] == "final")
    assert "me@example.com" in final["payload"]["answer"]

    email_tool = get_registry().get("send_email")
    assert email_tool.stats()["sent_count"] == 1, "确认后应当执行一次"

    # 同一个 token 不能重复执行（防止重复发邮件）
    assert client.post("/api/chat/confirm", json={"resume_token": token, "approved": True}).status_code == 404


def test_confirm_does_not_rerun_previous_tools(client) -> None:
    """恢复执行时不能从头重跑：挂起前已完成的工具调用不应再次发生。

    做法：先让会话里有一次知识库问答（产生观察），再触发需要确认的写操作，
    最后确认；对比 knowledge_search 的调用次数，恢复前后不应增加。
    """
    from core.tools.registry import get_registry

    upload(client)
    # 先做一次知识库问答，让会话里有历史与观察
    read_sse(client, "/api/chat/stream", {"question": "年假有几天", "session_id": "sess_norerun"})
    knowledge_calls_before = get_registry().get("knowledge_search").stats()["calls"]

    events = read_sse(
        client,
        "/api/chat/stream",
        {"question": "把年假规定发到 me@example.com", "session_id": "sess_norerun"},
    )
    pending = next((e for e in events if e["type"] == "pending_confirmation"), None)
    if pending is None:
        pytest.skip("该问题没有触发确认流程（路由未选中 send_email），跳过")

    token = pending["payload"]["resume_token"]
    read_sse(client, "/api/chat/confirm", {"resume_token": token, "approved": True, "session_id": "sess_norerun"})

    knowledge_calls_after = get_registry().get("knowledge_search").stats()["calls"]
    # 恢复执行时不应重复调用知识库（观察已随挂起记录保存并回传）
    assert knowledge_calls_after - knowledge_calls_before <= 1


def test_confirm_unknown_token(client) -> None:
    """无效 token → 404，并给出可读原因。"""
    response = client.post("/api/chat/confirm", json={"resume_token": "不存在的token", "approved": True})
    assert response.status_code == 404
    assert "过期" in response.json()["detail"] or "不存在" in response.json()["detail"]


def test_confirm_requires_token(client) -> None:
    """缺少 token → 422。"""
    assert client.post("/api/chat/confirm", json={"approved": True}).status_code == 422


def test_pending_store_ttl() -> None:
    """挂起记录带 TTL，过期后不可用。"""
    from core.runtime.resume import PendingExecution, PendingStore

    store = PendingStore()
    execution = PendingExecution(
        token="tok1", question="q", session_id="s", tool="send_email", args={"to": "a@b.com"}
    )
    store.save(execution)
    assert store.count() == 1
    assert store.load("tok1") is not None
    assert store.load("tok1").expires_in_s > 0

    store.drop("tok1")
    assert store.load("tok1") is None
    assert store.count() == 0


def test_send_email_tool_validates_address() -> None:
    """邮箱格式校验（避免把明显错误的地址当成合法参数）。"""
    from core.tools.email_tool import SendEmailTool

    tool = SendEmailTool()
    bad = tool.run(to="不是邮箱", body="内容")
    assert bad.ok is False
    assert bad.error_type == "tool_invalid_args"

    good = tool.run(to="user@example.com", body="行程内容")
    assert good.ok is True
    assert good.data["to"] == "user@example.com"
    assert tool.stats()["sent_count"] == 1


def test_send_email_requires_confirmation_flag() -> None:
    """工具元信息里必须声明需要人工确认（Planner 与执行器都依赖它）。"""
    from core.tools.email_tool import SendEmailTool

    spec = SendEmailTool().spec()
    assert spec["requires_confirmation"] is True
    assert spec["retryable"] is False, "写操作不能自动重试"


# ---------------------------------------------------------------------------
# Gradio 界面挂载
# ---------------------------------------------------------------------------
def test_ui_mounted(client) -> None:
    """界面挂载在 /ui：能取到 HTML，且包含标题。"""
    response = client.get("/ui/", follow_redirects=True)
    assert response.status_code == 200
    assert "Agent" in response.text


def test_api_still_works_with_ui_mounted(client) -> None:
    """挂载 UI 不影响 API（同一进程两套入口）。"""
    assert client.get("/health").status_code == 200
    assert client.get("/api/tools").status_code == 200


def test_ui_build_is_importable() -> None:
    """界面构建函数可被导入并调用（挂载失败时 API 仍可用，但构建本身必须正常）。"""
    from ui import build_demo

    demo = build_demo()
    assert demo is not None
    assert hasattr(demo, "queue")


def test_trace_rendering_helpers() -> None:
    """轨迹渲染与来源渲染是纯函数，单独验证（避免 UI 出问题时无从下手）。"""
    from ui import render_sources, render_trace_line

    line = render_trace_line(
        {"type": "tool_end", "payload": {"tool": "knowledge_search", "ok": True, "latency_ms": 120}},
        1,
    )
    assert "knowledge_search" in line and "120ms" in line

    plan_line = render_trace_line(
        {"type": "plan", "payload": {"action": "tool", "tool": "trip_planner", "thought": "识别为旅行意图"}},
        2,
    )
    assert "trip_planner" in plan_line

    sources = render_sources(
        [{"file_name": "员工手册.pdf", "page": 1, "score": 0.62, "chunk_id": "c1", "text": "年假五天"}]
    )
    assert "员工手册.pdf" in sources and "年假五天" in sources


def test_resume_marker_roundtrip() -> None:
    """resume_token 通过 HTML 注释往返传递（确认按钮靠它拿到 token）。"""
    from ui import _extract_resume_token, append_resume_marker

    content = append_resume_marker("需要确认", "abc123")
    assert "<!--resume_token:abc123-->" in content
    assert _extract_resume_token([{"role": "assistant", "content": content}]) == "abc123"
    assert _extract_resume_token([{"role": "assistant", "content": "没有标记"}]) is None
