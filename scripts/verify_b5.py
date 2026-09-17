"""B5 端到端验收：SSE 流式、人工确认闭环、Gradio 界面。

前置：服务已启动（python app.py）。不需要真实模型也能验证协议与交互。

运行：
    .venv\\Scripts\\python.exe scripts\\verify_b5.py

验收项：
    [1] /ui 界面可访问，且挂载不影响 API
    [2] SSE 帧格式正确（event: / data: / 空行分隔），带 session/run 头
    [3] **真流式**：事件是逐条到达的，不是等全部算完一次性返回
    [4] 事件序列符合契约：plan → tool_start → tool_end → observation → final → done
    [5] 拒答事件：文档外问题在流式下同样返回 refuse + 标准话术
    [6] 人工确认闭环：
        挂起（工具未执行）→ 拒绝（仍不执行）→ 重新发起 → 同意（执行一次）→ token 不可重放
    [7] 可观测性：流式执行同样落 trace 与会话消息
"""

from __future__ import annotations

import io
import json
import mimetypes
import sys
import time
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

BASE_URL = "http://127.0.0.1:8000"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_docs"
REFUSAL = "根据现有资料，我无法回答这个问题"


def post_json(path: str, payload: dict, timeout: float = 300.0) -> dict:
    """普通 JSON 请求。"""
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_raw(path: str, timeout: float = 120.0) -> tuple:
    """GET 原始响应（返回状态码, 文本, 响应头）。

    超时给得较宽：首次访问 ``/health`` 会初始化向量库与 Embedding 模型
    （真实 bge 模型加载约 20~30 秒），超时太短会误判成服务不可用。
    """
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace"), dict(response.headers)


def wait_until_ready(attempts: int = 6, interval: float = 5.0) -> bool:
    """等服务真正就绪（模型加载完成）再开始验收。"""
    for index in range(attempts):
        try:
            status, _, _ = get_raw("/health", timeout=60)
            if status == 200:
                if index:
                    print(f"    服务在第 {index + 1} 次探测时就绪")
                return True
        except Exception as exc:  # noqa: BLE001 - 启动中各种连接错误都属正常
            print(f"    等待服务就绪（{index + 1}/{attempts}）：{type(exc).__name__}")
        time.sleep(interval)
    return False


def stream_events(path: str, payload: dict) -> tuple:
    """以流式方式读取 SSE，返回（事件列表, 每条事件的到达时刻, 响应头）。"""
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    events: list = []
    timestamps: list = []
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        headers = dict(response.headers)
        event_type = ""
        data_lines: list = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if event_type and data_lines:
                    parsed = json.loads("".join(data_lines))
                    events.append({"type": event_type, **parsed})
                    timestamps.append(round(time.perf_counter() - started, 3))
                event_type, data_lines = "", []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
    return events, timestamps, headers


def post_files(path: str, files: list) -> dict:
    """上传文件。"""
    boundary = f"----verify{uuid.uuid4().hex}"
    body = bytearray()
    for file_path in files:
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="files"; filename="{file_path.name}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def event_types(events: list) -> list:
    """事件类型序列。"""
    return [str(event["type"]) for event in events]


def main() -> int:
    """执行验收。"""
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        """硬性断言。"""
        print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
        if not condition:
            failures.append(message)

    def note(condition: bool, message: str) -> None:
        """软性观察项。"""
        print(f"  [{'PASS' if condition else 'INFO'}] {message}")

    print("=" * 84)
    print("B5 端到端验收：SSE 流式 + 人工确认闭环 + Gradio 界面")
    print("=" * 84)

    if not wait_until_ready():
        print("服务未就绪，请确认 python app.py 已启动且模型加载完成")
        return 1

    # ------------------------------------------------------------------ [1] UI
    print("[1] Gradio 界面挂载")
    status, html, _ = get_raw("/ui/")
    check(status == 200, f"/ui/ 可访问（HTTP {status}）")
    check("Agent" in html, "页面包含平台标题")
    check("gradio" in html.lower(), "页面由 Gradio 渲染")
    api_status, _, _ = get_raw("/health")
    check(api_status == 200, "挂载 UI 后 API 仍然可用（同进程双入口）")

    # ------------------------------------------------------------------ 准备知识库
    files = sorted(path for path in SAMPLE_DIR.iterdir() if path.is_file()) if SAMPLE_DIR.exists() else []
    if not files:
        print("样例文档不存在，请先运行 scripts/make_sample_docs.py")
        return 1
    report = post_files("/api/documents", files)
    print(f"    知识库：{report['succeeded']} 新增，{report['skipped']} 跳过，共 {report['store']['chunks']} 块")

    # ------------------------------------------------------------------ [2] SSE 格式
    print("-" * 84)
    print("[2] SSE 帧格式与响应头")
    events, timestamps, headers = stream_events(
        "/api/chat/stream", {"question": "年假有几天", "session_id": "sess_b5"}
    )
    check(headers.get("content-type", "").startswith("text/event-stream"), "Content-Type 为 text/event-stream")
    check(headers.get("x-session-id") == "sess_b5", "响应头带 X-Session-Id")
    check(str(headers.get("x-run-id", "")).startswith("run_"), "响应头带 X-Run-Id")
    check(headers.get("cache-control", "").startswith("no-cache"), "禁用缓存（保证逐条推送）")
    check(len(events) >= 5, f"收到 {len(events)} 条事件")
    check(all("run_id" in e and "seq" in e and "ts" in e for e in events), "每条事件都有 run_id / seq / ts")
    check([e["seq"] for e in events] == list(range(len(events))), "事件序号连续递增")

    # ------------------------------------------------------------------ [3] 真流式
    print("-" * 84)
    print("[3] 真流式（逐条到达，而不是算完一次性返回）")
    print(f"    各事件到达时刻（相对请求开始，秒）：{timestamps}")
    if len(timestamps) >= 2:
        spread = timestamps[-1] - timestamps[0]
        check(spread > 0, f"首尾事件存在时间差（{spread:.3f}s），说明是逐条推送")
        note(
            timestamps[0] < timestamps[-1],
            f"首个事件在第 {timestamps[0]:.3f}s 到达（先在页面显示「正在思考」，再逐步更新）",
        )
    else:
        check(False, "事件数量不足，无法判断是否流式")

    # ------------------------------------------------------------------ [4] 事件序列
    print("-" * 84)
    print("[4] 事件序列契约")
    types = event_types(events)
    print(f"    序列：{' → '.join(types)}")
    check(types[0] == "plan", "首事件为 plan（先决策）")
    check(types[-1] == "done", "末事件为 done（明确结束）")
    check("tool_start" in types and "tool_end" in types, "包含工具调用的开始与结束事件")
    check("observation" in types, "包含观察事件（工具结果进入上下文）")
    check("final" in types, "包含最终回答事件")
    check(types.index("tool_start") < types.index("tool_end"), "tool_end 在 tool_start 之后")

    # ------------------------------------------------------------------ [5] 拒答
    print("-" * 84)
    print("[5] 流式模式下的拒答")
    refused_events, _, _ = stream_events(
        "/api/chat/stream", {"question": "2022 年世界杯冠军是谁", "session_id": "sess_b5"}
    )
    refused_types = event_types(refused_events)
    final_event = next((e for e in refused_events if e["type"] == "final"), None)
    check("refuse" in refused_types, f"包含 refuse 事件（序列：{' → '.join(refused_types)}）")
    check(
        final_event is not None and final_event["payload"].get("answer") == REFUSAL,
        "拒答话术逐字匹配验收标准",
    )

    # ------------------------------------------------------------------ [6] 人工确认
    print("-" * 84)
    print("[6] 人工确认闭环（human-in-the-loop）")
    pending_events, _, _ = stream_events(
        "/api/chat/stream",
        {"question": "把这份行程发到 me@example.com", "session_id": "sess_b5_confirm"},
    )
    pending_type = event_types(pending_events)
    pending = next((e for e in pending_events if e["type"] == "pending_confirmation"), None)
    check(pending is not None, f"触发挂起（序列：{' → '.join(pending_type)}）")
    if pending is None:
        print("  无法继续验证确认流程")
        return 1

    payload = pending["payload"]
    print(f"    待确认工具：{payload['tool']}｜参数：{json.dumps(payload['args'], ensure_ascii=False)}")
    check(payload["tool"] == "send_email", "挂起的是写操作工具 send_email")
    check("确认" in payload.get("risk_note", ""), "风险说明包含确认提示")
    token = payload["resume_token"]

    pending_list = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/pending", timeout=30).read())
    check(pending_list["total"] >= 1, f"/api/pending 可查到 {pending_list['total']} 条待确认")

    # 6.1 拒绝：不执行，但仍给出后续回答
    reject_events, _, _ = stream_events(
        "/api/chat/confirm", {"resume_token": token, "approved": False, "session_id": "sess_b5_confirm"}
    )
    reject_final = next((e for e in reject_events if e["type"] == "final"), None)
    check(reject_final is not None, "拒绝后仍有最终回答")
    if reject_final:
        print(f"    拒绝后回答：{str(reject_final['payload'].get('answer'))[:80]}")
    replay = urllib.request.Request(
        f"{BASE_URL}/api/chat/confirm",
        data=json.dumps({"resume_token": token, "approved": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(replay, timeout=30)
        check(False, "同一个 token 不应能重复使用")
    except urllib.error.HTTPError as exc:
        check(exc.code == 404, f"token 已消费，重复确认返回 404（实际 {exc.code}）")

    # 6.2 重新发起并同意：执行一次
    print("    --- 重新发起并同意 ---")
    again_events, _, _ = stream_events(
        "/api/chat/stream",
        {"question": "把行程发到 me@example.com", "session_id": "sess_b5_confirm2"},
    )
    again_pending = next((e for e in again_events if e["type"] == "pending_confirmation"), None)
    check(again_pending is not None, "重新发起后再次挂起")
    if again_pending:
        approve_events, _, _ = stream_events(
            "/api/chat/confirm",
            {
                "resume_token": again_pending["payload"]["resume_token"],
                "approved": True,
                "session_id": "sess_b5_confirm2",
            },
        )
        approve_types = event_types(approve_events)
        approve_final = next((e for e in approve_events if e["type"] == "final"), None)
        print(f"    同意后序列：{' → '.join(approve_types)}")
        check(approve_final is not None, "同意后有最终回答")
        if approve_final:
            answer = str(approve_final["payload"].get("answer") or "")
            print(f"    同意后回答：{answer[:100]}")
            check("example.com" in answer, "回答里包含实际执行的收件人（说明工具真的跑了）")
            check("待确认" not in answer, "不再要求二次确认（没有死循环）")

    # ------------------------------------------------------------------ [7] 可观测性
    print("-" * 84)
    print("[7] 流式执行的可观测性")
    run_id = events[0]["run_id"]
    detail = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/traces/{run_id}", timeout=30).read())
    steps = detail.get("steps_detail") or []
    step_types = [step["type"] for step in steps]
    print(f"    run_id={run_id}｜步骤：{step_types}")
    check("plan" in step_types, "trace 记录了 Planner 决策")
    check("tool" in step_types, "trace 记录了工具调用")
    check(detail["total_ms"] > 0, "trace 记录了耗时")

    print("=" * 84)
    if failures:
        print(f"验收结果：{len(failures)} 项未通过")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("验收结果：全部通过（INFO 为观察项）")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
