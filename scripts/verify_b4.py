"""B4 端到端验收：Agent 自主路由 + 工具协议 + 可观测性。

前置：
    1. 已下载模型（scripts/download_embedding_model.py）
    2. 服务已启动（python app.py，EMBEDDING_BACKEND=auto 或 sentence_transformers）
    3. 样例文档已生成（scripts/make_sample_docs.py）

运行：
    .venv\\Scripts\\python.exe scripts\\verify_b4.py

验收项：
    [1] 工具目录：两个工具的 Schema / 成本 / 耗时档位是否完整
    [2] 自主路由：文档类问题 → knowledge_search；旅行需求 → trip_planner
        （没有任何 if-else 硬编码，完全由 Planner 决定）
    [3] 同一套协议调用异构能力：一个是"检索+生成"管道，一个是 4 智能体子系统
    [4] 拒答优先级：文档外问题必须返回标准话术（Agent 模式下同样成立）
    [5] 可观测性：trace 里能看到决策、工具调用、耗时
    [6] 降级可见：离线/失败时标记 degraded，不伪装
"""

from __future__ import annotations

import json
import mimetypes
import sys
import urllib.parse
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

# (问题, 期望使用的工具)
ROUTING_CASES = [
    ("年假有几天", "knowledge_search"),
    ("报销住宿标准是多少", "knowledge_search"),
    ("接口返回 502 怎么排查", "knowledge_search"),
    ("帮我规划北京三日游，喜欢历史文化，预算 8000", "trip_planner"),
    ("想去成都玩四天，主要是美食", "trip_planner"),
]


def get_json(path: str) -> dict:
    """GET 请求。"""
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict) -> dict:
    """POST JSON 请求。"""
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def post_files(path: str, files: list[Path]) -> dict:
    """上传文件（multipart）。"""
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
    print("B4 端到端验收：Agent 自主路由 + 工具协议 + 可观测性")
    print("=" * 84)

    health = get_json("/health")
    print(f"LLM：online={health['capabilities']['llm']['online']}（{health['capabilities']['llm']['model']}）")
    print(f"Embedding：{health['capabilities']['embedding']['backend']}")

    # ---------------------------------------------------------------- 准备知识库
    files = sorted(path for path in SAMPLE_DIR.iterdir() if path.is_file()) if SAMPLE_DIR.exists() else []
    if not files:
        print("样例文档不存在，请先运行 scripts/make_sample_docs.py")
        return 1
    report = post_files("/api/documents", files)
    print(f"知识库：{report['succeeded']} 个新增，{report['skipped']} 个跳过，共 {report['store']['chunks']} 个块")

    # ---------------------------------------------------------------- [1] 工具目录
    print("-" * 84)
    print("[1] 工具目录（Planner 决策的依据）")
    tools = {item["name"]: item for item in get_json("/api/tools")["tools"]}
    for name, item in tools.items():
        print(
            f"    · {name}：成本={item['est_cost']} 耗时={item['est_latency']} "
            f"超时={item['timeout_s']}s 可重试={item['retryable']} 需确认={item['requires_confirmation']}"
        )
        print(f"      说明：{item['description'][:70]}…")
    check({"knowledge_search", "trip_planner"} <= set(tools), "两个内置工具都已注册")
    check(tools["trip_planner"]["est_cost"] == "high", "旅行规划标记为高成本（内部 4 个智能体）")
    check(tools["trip_planner"]["retryable"] is False, "旅行规划不可重试（重试等于重跑 4 个智能体）")

    # ---------------------------------------------------------------- [2] 自主路由
    print("-" * 84)
    print("[2] 自主路由（无 if-else 硬编码，完全由 Planner 决定）")
    session = f"sess_b4_{uuid.uuid4().hex[:6]}"
    for question, expected_tool in ROUTING_CASES:
        body = post_json(
            "/api/chat",
            {"question": question, "mode": "agent", "session_id": session},
        )
        used = body.get("tools_used") or []
        hit = used == [expected_tool]
        first_call = (body.get("tool_calls") or [{}])[0]
        print(
            f"  [{'PASS' if hit else 'FAIL'}] 「{question}」\n"
            f"         路由 → {used or '（未调用工具）'}（期望 {expected_tool}）"
            f"｜状态={body['status']}｜步数={body['steps']}｜耗时={body['latency_ms']}ms"
            f"｜工具耗时={first_call.get('latency_ms')}ms"
        )
        if not hit:
            failures.append(f"路由错误：{question} → {used}（期望 {expected_tool}）")

    # ---------------------------------------------------------------- [3] 异构能力
    print("-" * 84)
    print("[3] 同一套协议编排异构能力")
    trip_body = post_json(
        "/api/chat", {"question": "帮我规划上海三日游，喜欢美食", "mode": "agent"}
    )
    print(f"    旅行规划回答（前 120 字）：{trip_body['answer'][:120].replace(chr(10), ' ')}")
    check("上海" in trip_body["answer"], "行程规划结果里包含目标城市")

    kb_body = post_json("/api/chat", {"question": "年假有几天", "mode": "agent"})
    print(f"    知识库回答（前 120 字）：{kb_body['answer'][:120].replace(chr(10), ' ')}")
    check(bool(kb_body["sources"]) or kb_body["refused"], "知识库路径返回来源或明确拒答")

    # ---------------------------------------------------------------- [4] 拒答优先级
    print("-" * 84)
    print("[4] 拒答优先级（Agent 模式下同样成立）")
    for question in ("2022 年世界杯冠军是谁", "推荐几部科幻电影", "怎么给猫剪指甲"):
        body = post_json("/api/chat", {"question": question, "mode": "agent"})
        exact = body["answer"] == REFUSAL
        print(
            f"  [{'PASS' if exact else 'FAIL'}] 「{question}」→ refused={body['refused']}"
            f"｜原因={body['refuse_reason']}"
        )
        if not exact:
            failures.append(f"文档外问题未标准拒答：{question}")

    # ---------------------------------------------------------------- [5] 可观测性
    print("-" * 84)
    print("[5] 可观测性（trace 能看到决策与工具调用）")
    traces = get_json("/api/traces?limit=3")
    check(traces["total"] >= 1, f"已记录 {traces['total']} 条轨迹")
    if traces["traces"]:
        run_id = traces["traces"][0]["run_id"]
        detail = get_json(f"/api/traces/{run_id}")
        steps = detail.get("steps_detail") or []
        types = [step["type"] for step in steps]
        print(f"    run_id={run_id}｜问题={detail['question']}")
        print(f"    工具={detail['tools_used']}｜状态={detail['status']}｜总耗时={detail['total_ms']}ms")
        for step in steps:
            extra = ""
            if step["type"] == "tool":
                extra = f"（{step['tool_name']}，{step['latency_ms']}ms，ok={bool(step['ok'])}）"
            print(f"      · 步骤{step['idx']} {step['type']}{extra}")
        check("tool" in types, "trace 里记录了工具调用步骤")
        check("plan" in types, "trace 里记录了 Planner 决策步骤")
        check(detail["total_ms"] > 0, "trace 里记录了耗时")
    stats = get_json("/api/tools/stats")
    print(f"    工具调用统计：{json.dumps(stats['stats'], ensure_ascii=False)[:200]}")
    note(any(item["calls"] > 0 for item in stats["stats"]), "工具调用统计已累计")

    # ---------------------------------------------------------------- [6] 降级可见
    print("-" * 84)
    print("[6] 降级可见（不伪装）")
    body = post_json("/api/chat", {"question": "帮我规划北京两日游", "mode": "agent"})
    degraded = bool(body.get("llm", {}).get("degraded"))
    print(
        f"    离线模式下旅行规划：degraded={degraded}｜"
        f"生成方式见回答：{body['answer'][:60].replace(chr(10), ' ')}"
    )
    note(degraded, "离线/降级时明确标记 degraded（不是伪装成大模型输出）")

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
