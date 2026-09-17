"""LLM 连通性自检（换 Key / 换模型后跑一次，30 秒确认可用）。

检查项
------
[1] 配置：provider / base_url / model / Key（脱敏）
[2] 基础对话：能否正常调用并返回文本
[3] JSON 模式：能否稳定返回可解析的 JSON（Planner 与问答都依赖它）
[4] Function Calling：能否识别并按 schema 调用工具（Agent 自主路由依赖它）
[5] 真实 Agent 端到端：给几个不同意图的问题，看模型是否选对工具

用法：
    .venv\\Scripts\\python.exe scripts\\check_llm.py
    .venv\\Scripts\\python\\exe scripts\\check_llm.py --skip-agent   # 只查连通性

注意：脚本只打印脱敏后的 Key（前 6 后 4 位），不会输出明文。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def main() -> int:
    """执行自检。"""
    parser = argparse.ArgumentParser(description="DeepSeek 连通性自检")
    parser.add_argument("--skip-agent", action="store_true", help="跳过真实 Agent 端到端测试")
    args = parser.parse_args()

    from config.settings import get_settings

    settings = get_settings()
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        """断言并记录。"""
        print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
        if not condition:
            failures.append(message)

    print("=" * 80)
    print("LLM 连通性自检")
    print("=" * 80)

    # ---------------------------------------------------------------- [1] 配置
    print("[1] 配置")
    print(f"    provider ：{settings.llm_provider}")
    print(f"    base_url ：{settings.provider_base_url}")
    print(f"    model    ：{settings.provider_model}")
    print(f"    api_key  ：{settings.masked_key}")
    print(f"    mode     ：{settings.llm_mode}（llm_online={settings.llm_online}）")
    check(settings.has_api_key, "API Key 已配置且不是占位符")
    check(settings.llm_online, "llm_online 为 True（会真正调用大模型）")
    if not settings.llm_online:
        print("\n提示：请在 agent-platform\\.env 中设置 DEEPSEEK_API_KEY=sk-xxx，并把 LLM_MODE 设为 auto 或 online")
        return 1

    from core.llm import get_llm_client

    client = get_llm_client()
    if not client.available:
        print(f"\n[FAIL] LLM 客户端不可用：{client.describe().get('init_error')}")
        return 1

    # ---------------------------------------------------------------- [2] 基础对话
    print("-" * 80)
    print("[2] 基础对话")
    started = time.perf_counter()
    result = client.chat("你是一个简洁的助手。", "用一句话说明你是谁，不要超过 20 字。")
    elapsed = round(time.perf_counter() - started, 1)
    if result.ok:
        print(f"    返回（{elapsed}s，{result.total_tokens} tokens）：{result.text.strip()[:80]}")
    check(result.ok, f"调用成功（错误类型：{result.error_type}）")
    if not result.ok:
        print(f"\n排查建议：")
        print(f"  · 401/鉴权失败 → Key 与 base_url 是否属于同一家（DeepSeek 官方 vs 阿里云百炼）")
        print(f"  · 模型不存在 → DEEPSEEK_MODEL 是否是该平台真实存在的模型名")
        print(f"  · 连接失败 → 网络/代理，或 base_url 写错")
        return 1
    check(result.total_tokens > 0, f"返回了 token 用量（{result.total_tokens}）")

    # ---------------------------------------------------------------- [3] JSON 模式
    print("-" * 80)
    print("[3] JSON 模式（Planner 与问答都依赖）")
    parsed, json_result = client.chat_json(
        "你只输出 JSON，不要输出任何解释。",
        '请输出 {"city": "北京", "days": 3, "ok": true} 这样的结构，city 填"上海"，days 填 5。',
    )
    check(json_result.ok and isinstance(parsed, dict), f"返回可解析的 JSON（错误类型：{json_result.error_type}）")
    if isinstance(parsed, dict):
        print(f"    解析结果：{json.dumps(parsed, ensure_ascii=False)[:100]}")
        check(parsed.get("city") == "上海" and parsed.get("days") == 5, "字段值与要求一致")

    # ---------------------------------------------------------------- [4] Function Calling
    print("-" * 80)
    print("[4] Function Calling（Agent 自主路由依赖）")
    from core.tools.registry import bootstrap_tools

    registry = bootstrap_tools(force=True)
    tools = registry.openai_tools()
    print(f"    已注册工具：{', '.join(registry.names())}")

    tool_result = client.chat(
        "你可以调用工具。需要外部信息时先调用工具，不要凭记忆回答。",
        "帮我查一下公司年假有几天。",
    )
    check(tool_result.ok, "工具场景下的基础调用成功")

    # 用 deepseek 的 bind_tools 做一次真实的工具选择
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        bound = client._get_client().bind_tools(tools)  # noqa: SLF001 - 自检脚本允许访问内部
        message = bound.invoke(
            [
                SystemMessage(content="你需要外部信息时调用工具，不要凭记忆回答。"),
                HumanMessage(content="帮我规划一个去成都的四天旅行，喜欢美食。"),
            ]
        )
        calls = getattr(message, "tool_calls", None) or []
        chosen = [call.get("name") for call in calls]
        print(f"    模型选择的工具：{chosen or '（未调用工具）'}")
        check(bool(calls), "模型主动调用了工具")
        check(chosen[:1] == ["trip_planner"], f"为旅行需求选中了 trip_planner（实际：{chosen}）")
        if calls:
            print(f"    参数：{json.dumps(calls[0].get('args', {}), ensure_ascii=False)[:160]}")
    except Exception as exc:  # noqa: BLE001
        check(False, f"Function Calling 调用失败：{type(exc).__name__}: {exc}")

    if args.skip_agent:
        print("=" * 80)
        print(f"自检结果：{'全部通过' if not failures else f'{len(failures)} 项未通过'}")
        return 1 if failures else 0

    # ---------------------------------------------------------------- [5] 真实 Agent
    print("-" * 80)
    print("[5] 真实 Agent 端到端（LLM 自主路由）")
    sample_dir = PROJECT_ROOT / "data" / "sample_docs"
    if sample_dir.exists():
        from rag.pipeline import get_ingest_pipeline

        files = sorted(path for path in sample_dir.iterdir() if path.is_file())
        report = get_ingest_pipeline().ingest_uploads([(p.name, p.read_bytes()) for p in files])
        print(
            f"    知识库：新增 {report.succeeded}，跳过 {report.skipped}，"
            f"共 {report.store_stats.get('chunks', 0)} 块"
        )
    else:
        print("    样例文档不存在，跳过知识库相关用例")

    from core.runtime.agent import get_agent_runtime

    runtime = get_agent_runtime()
    cases = [
        ("公司的年假有几天", "knowledge_search"),
        ("帮我规划上海三日游，喜欢美食", "trip_planner"),
    ]
    for question, expected in cases:
        started = time.perf_counter()
        result = runtime.run(question)
        elapsed = round(time.perf_counter() - started, 1)
        used = result.tools_used
        hit = used[:1] == [expected]
        print(
            f"  [{'PASS' if hit else 'FAIL'}] 「{question}」\n"
            f"         路由 → {used or '（未调用工具）'}（期望 {expected}）"
            f"｜步数={result.steps}｜耗时={elapsed}s｜tokens={result.total_tokens}"
        )
        print(f"         回答：{result.answer[:100].replace(chr(10), ' ')}")
        if not hit:
            failures.append(f"LLM 路由错误：{question} → {used}")

    print("=" * 80)
    if failures:
        print(f"自检结果：{len(failures)} 项未通过")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("自检结果：全部通过 ✅")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
