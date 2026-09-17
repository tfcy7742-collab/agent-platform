"""实测通道一（结构化行程表单）：参数透传 + 富结构渲染。

用法（服务已启动）：
    .venv\\Scripts\\python.exe scripts\\test_trip_form.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 允许指向别的端口：本机可能已有旧实例占用 8000，用 8001 起新实例做验证
API_BASE = os.getenv("AGENT_PLATFORM_API")
if API_BASE:
    import ui

    ui.API_BASE = API_BASE
    print(f"（本次指向 {API_BASE}）")


async def main() -> int:
    """按表单方式生成一次行程，检查参数与渲染完整性。"""
    from ui import generate_trip_plan, render_trip_plan

    print("=" * 90)
    print("通道一实测：结构化表单 → 行程规划工具 → 富结构渲染")
    print("=" * 90)

    # 模拟用户填表：成都 / 4 天 / 3 人 / 15000 元 / 豪华 / 美食+摄影
    args = ("成都", "2026-10-01", 4, 3, 15000, "豪华", ["美食", "摄影"], "带两位老人，节奏慢一点")

    rendered = ""
    async for progress, output in generate_trip_plan(*args):
        print(f"\n[进度] {progress}")
        if output:
            rendered = output

    if not rendered:
        print("\n❌ 没有拿到渲染结果")
        return 1

    print("\n" + "=" * 90)
    print("渲染结果检查（前 3500 字）")
    print("=" * 90)
    print(rendered[:3500])

    print("\n" + "=" * 90)
    print("完整性校验")
    print("=" * 90)
    checks = [
        ("包含逐日标题", "### 📅 逐日行程" in rendered),
        ("包含 4 天的卡片", rendered.count("#### 第 ") >= 4),
        ("每天都有上午", rendered.count("- **上午**：") >= 4),
        ("每天都有下午", rendered.count("- **下午**：") >= 4),
        ("每天都有晚间", rendered.count("- **晚间**：") >= 4),
        ("包含预算明细表", "### 📊 预算明细" in rendered),
        ("预算表含分项", "住宿" in rendered and "餐饮" in rendered),
        ("包含预算结论", "### 💰 预算结论" in rendered),
        ("包含景点折叠面板", "🎫 推荐景点" in rendered),
        ("包含酒店折叠面板", "🏨 推荐酒店" in rendered),
        ("包含天气表格", "🌤 逐日天气" in rendered),
        ("包含注意事项", "### ⚠️ 注意事项" in rendered),
        ("包含子智能体轨迹", "子智能体协作轨迹" in rendered),
        ("参数按填写值生效（3 人）", "3 人" in rendered),
        ("参数按填写值生效（15000 元）", "15000" in rendered),
        ("档位生效（豪华）", "豪华" in rendered),
    ]
    failed = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1

    print("\n" + "=" * 90)
    print(f"结果：{len(checks) - failed}/{len(checks)} 项通过｜渲染长度 {len(rendered)} 字符")
    print("=" * 90)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
