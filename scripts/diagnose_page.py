"""诊断：确认浏览器页面到底来自哪个服务。

用法：
    .venv\\Scripts\\python.exe scripts\\diagnose_page.py
"""

from __future__ import annotations

import re
import sys
import urllib.request

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 各服务可能占用的地址
TARGETS = [
    ("agent-platform 界面", "http://127.0.0.1:8000/ui/"),
    ("agent-platform 根路径", "http://127.0.0.1:8000/"),
    ("trip-planner 前端(Vite)", "http://127.0.0.1:5173/"),
    ("Gradio 独立模式", "http://127.0.0.1:7860/"),
]

# 关键字 → 归属哪个项目
MARKERS = {
    "gradio": "Agent 平台（Gradio 界面）",
    "ant-design": "trip-planner（Vue + Ant Design Vue）",
    "src/main.ts": "trip-planner（Vite 开发服务器）",
    "/assets/index-": "trip-planner（已构建产物）",
    "正在连接后端": "trip-planner 的加载占位文案",
}


def fetch(url: str, timeout: float = 10.0) -> tuple:
    """返回 (状态码, 页面片段)。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        return response.status, body
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


print("=" * 78)
print("页面归属诊断")
print("=" * 78)

for label, url in TARGETS:
    status, body = fetch(url)
    print(f"\n[{label}] {url}")
    if status is None:
        print(f"    无法访问：{body}")
        continue

    print(f"    HTTP {status}｜页面长度 {len(body)}")
    title = re.search(r"<title>(.*?)</title>", body, re.DOTALL)
    if title:
        print(f"    标题：{title.group(1).strip()[:60]}")

    hits = [name for marker, name in MARKERS.items() if marker in body.lower() or marker in body]
    if hits:
        print(f"    识别为：{'、'.join(sorted(set(hits)))}")
    else:
        print("    识别为：（未匹配到已知标记）")
