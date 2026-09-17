"""诊断 Gradio 挂载状态与上传相关路由（解释浏览器里的 404）。

用法：
    .venv\\Scripts\\python.exe scripts\\diagnose_gradio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import app as app_module  # noqa: E402

print("=" * 78)
print("Gradio 挂载诊断")
print("=" * 78)

# [1] 主 app 的路由（挂载点会以 Mount 形式出现）
print("\n[1] 主应用路由：")
for route in app_module.app.routes:
    path = getattr(route, "path", "?")
    kind = type(route).__name__
    if path.startswith("/ui") or "manifest" in path:
        print(f"    {kind:18s} {path}")

# [2] 找到 Gradio 子应用，列出它自己的路由
print("\n[2] Gradio 子应用路由（上传相关）：")
found_gradio = False
for route in app_module.app.routes:
    sub_app = getattr(route, "app", None)
    if sub_app is None or not hasattr(sub_app, "routes"):
        continue
    found_gradio = True
    for sub in sub_app.routes:
        sub_path = getattr(sub, "path", "?")
        if any(key in sub_path for key in ("upload", "queue", "file", "config", "api")):
            print(f"    {type(sub).__name__:18s} {sub_path}")

if not found_gradio:
    print("    没找到 Gradio 子应用（可能挂载失败）")

# [3] 关键结论
print("\n[3] 说明：")
print("    · 上传请求打到 /ui/gradio_api/upload，成功后才会有 upload_id；")
print("    · 浏览器报 upload_id=undefined + upload_progress 404，说明上传会话没建立起来，")
print("      最常见原因是**页面是在服务重启前加载的**（服务端状态与页面不同步）。")
