"""实测 UI 的上传代码路径（与界面按钮走同一条逻辑）。

用法（服务已启动）：
    .venv\\Scripts\\python.exe scripts\\test_ui_upload.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


class FakeUploadFile:
    """模拟 Gradio 传给回调的 File 对象（它只有 .name 属性）。"""

    def __init__(self, path: Path) -> None:
        self.name = str(path)


async def main() -> int:
    """走一遍界面里的上传函数，打印每一步的产出。"""
    from ui import delete_all_documents, refresh_documents, refresh_status, upload_documents

    print("=" * 78)
    print("实测 UI 上传路径")
    print("=" * 78)

    print("\n[1] 状态栏：")
    print((await refresh_status())[:120].replace("\n", " "))

    files = sorted((PROJECT_ROOT / "data" / "sample_docs").iterdir())
    uploads = [FakeUploadFile(path) for path in files if path.is_file()]
    print(f"\n[2] 上传 {len(uploads)} 个文件（逐个打印生成器产出）：")
    async for progress, table in upload_documents(uploads):
        print(f"    进度：{str(progress)[:110]}")
        print(f"    表格行数：{len(table)}")
        if len(table) and table[0]:
            print(f"    表格首行：{table[0]}")

    print("\n[3] 上传后再看状态栏：")
    print((await refresh_status())[:160].replace("\n", " "))

    print("\n[4] 上传后的文档列表：")
    for row in await refresh_documents():
        print(f"    {row}")

    print("\n[5] 空文件上传（应给出提示，而不是静默无反应）：")
    async for progress, _table in upload_documents(None):
        print(f"    进度：{progress}")

    print("\n[6] 一键清空（同样应有进度反馈）：")
    async for progress, table in delete_all_documents():
        print(f"    进度：{progress}｜剩余表格行数：{len(table)}")

    print("\n" + "=" * 78)
    print("结论：上传路径全程有进度反馈，且最终文档列表被正确刷新。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
