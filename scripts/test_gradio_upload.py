"""实测 Gradio 上传端点是否可用（定位浏览器里的 upload_progress 404）。

用法：
    .venv\\Scripts\\python.exe scripts\\test_gradio_upload.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = "http://127.0.0.1:8000"


def probe(path: str, method: str = "GET") -> str:
    """请求一个路径，返回状态说明。"""
    request = urllib.request.Request(BASE + path, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return f"HTTP {response.status}（{len(response.read())} 字节）"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def upload_file(path: Path) -> str:
    """向 Gradio 的上传端点发一个真实文件，返回结果摘要。"""
    boundary = "----probe" + uuid.uuid4().hex
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'.encode("utf-8")
    )
    body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
    body.extend(path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    for endpoint in ("/ui/gradio_api/upload", "/ui/upload"):
        request = urllib.request.Request(
            BASE + endpoint,
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read().decode("utf-8", errors="replace")
            return f"{endpoint} → HTTP {response.status}｜返回：{payload[:200]}"
        except urllib.error.HTTPError as exc:
            print(f"    {endpoint} → HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {endpoint} → {type(exc).__name__}: {exc}")
    return "两个上传端点都不可用"


def main() -> int:
    """执行诊断。"""
    print("=" * 78)
    print("Gradio 上传端点实测")
    print("=" * 78)

    print("\n[1] 基础端点探测：")
    for path in (
        "/ui/",
        "/ui/config",
        "/ui/gradio_api/upload",
        "/ui/gradio_api/upload_progress?upload_id=x",
        "/ui/gradio_api/queue/join",
    ):
        method = "POST" if "join" in path else "GET"
        print(f"    {path:52s} → {probe(path, method)}")

    print("\n[2] 真实文件上传：")
    sample = PROJECT_ROOT / "data" / "sample_docs" / "运维技术FAQ.md"
    if not sample.exists():
        print("    样例文档不存在")
    else:
        print(f"    {upload_file(sample)}")

    print("\n[3] 结论：")
    print("    上传能拿到文件 id 列表 → Gradio 上传通道正常，浏览器报 upload_id=undefined")
    print("    属于**页面状态与服务端不同步**（页面是旧版本加载的），强制刷新即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
