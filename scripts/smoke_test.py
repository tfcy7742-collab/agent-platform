"""最终端到端冒烟测试（真实服务 + 真实大模型）。

覆盖：健康检查 → 上传样例文档 → RAG 问答 → Agent 自主路由 → 评测接口 → 界面。
"""

from __future__ import annotations

import json
import mimetypes
import sys
import time
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


def post_json(path: str, payload: dict, timeout: float = 300.0) -> dict:
    """POST JSON。"""
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path: str, timeout: float = 120.0) -> dict:
    """GET JSON。"""
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def upload_samples() -> dict:
    """上传样例文档目录下的所有文件。"""
    files = sorted((PROJECT_ROOT / "data" / "sample_docs").iterdir())
    boundary = "----smoke" + uuid.uuid4().hex
    body = bytearray()
    for path in files:
        if not path.is_file():
            continue
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        BASE + "/api/documents",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    """执行冒烟测试。"""
    health = get_json("/health")
    print(
        f"[health] status={health['status']}｜llm_online={health['capabilities']['llm']['online']}"
        f"｜model={health['capabilities']['llm']['model']}｜chunks={health['vector_store']['chunks']}"
    )

    report = upload_samples()
    print(
        f"[upload] 新增 {report['succeeded']}｜跳过 {report['skipped']}｜"
        f"合计 {report['store']['chunks']} 块"
    )

    started = time.perf_counter()
    rag = post_json("/api/chat", {"question": "年假有几天", "mode": "rag"})
    print(
        f"[chat rag] {time.perf_counter() - started:.1f}s｜refused={rag['refused']}"
        f"｜来源 {len(rag['sources'])} 条｜tokens={rag['usage']['total_tokens']}"
    )
    print(f"    回答：{rag['answer'][:120].replace(chr(10), ' ')}")

    started = time.perf_counter()
    agent = post_json("/api/chat", {"question": "帮我规划杭州两日游，喜欢自然风光", "mode": "agent"})
    print(
        f"[chat agent] {time.perf_counter() - started:.1f}s｜路由={agent['tools_used']}"
        f"｜步数={agent['steps']}｜tokens={agent['usage']['total_tokens']}"
    )
    print(f"    回答：{agent['answer'][:120].replace(chr(10), ' ')}")

    datasets = get_json("/api/eval/datasets")
    print(
        f"[eval] 问答集 {datasets['qa']['total']} 条"
        f"（可答 {datasets['qa']['answerable']}／应拒答 {datasets['qa']['unanswerable']}）"
        f"｜路由集 {datasets['routing']['total']} 条"
    )
    reports = get_json("/api/eval/reports")
    print(f"[reports] 历史报告 {reports['total']} 份")
    if reports["reports"]:
        latest = reports["reports"][0]
        metrics = latest["metrics"]
        print(
            f"    最新：{latest['config']}｜Recall={metrics['recall']}"
            f"｜拒答={metrics['refusal_accuracy']}｜路由={metrics['route_accuracy']}"
        )

    status = urllib.request.urlopen(BASE + "/ui/", timeout=60).status
    print(f"[ui] /ui/ -> HTTP {status}")
    print("\n冒烟测试完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
