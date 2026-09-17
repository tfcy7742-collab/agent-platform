"""探测可用的模型下载源。

背景：部分网络环境无法访问 huggingface.co。本脚本探测几个候选源，
列出 ``BAAI/bge-small-zh-v1.5`` 的文件清单，供 scripts/download_embedding_model.py
选择可用源。

用法：
    .venv\\Scripts\\python.exe scripts/probe_model_sources.py
"""

from __future__ import annotations

import json
import sys
import urllib.request

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

MODEL_ID = "BAAI/bge-small-zh-v1.5"

SOURCES = [
    ("HuggingFace 官方", f"https://huggingface.co/api/models/{MODEL_ID}"),
    ("HF 镜像 hf-mirror", f"https://hf-mirror.com/api/models/{MODEL_ID}"),
    ("ModelScope", f"https://www.modelscope.cn/api/v1/models/{MODEL_ID}"),
]


def fetch(url: str, timeout: int = 20) -> tuple[bool, str]:
    """请求 URL，返回 (是否成功, 说明或内容)。"""
    request = urllib.request.Request(url, headers={"User-Agent": "agent-platform-probe/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    """逐个探测下载源。"""
    print("=" * 78)
    print(f"探测模型 {MODEL_ID} 的可用下载源")
    print("=" * 78)

    working: list[str] = []
    for name, url in SOURCES:
        ok, payload = fetch(url)
        if not ok:
            print(f"[FAIL] {name:20s} {payload[:100]}")
            continue
        print(f"[ OK ] {name:20s} 可达")

        # ModelScope 返回 {"Data": {"Files": [...]}} 结构
        if "modelscope" in url:
            try:
                data = json.loads(payload)
                files = data.get("Data", {}).get("Files", [])
                names = [item.get("Path") or item.get("Name") for item in files]
                print(f"        文件（{len(names)} 个）：{', '.join(str(n) for n in names[:15])}")
                if any(str(n).endswith("model.safetensors") or str(n).endswith("pytorch_model.bin") for n in names):
                    working.append("modelscope")
            except (json.JSONDecodeError, AttributeError) as exc:
                print(f"        解析文件清单失败：{exc}")
        else:
            # HuggingFace 兼容 API 返回 siblings
            try:
                data = json.loads(payload)
                names = [item.get("rfilename") for item in data.get("siblings", [])]
                print(f"        文件（{len(names)} 个）：{', '.join(str(n) for n in names[:15])}")
                if any(str(n).endswith("model.safetensors") or str(n).endswith("pytorch_model.bin") for n in names):
                    working.append("hf_mirror" if "mirror" in url else "huggingface")
            except (json.JSONDecodeError, AttributeError) as exc:
                print(f"        解析文件清单失败：{exc}")

    print("-" * 78)
    if working:
        print(f"可用源：{working}")
        print("下一步：设置对应环境变量后重新运行 scripts/download_embedding_model.py")
        print("  · ModelScope 源需要 modelscope 包：pip install modelscope")
        print("  · HF 镜像源可设置：$env:HF_ENDPOINT='https://hf-mirror.com'")
    else:
        print("没有可用源：请检查网络/代理，或手动下载模型文件放入 data/models/")
    print("=" * 78)
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
