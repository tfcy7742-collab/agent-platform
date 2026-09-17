"""依赖自检脚本。

用途：安装完依赖后（或在 CI 中）快速确认关键包都能正常导入，
并打印实际解析出的版本——这些版本号会写进 README 的「依赖版本」一节。

运行：
    .venv\\Scripts\\python.exe scripts\\check_deps.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# 项目根目录入 sys.path，便于直接以脚本方式运行
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# (模块名, 用途说明)
REQUIRED_MODULES = [
    ("langchain", "LangChain 主包"),
    ("langchain_core", "LangChain 核心抽象"),
    ("langchain_community", "社区加载器（PyPDFLoader/Docx2txtLoader/TextLoader）"),
    ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
    ("langchain_chroma", "Chroma 向量库集成"),
    ("langchain_openai", "OpenAI 兼容客户端（DeepSeek）"),
    ("chromadb", "向量数据库"),
    ("sentence_transformers", "本地 Embedding 模型加载"),
    ("rank_bm25", "BM25 稀疏检索"),
    ("jieba", "中文分词（BM25 用）"),
    ("fastapi", "Web 框架"),
    ("uvicorn", "ASGI 服务器"),
    ("gradio", "前端界面"),
    ("pypdf", "PDF 解析"),
    ("docx2txt", "DOCX 解析（Docx2txtLoader 依赖）"),
    ("docx", "python-docx，生成 DOCX 样例文档用"),
    ("pydantic", "数据校验"),
    ("pydantic_settings", "配置管理"),
    ("dotenv", "环境变量加载"),
    ("httpx", "HTTP 客户端（ui.py 独立启动时调 API）"),
    ("pytest", "测试框架"),
]


def main() -> int:
    """逐个导入并打印结果，返回失败个数作为退出码。"""
    failures = 0
    print("=" * 78)
    print(f"Python: {sys.version.split()[0]}   ({sys.executable})")
    print("=" * 78)
    for module_name, purpose in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "-")
            print(f"[ OK ] {module_name:26s} {str(version):16s} {purpose}")
        except Exception as exc:  # noqa: BLE001 - 自检脚本需要捕获所有异常
            failures += 1
            print(f"[FAIL] {module_name:26s} {'':16s} {type(exc).__name__}: {exc}")
    print("=" * 78)
    print(f"结果：{len(REQUIRED_MODULES) - failures}/{len(REQUIRED_MODULES)} 个模块可用")
    if failures:
        print("提示：出现 FAIL 时先看是不是 requirements.txt 未装全，"
              "或 pip 解析出的版本互相冲突。")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
