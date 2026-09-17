"""把 trip-planner 项目的后端代码迁移为 agent-platform 的 trip_planner 子包。

迁移内容
--------
从 ``../trip-planner/backend`` 复制以下模块到 ``agent-platform/trip_planner/``：

* ``config.py``            —— 子包自己的配置（LLM 连接与工具轮数）
* ``models/``              —— Pydantic 数据模型（TripRequest / TripPlan 等）
* ``tools/``               —— search_attractions / get_weather / search_hotels（模拟数据）
* ``agents/``              —— 4 个专用智能体 + 本地规则降级生成器
* ``coordinator.py``       —— MultiAgentTripPlanner（协调者）

不迁移：``app.py``（本项目已有自己的 FastAPI 入口）、``tests/``（避免被本项目的
pytest 收集到而干扰测试集）。

导入改写
--------
原项目的包根是 ``backend``，所有跨子包引用写作 ``from ..models.schemas import ...``。
迁移后包根变成 ``trip_planner``，子包之间的 ``..`` 语义发生变化，因此统一改写成
**绝对导入** ``from trip_planner.models.schemas import ...``：
绝对导入不受包层级变化影响，也不会和本项目的 ``core`` / ``api`` 等顶层包撞名。

用法：
    .venv\\Scripts\\python.exe scripts\\migrate_trip_planner.py [--source <路径>]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

DEFAULT_SOURCE = PROJECT_ROOT.parent / "trip-planner" / "backend"
TARGET = PROJECT_ROOT / "trip_planner"

# 需要迁移的文件/目录（相对 source）
ITEMS = ["config.py", "coordinator.py", "models", "tools", "agents"]

# 导入改写规则：包根从 backend 变成 trip_planner
#   from ..models.schemas import X   →  from trip_planner.models.schemas import X
#   from ..config import X           →  from trip_planner.config import X
#   from ..tools.travel_tools import X → from trip_planner.tools.travel_tools import X
REWRITES: list[tuple[str, str]] = [
    (r"from \.\.models", "from trip_planner.models"),
    (r"from \.\.tools", "from trip_planner.tools"),
    (r"from \.\.config", "from trip_planner.config"),
    (r"from \.\.agents", "from trip_planner.agents"),
    (r"import \.\.models", "import trip_planner.models"),
]


def rewrite_imports(path: Path) -> int:
    """改写单个文件的导入语句，返回改动行数。"""
    original = path.read_text(encoding="utf-8")
    text = original
    changes = 0
    for pattern, replacement in REWRITES:
        new_text, count = re.subn(pattern, replacement, text)
        if count:
            changes += count
            text = new_text
    if text != original:
        path.write_text(text, encoding="utf-8")
    return changes


def main() -> int:
    """执行迁移。"""
    parser = argparse.ArgumentParser(description="迁移 trip-planner 后端代码")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="源 backend 目录")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        print(f"[FAIL] 源目录不存在：{source}")
        return 1

    print("=" * 78)
    print("迁移 trip-planner → agent-platform/trip_planner")
    print("=" * 78)
    print(f"源目录  ：{source}")
    print(f"目标目录：{TARGET}")

    # 清理旧目录（幂等：重复执行不会留下上一版的残留文件）
    if TARGET.exists():
        shutil.rmtree(TARGET, ignore_errors=True)
        print("已清理旧的 trip_planner 目录")
    TARGET.mkdir(parents=True, exist_ok=True)

    copied_files: list[Path] = []
    for item in ITEMS:
        src = source / item
        dst = TARGET / item
        if not src.exists():
            print(f"  [SKIP] 源中不存在：{item}")
            continue
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "test_*.py"),
            )
        else:
            shutil.copy2(src, dst)
        print(f"  [COPY] {item}")

    # 移除复制过来的 tests（如果 copytree 的 ignore 未覆盖）
    for stale in TARGET.rglob("test_*.py"):
        stale.unlink()
        print(f"  [DEL ] {stale.relative_to(TARGET)}")

    # 改写导入
    total_changes = 0
    for path in sorted(TARGET.rglob("*.py")):
        copied_files.append(path)
        total_changes += rewrite_imports(path)
    print(f"\n导入改写：{total_changes} 处（相对导入 → 绝对导入 trip_planner.*）")

    # 生成子包 __init__.py（暴露最常用的入口）
    (TARGET / "__init__.py").write_text(
        '"""旅行规划子系统（从 trip-planner 项目迁移而来）。\n'
        "\n"
        "对外入口：\n"
        "    from trip_planner import MultiAgentTripPlanner, TripRequest\n"
        "\n"
        "它在本平台中作为 ``trip_planner`` 工具被 Agent 调用：\n"
        "用户说「帮我规划北京三日游」时，Planner 会路由到这个子系统，\n"
        "由内部 4 个智能体（景点/天气/酒店/行程规划）协作产出完整行程。\n"
        '"""\n'
        "\n"
        "from .coordinator import MultiAgentTripPlanner\n"
        "from .models.schemas import TripPlan, TripRequest\n"
        "\n"
        '__all__ = ["MultiAgentTripPlanner", "TripPlan", "TripRequest"]\n',
        encoding="utf-8",
    )
    print(f"\n共迁移 {len(copied_files)} 个 Python 文件")
    print("=" * 78)
    print("下一步：运行 python -c \"import trip_planner; print('ok')\" 验证导入")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
