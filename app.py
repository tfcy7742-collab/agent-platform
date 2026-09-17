"""FastAPI 服务入口。

B1（本批）实现：
* 应用装配与 lifespan（建目录、初始化 SQLite、打印脱敏配置快照）；
* ``GET /``、``GET /health``、``GET /api/metrics``、``GET /api/config``；
* 统一的异常响应格式（前端永远拿到 ``{ok, error_type, detail}`` 结构）。

后续批次追加（接口契约已在架构文档中定死）：
* B2 ``POST/GET/DELETE /api/documents``
* B3+B4 ``POST /api/chat``（SSE）、``/api/tools``、``/api/traces``
* B5 Gradio 挂载到 ``UI_MOUNT_PATH``
* B6 ``/api/eval/*``

启动：
    python app.py
    # 或
    uvicorn app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import ensure_directories, get_settings
from infra import db, metrics
from infra.logging_setup import log_config_snapshot, setup_logging
from api import chat as chat_api
from api import chat_stream as chat_stream_api
from api import documents as documents_api
from api import evaluation as evaluation_api
from api import tools as tools_api

# 服务启动时刻，用于计算运行时长
_STARTED_AT = time.time()

logger = logging.getLogger("agent-platform")


# ---------------------------------------------------------------------------
# 生命周期：启动时做一次性的初始化，关闭时释放资源
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用启动/关闭钩子。"""
    settings = get_settings()
    setup_logging(settings, force=True)
    ensure_directories()          # 上传目录 / Chroma 目录 / 模型缓存 / 数据库目录
    db.init_db()                  # 幂等建表
    log_config_snapshot(settings)  # 脱敏配置快照（出问题时的第一条线索）

    # 注册平台内置工具（knowledge_search / trip_planner / send_email）
    from core.tools.registry import bootstrap_tools

    registry = bootstrap_tools()
    logger.info("可用工具：%s", "、".join(registry.names()) or "（无）")

    logger.info(
        "服务启动完成：http://%s:%s  （UI 挂载路径 %s）",
        settings.host, settings.port, settings.ui_mount_path,
    )
    if settings.llm_online:
        logger.info("大模型：%s @ %s", settings.provider_model, settings.provider_base_url)
    yield
    logger.info("服务关闭，释放数据库连接")
    db.close_connection()


settings = get_settings()

app = FastAPI(
    title="Agent 平台",
    description=(
        "可插拔工具协议的企业级 Agent 系统：统一 Tool 接口接入 RAG 知识检索与多智能体任务编排，"
        "由 Planner 自主路由；内置流式输出、超时熔断、失败降级、人工确认、全链路 trace 与离线评测。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# 跨域：允许本地前端调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 业务路由（按资源拆分，便于随批次扩展）
app.include_router(documents_api.router)
app.include_router(chat_api.router)
app.include_router(chat_stream_api.router)
app.include_router(tools_api.router)
app.include_router(evaluation_api.router)


# ---------------------------------------------------------------------------
# 挂载 Gradio 界面（B5）
# ---------------------------------------------------------------------------
def mount_ui(application: FastAPI) -> bool:
    """把 Gradio 界面挂到 ``UI_MOUNT_PATH``（默认 /ui）。

    为什么挂载而不是独立跑 7860：
    * 单进程单端口，部署与演示都只需暴露一个端口；
    * 不需要 CORS，也不会有"前端连不上后端"这类环境问题；
    * ``ui.py`` 仍然保留了独立启动的能力（``python ui.py`` 走 7860），
      两种情况共用同一份界面代码。

    为什么在**模块导入时**调用（而不是放进 lifespan 启动钩子）：
    FastAPI / Starlette 的约定是"路由在应用开始服务之前注册完毕"，
    挂载本质上也是注册路由。放在 lifespan 里虽然通常也能跑，但属于
    "服务已启动后再改路由表"，容易在热重载或某些启动顺序下出现
    界面路由与前端状态不一致（浏览器表现为上传拿不到 upload_id、
    ``upload_progress`` 返回 404）。挪到导入期即可消除这类隐患。
    注意这里只构建 Gradio 应用对象，不会加载模型或建索引，开销很小。

    Returns:
        是否挂载成功（Gradio 导入失败时不影响 API 服务本身）。
    """
    try:
        import gradio as gr

        from ui import build_demo

        demo = build_demo()
        # queue() 让流式生成器可以逐块推送；default_concurrency_limit 支持多人同时用
        gr.mount_gradio_app(
            application,
            demo.queue(default_concurrency_limit=8),
            path=settings.ui_mount_path,
        )
        logger.info("Gradio 界面已挂载：%s", settings.ui_mount_path)
        return True
    except Exception as exc:  # noqa: BLE001 - UI 挂载失败不应影响 API
        logger.error("挂载 Gradio 界面失败（API 仍可用）：%s", exc)
        return False


# 导入期完成挂载：保证 /ui 相关路由在应用开始服务前就绪
mount_ui(app)


# ---------------------------------------------------------------------------
# 统一异常处理：任何错误都返回结构化 JSON，前端不会白屏
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """参数校验失败 → 422，并给出人类可读的中文提示。"""
    problems = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item.get("loc", []) if part != "body")
        problems.append(f"{location or 'body'}: {item.get('msg', '参数不合法')}")
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error_type": "invalid_request",
            "detail": "；".join(problems),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理：记录堆栈，但不把内部细节暴露给前端。"""
    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error_type": type(exc).__name__,
            "detail": "服务内部错误，请查看服务端日志（trace 中已记录）",
            "path": str(request.url.path),
        },
    )


# ---------------------------------------------------------------------------
# 基础接口
# ---------------------------------------------------------------------------
@app.get("/manifest.json", include_in_schema=False)
async def web_manifest() -> Dict[str, Any]:
    """返回极简的 Web App Manifest。

    浏览器加载页面时会自动向**站点根路径**请求 ``/manifest.json``（PWA 清单），
    而 Gradio 只在挂载路径 ``/ui`` 下提供它，所以服务日志里总会出现一条
    ``GET /manifest.json 404``。这条日志与功能无关，但容易让人误判成错误，
    这里补一个空清单把它消掉。
    """
    return {
        "name": "Agent 平台",
        "short_name": "Agent",
        "start_url": settings.ui_mount_path,
        "display": "standalone",
        "background_color": "#f0f2f5",
        "theme_color": "#1677ff",
    }


@app.get("/", summary="服务信息", tags=["基础"])
async def root() -> Dict[str, Any]:
    """返回服务基本信息与接口导航（便于用浏览器直接确认服务是否活着）。"""
    return {
        "ok": True,
        "service": "Agent 平台（可插拔工具协议）",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ui": settings.ui_mount_path,
        "endpoints": {
            "health": "GET /health",
            "metrics": "GET /api/metrics",
            "config": "GET /api/config",
            "documents": "GET/POST/DELETE /api/documents（B2 已交付）",
            "chat": "POST /api/chat（mode=agent|rag，B4 已支持 Agent 自主路由）",
            "search": "GET /api/search?q=...（B3 已交付，仅检索不生成）",
            "tools": "GET /api/tools（B4 已交付）",
            "traces": "GET /api/traces（B4 已交付）",
            "eval": "POST /api/eval/run（B6 交付）",
        },
    }


@app.get("/health", summary="健康检查", tags=["基础"])
async def health() -> Dict[str, Any]:
    """健康检查：服务状态 + 各项能力（LLM / Embedding / 检索 / 存储）的可用性与降级情况。

    这是「降级必须可见」原则的落点：外部系统与前端都通过它判断当前能力边界。
    """
    capabilities = settings.capabilities()

    # 数据统计失败不影响健康检查本身
    try:
        doc_stats = db.document_stats()
        trace_stats = db.trace_stats()
        db_ok = True
    except Exception as exc:  # pragma: no cover - 极端环境
        logger.warning("读取存储统计失败：%s", exc)
        doc_stats = {"documents": 0, "chunks": 0, "chars": 0}
        trace_stats = {}
        db_ok = False

    # 向量库与 Embedding 的真实状态（B2 起提供；延迟创建，避免启动即加载模型）
    try:
        from rag.store import get_vector_store

        vector_stats = get_vector_store().stats()
    except Exception as exc:  # pragma: no cover
        logger.warning("读取向量库统计失败：%s", exc)
        vector_stats = {"backend": "chromadb", "chunks": 0, "error": str(exc)}

    return {
        "status": "ok" if db_ok else "degraded",
        "uptime_s": round(time.time() - _STARTED_AT, 1),
        "capabilities": capabilities,
        "storage": {"database_ok": db_ok, "documents": doc_stats, "traces": trace_stats},
        "vector_store": vector_stats,
        "metrics": metrics.snapshot(),
    }


@app.get("/api/metrics", summary="运行指标", tags=["观测"])
async def get_metrics() -> Dict[str, Any]:
    """进程内累计指标：请求数、拒答数、工具耗时、延迟分位数、token 与成本。"""
    return {"ok": True, "metrics": metrics.snapshot(), "trace_stats": db.trace_stats()}


@app.get("/api/config", summary="当前配置（脱敏）", tags=["基础"])
async def get_config() -> Dict[str, Any]:
    """返回脱敏后的当前配置，便于排查"到底加载了哪份 .env"。"""
    return {"ok": True, "config": settings.masked_snapshot(), "capabilities": settings.capabilities()}


# ---------------------------------------------------------------------------
# 本地启动入口
# ---------------------------------------------------------------------------
def _port_owner(port: int) -> Optional[int]:
    """查询占用指定端口的进程 id（查不到返回 None）。

    仅用于把"启动失败"翻译成一句人话。跨平台实现依赖 ``psutil``，
    没装则返回 None，不影响主流程。
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                return conn.pid
    except Exception:  # noqa: BLE001 - 权限不足或平台差异，忽略
        return None
    return None


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    # 端口被占用时，uvicorn 只会抛出难懂的 winerror 10048 / EADDRINUSE。
    # 这里先探测一次，给出"哪个进程占着、怎么停掉"的明确指引。
    owner = _port_owner(settings.port)
    if owner is not None:
        print("=" * 78)
        print(f"启动失败：端口 {settings.port} 已被占用（进程 id {owner}）")
        print("这说明**已经有一个服务在运行了**，你不需要再启动一个。")
        print()
        print(f"直接使用现有服务： http://{settings.host}:{settings.port}/ui")
        print("如果确实要重启，先停掉旧进程（在下面这条命令里执行）：")
        print(f"    Stop-Process -Id {owner} -Force")
        print("或者回到正在运行的那个 PowerShell 窗口按 Ctrl+C。")
        print("=" * 78)
        raise SystemExit(1)

    # 以脚本方式运行时（python app.py），reload 必须关闭：
    # reload 依赖 import string，且在 Windows 下会重复执行启动逻辑。
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
