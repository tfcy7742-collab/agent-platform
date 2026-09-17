"""Gradio 前端界面（B5 / B6）。

形态
----
**挂载在 FastAPI 的 /ui 路径下**（单进程单端口），也可以独立启动：

    # 方式一（推荐）：随服务一起启动
    python app.py            # 打开 http://127.0.0.1:8000/ui

    # 方式二：独立进程（此时它会通过 HTTP 调用 FastAPI，需要服务已在跑）
    python ui.py             # 打开 http://127.0.0.1:7860

界面组成（两条输入通道）
------------------------
::

    ┌──────────────────────── 顶部状态栏（模型 / 工具 / 向量库块数）────────────────────────┐
    ├──────────────┬──────────────────────────────────┬────────────────────────────────────┤
    │ 知识库管理     │ 【通道一】行程规划表单             │ 【通道二】对话区                    │
    │ · 上传文档     │ · 目的地/日期/天数/人数/预算/偏好   │ · 聊天气泡 + 流式进度               │
    │ · 文档列表     │ · 生成后按结构渲染：逐日卡片 +      │ · 每条回答下：来源折叠 / 执行轨迹   │
    │ · 一键清空     │   景点/酒店列表 + 预算明细表        │ · 危险操作：确认 / 拒绝             │
    └──────────────┴──────────────────────────────────┴────────────────────────────────────┘

**为什么要两条通道**：旅行行程是高度结构化的长产物（逐日 × 三个时段 × 景点/门票/酒店/预算表），
把它塞进"模型读一遍再转述成一段话"的对话链路里天然会丢信息、也无法逐项调整参数。
所以结构化表单**直接调用工具**、结果**按结构渲染**；对话通道则保留"随便说一句就能规划"的便利。
这两条路径调用的是同一个 `trip_planner` 工具，只是输入输出形态不同。

实现要点
--------
* 全部通过 **HTTP 调用 FastAPI**（不直接 import 业务代码）：这样"UI 与 API 分离"
  的边界是真实的，独立启动与挂载运行的行为完全一致；
* 用 ``httpx.AsyncClient.stream`` 逐行解析 SSE，边收边 ``yield``，
  因此打字机效果与轨迹面板的滚动都是**真流式**，不是假动画；
* 对话历史保存在浏览器会话（``gr.State``）里，刷新即重置——
  演示场景不需要持久化，真要持久化由后端的 ``sessions`` 表承担。
"""

from __future__ import annotations

import json
import logging
import os
from functools import partial
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

# 后端地址（独立启动时可通过环境变量指向远程服务）
API_BASE = os.getenv("AGENT_PLATFORM_API", "http://127.0.0.1:8000")
# 流式请求超时（秒）：首字节 + 每步执行都可能较慢
STREAM_TIMEOUT = 300.0
# 支持上传的文档后缀
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]

# 事件类型 → 中文标签（轨迹面板展示用）
EVENT_LABELS: Dict[str, str] = {
    "plan": "🧠 决策",
    "rewrite": "✏️ 问题改写",
    "tool_start": "🔧 调用工具",
    "tool_end": "✅ 工具完成",
    "observation": "📄 观察",
    "refuse": "🚫 拒答",
    "pending_confirmation": "⚠️ 待确认",
    "token": "💬 生成中",
    "final": "🏁 最终回答",
    "error": "❌ 错误",
    "done": "◼ 结束",
}


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------
def _client(timeout: float = 60.0) -> httpx.AsyncClient:
    """创建异步 HTTP 客户端。"""
    return httpx.AsyncClient(base_url=API_BASE, timeout=timeout)


async def _safe_get(path: str, default: Any = None) -> Any:
    """GET 请求（失败返回默认值，保证 UI 不会因为后端没起来就崩）。"""
    try:
        async with _client() as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.json()
    except Exception as exc:  # noqa: BLE001 - UI 层必须容错
        logger.warning("请求 %s 失败：%s", path, exc)
        return default


# ---------------------------------------------------------------------------
# 顶部状态栏 / 文档管理
# ---------------------------------------------------------------------------
async def refresh_status() -> str:
    """渲染顶部状态栏（模型、工具、知识库规模）。"""
    health = await _safe_get("/health")
    if not health:
        return (
            "### ⚠️ 后端未连接\n"
            f"请先启动服务：`python app.py`（当前尝试连接 `{API_BASE}`）"
        )

    capabilities = health.get("capabilities", {})
    llm = capabilities.get("llm", {})
    embedding = capabilities.get("embedding", {})
    retrieval = capabilities.get("retrieval", {})
    store = health.get("vector_store", {})
    documents = health.get("storage", {}).get("documents", {})

    tools_body = await _safe_get("/api/tools", {})
    tool_names = "、".join(item["name"] for item in tools_body.get("tools", [])) or "（无）"

    llm_state = "🟢 已连接 DeepSeek" if llm.get("online") else "🟡 离线模式（规则路由 + 片段摘取）"
    return (
        f"### Agent 平台　{llm_state}\n"
        f"模型 `{llm.get('model')}`｜Embedding `{embedding.get('backend')}`"
        f"（`{embedding.get('model')}`）｜拒答阈值 `{retrieval.get('refuse_threshold')}`\n\n"
        f"可用工具：`{tool_names}`\n\n"
        f"知识库：**{documents.get('documents', 0)}** 份文档 / "
        f"**{store.get('chunks', 0)}** 个向量块"
    )


async def upload_documents(files: Optional[List[Any]]) -> AsyncIterator[Tuple[Any, ...]]:
    """上传文档并**边处理边反馈**进度。

    为什么改成异步生成器：上传要做三件慢事——首次加载向量模型（十几秒）、
    逐块向量化、写入向量库。如果做成一次性返回（原来的写法），界面上会长时间
    **毫无反馈**，看起来就像"点了没反应"；这条是实际踩到的体验问题。

    Yields:
        ``(进度说明, 文档列表)``，每完成一个阶段更新一次。
    """
    if not files:
        yield "请先选择文件（支持 PDF / DOCX / TXT / Markdown）", await refresh_documents()
        return

    file_list = files if isinstance(files, list) else [files]
    names = [os.path.basename(getattr(item, "name", str(item))) for item in file_list]
    yield f"⏳ 正在准备上传 {len(file_list)} 个文件：{'、'.join(names)}", await refresh_documents()

    # 首次上传会加载向量模型，这一步最慢，先明确告诉用户，避免误以为卡死
    health = await _safe_get("/health", {})
    embedding = (health or {}).get("capabilities", {}).get("embedding", {})
    if embedding.get("backend") == "auto":
        yield (
            "⏳ 首次上传需要加载向量模型（约十几秒，仅第一次），请稍候……",
            await refresh_documents(),
        )
    else:
        yield "⏳ 正在解析、切块并向量化……", await refresh_documents()

    multipart = []
    for item in file_list:
        path = getattr(item, "name", str(item))
        try:
            with open(path, "rb") as handle:
                multipart.append(
                    ("files", (os.path.basename(path), handle.read(), "application/octet-stream"))
                )
        except OSError as exc:
            logger.warning("读取上传文件失败：%s", exc)

    if not multipart:
        yield "❌ 没有可读取的文件（可能文件已被移动或删除）", await refresh_documents()
        return

    try:
        async with _client(timeout=600.0) as client:
            response = await client.post("/api/documents", files=multipart, data={"force": "false"})
            response.raise_for_status()
            report = response.json()
    except Exception as exc:  # noqa: BLE001
        yield f"❌ 上传失败：{exc}", await refresh_documents()
        return

    lines = [f"**{report.get('message', '')}**"]
    for item in report.get("results", []):
        if not item.get("ok"):
            lines.append(f"- ❌ `{item['file_name']}`：{item.get('error')}")
        elif item.get("skipped"):
            lines.append(f"- ⏭ `{item['file_name']}`：内容未变化，已跳过")
        else:
            page = f"／{item['page_count']} 页" if item.get("page_count") else ""
            lines.append(
                f"- ✅ `{item['file_name']}`：{item['chunk_count']} 块／{item['char_count']} 字{page}"
            )
    yield "\n".join(lines), await refresh_documents()


async def refresh_documents() -> Any:
    """刷新文档列表（返回 Gradio Dataframe 数据）。"""
    body = await _safe_get("/api/documents", {})
    rows = []
    for item in body.get("documents", []):
        rows.append(
            [
                item.get("file_name", ""),
                item.get("chunk_count", 0),
                item.get("char_count", 0),
                item.get("page_count") or "-",
                item.get("created_at", ""),
            ]
        )
    return rows


async def delete_all_documents() -> AsyncIterator[Tuple[Any, ...]]:
    """删除所有文档（演示用的一键清空），同样带进度反馈。"""
    body = await _safe_get("/api/documents", {})
    documents = body.get("documents", [])
    if not documents:
        yield "知识库本来就是空的", await refresh_documents()
        return

    yield f"⏳ 正在删除 {len(documents)} 份文档……", await refresh_documents()

    deleted = 0
    for item in documents:
        try:
            async with _client() as client:
                response = await client.delete(f"/api/documents/{item['doc_id']}")
                if response.status_code == 200:
                    deleted += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除文档失败：%s", exc)

    yield f"✅ 已删除 {deleted} 份文档", await refresh_documents()


# ---------------------------------------------------------------------------
# SSE 解析
# ---------------------------------------------------------------------------
async def stream_events(path: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    """调用流式接口并逐条产出事件。

    SSE 格式为 ``event: <type>\\ndata: <json>\\n\\n``，
    这里按空行分组、按行解析；注释行（``: ping``）直接跳过。
    """
    event_type = ""
    data_buffer: List[str] = []

    async with _client(timeout=STREAM_TIMEOUT) as client:
        async with client.stream("POST", path, json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                try:
                    detail = json.loads(body.decode("utf-8")).get("detail")
                except Exception:  # noqa: BLE001
                    detail = body.decode("utf-8", errors="replace")[:200]
                yield {"type": "error", "payload": {"message": f"HTTP {response.status_code}：{detail}"}}
                return

            async for line in response.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    # 空行 = 一条事件结束
                    if event_type and data_buffer:
                        try:
                            parsed = json.loads("".join(data_buffer))
                        except json.JSONDecodeError:
                            parsed = {"raw": "".join(data_buffer)}
                        yield {"type": event_type, **parsed}
                    event_type, data_buffer = "", []
                    continue
                if line.startswith(":"):
                    continue                      # 心跳注释
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_buffer.append(line[5:].strip())

    if event_type and data_buffer:                # 处理没有以空行结尾的最后一帧
        try:
            yield {"type": event_type, **json.loads("".join(data_buffer))}
        except json.JSONDecodeError:
            pass


# ---------------------------------------------------------------------------
# 轨迹面板渲染
# ---------------------------------------------------------------------------
def render_trace_line(event: Dict[str, Any], index: int) -> str:
    """把一条事件渲染成轨迹面板上的一行 Markdown。"""
    event_type = str(event.get("type", ""))
    payload = event.get("payload") or {}
    label = EVENT_LABELS.get(event_type, event_type)

    if event_type == "plan":
        detail = f"`{payload.get('action')}`"
        if payload.get("tool"):
            detail += f" → `{payload['tool']}`"
        thought = str(payload.get("thought") or "")
        return f"**{index}. {label}**　{detail}　<sub>{thought[:80]}</sub>"

    if event_type == "tool_end":
        status = "成功" if payload.get("ok") else f"失败（{payload.get('error_type')}）"
        extra = "｜已重试" if payload.get("retried") else ""
        degraded = "｜降级" if payload.get("degraded") else ""
        return (
            f"**{index}. {label}**　`{payload.get('tool')}`　{status}"
            f"｜{payload.get('latency_ms', 0)}ms{extra}{degraded}"
        )

    if event_type == "tool_start":
        args = json.dumps(payload.get("args") or {}, ensure_ascii=False)
        return f"**{index}. {label}**　`{payload.get('tool')}`　<sub>{args[:100]}</sub>"

    if event_type == "rewrite":
        return f"**{index}. {label}**　「{payload.get('original')}」→「{payload.get('rewritten')}」"

    if event_type == "refuse":
        return f"**{index}. {label}**　原因：`{payload.get('reason')}`"

    if event_type == "observation":
        return f"**{index}. {label}**　<sub>{str(payload.get('summary') or '')[:100]}</sub>"

    if event_type == "pending_confirmation":
        return (
            f"**{index}. {label}**　准备执行 `{payload.get('tool')}`，"
            f"参数 <sub>{json.dumps(payload.get('args') or {}, ensure_ascii=False)[:120]}</sub>"
        )

    if event_type == "error":
        return f"**{index}. {label}**　{payload.get('message')}"

    if event_type == "done":
        return f"**{index}. {label}**　状态：`{payload.get('status')}`"

    return f"**{index}. {label}**"


# ---------------------------------------------------------------------------
# 对话（核心）
# ---------------------------------------------------------------------------
def render_sources(sources: List[Dict[str, Any]]) -> str:
    """把来源片段渲染成折叠面板里的 Markdown。"""
    if not sources:
        return ""
    lines: List[str] = []
    for index, source in enumerate(sources, start=1):
        page = f"　第 {source['page']} 页" if source.get("page") else ""
        lines.append(
            f"**【{index}】`{source.get('file_name')}`{page}**　"
            f"相关度 `{source.get('score')}`　`{source.get('chunk_id')}`\n\n"
            f"> {str(source.get('text') or '').replace(chr(10), ' ')}\n"
        )
    return "\n".join(lines)


def append_resume_marker(content: str, token: Optional[str]) -> str:
    """把 resume_token 以 HTML 注释形式附在消息开头。

    为什么用这种方式传 token：Gradio 的按钮回调无法直接携带"上一条消息里的字段"，
    而把 token 存进 gr.State 需要跨回调同步，容易出现状态不一致。
    放进消息内容里（HTML 注释对用户不可见）可以保证"点了哪个确认按钮，
    就一定是那条挂起记录的 token"，不会串台。
    """
    if not token:
        return content
    return f"<!--resume_token:{token}-->{content}"


def build_assistant_message(
    answer: str,
    sources: List[Dict[str, Any]],
    trace: List[str],
    pending: Optional[Dict[str, Any]] = None,
) -> str:
    """组装助手消息（答案 + 来源折叠 + 待确认提示）。"""
    parts = [answer or "_（正在生成…）_"]
    if pending:
        parts.append(
            f"\n\n---\n⚠️ **需要你确认**：准备执行 `{pending.get('tool')}`"
            f"（参数 `{json.dumps(pending.get('args') or {}, ensure_ascii=False)}`）。"
            f"请点击下方的「✅ 确认执行」或「❌ 拒绝」。"
        )
    if sources:
        body = render_sources(sources)
        parts.append(
            "\n\n<details><summary>📚 来源（点击展开，共 "
            f"{len(sources)} 条）</summary>\n\n{body}\n</details>"
        )
    if trace:
        parts.append(
            "\n\n<details><summary>🔍 执行轨迹（点击展开，共 "
            f"{len(trace)} 步）</summary>\n\n" + "\n\n".join(trace) + "\n</details>"
        )
    return append_resume_marker("\n".join(parts), (pending or {}).get("resume_token"))


async def chat_respond(
    question: str,
    history: List[Dict[str, str]],
    session_id: str,
    max_steps: int,
    enable_trace: bool,
) -> AsyncIterator[Tuple[Any, ...]]:
    """流式对话主函数。

    Gradio 的异步生成器：每收到一个事件就 ``yield`` 一次，
    因此打字机效果与轨迹滚动都是真实进度。

    Yields:
        ``(对话历史, 清空后的输入框, session_id, 状态行, 确认按钮可见性, resume_token)``
    """
    question = (question or "").strip()
    if not question:
        yield history, "", session_id, "_请输入问题_", gr.update(visible=False), ""
        return

    history = list(history or [])
    history.append({"role": "user", "content": question})

    answer_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    trace_lines: List[str] = []
    pending: Optional[Dict[str, Any]] = None
    status_line = "🔄 正在执行…"
    run_id = ""

    history.append({"role": "assistant", "content": "_（正在思考…）_"})
    yield history, "", session_id, status_line, gr.update(visible=False), ""

    payload = {"question": question, "session_id": session_id, "max_steps": int(max_steps)}
    try:
        async for event in stream_events("/api/chat/stream", payload):
            event_type = str(event.get("type"))
            event_payload = event.get("payload") or {}
            run_id = event.get("run_id") or run_id

            trace_lines.append(render_trace_line(event, len(trace_lines) + 1))

            if event_type == "plan":
                status_line = f"🧠 决策：{event_payload.get('action')}" + (
                    f" → {event_payload.get('tool')}" if event_payload.get("tool") else ""
                )
            elif event_type == "tool_start":
                status_line = f"🔧 正在调用 `{event_payload.get('tool')}`…"
            elif event_type == "tool_end":
                status_line = (
                    f"✅ `{event_payload.get('tool')}` 完成"
                    f"（{event_payload.get('latency_ms', 0)}ms）"
                )
            elif event_type == "refuse":
                status_line = "🚫 资料中没有相关内容，已拒答"
                answer_parts = [str(event_payload.get("answer") or "")]
            elif event_type == "pending_confirmation":
                pending = event_payload
                status_line = "⚠️ 等待你确认危险操作"
            elif event_type == "final":
                sources = event_payload.get("sources") or []
                answer = str(event_payload.get("answer") or "")
                if answer:
                    answer_parts = [answer]
                status_line = "🏁 完成"
            elif event_type == "error":
                answer_parts = [f"❌ 执行出错：{event_payload.get('message')}"]
                status_line = "❌ 出错"

            content = "".join(answer_parts) or "_（正在生成…）_"
            history[-1] = {
                "role": "assistant",
                "content": build_assistant_message(
                    content, sources, trace_lines if enable_trace else [], pending
                ),
            }
            yield (
                history,
                "",
                session_id,
                status_line,
                gr.update(visible=pending is not None),
                (pending or {}).get("resume_token") or "",
            )
    except Exception as exc:  # noqa: BLE001 - UI 层兜底
        logger.exception("对话失败：%s", exc)
        history[-1] = {"role": "assistant", "content": f"❌ 请求失败：{exc}"}
        yield history, "", session_id, "❌ 请求失败", gr.update(visible=False), ""
        return

    # 收尾：把最终答案与折叠面板再渲染一次（保证内容完整）
    if not answer_parts:
        answer_parts = ["_没有收到回答，请检查后端日志_"]
    history[-1] = {
        "role": "assistant",
        "content": build_assistant_message(
            "".join(answer_parts), sources, trace_lines if enable_trace else [], pending
        ),
    }
    if pending:
        status_line = "⚠️ 等待你确认危险操作（点击下方按钮）"
    elif run_id:
        status_line = f"🏁 完成（trace: `{run_id}`）"
    yield (
        history,
        "",
        session_id,
        status_line,
        gr.update(visible=pending is not None),
        (pending or {}).get("resume_token") or "",
    )


async def confirm_action(
    approved: bool,
    history: List[Dict[str, str]],
    session_id: str,
    token: str,
    enable_trace: bool,
) -> AsyncIterator[Tuple[Any, ...]]:
    """确认或拒绝挂起的危险操作，并继续流式对话。

    Args:
        approved: True=执行该操作，False=拒绝。
        token: 挂起时返回的 resume_token（由隐藏输入框传入）。
    """
    history = list(history or [])
    token = (token or "").strip() or (_extract_resume_token(history) or "")
    if not token:
        yield history, session_id, "没有待确认的操作", gr.update(visible=False)
        return

    answer_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    trace_lines: List[str] = []
    pending: Optional[Dict[str, Any]] = None
    action = "确认执行" if approved else "已拒绝"
    status_line = f"🔄 {action}，继续执行…"

    history.append({"role": "assistant", "content": f"_{action}，正在继续…_"})
    yield history, session_id, status_line, gr.update(visible=False)

    try:
        async for event in stream_events(
            "/api/chat/confirm",
            {"resume_token": token, "approved": bool(approved), "session_id": session_id},
        ):
            event_type = str(event.get("type"))
            event_payload = event.get("payload") or {}
            trace_lines.append(render_trace_line(event, len(trace_lines) + 1))

            if event_type == "tool_end":
                status_line = f"✅ `{event_payload.get('tool')}` 完成"
            elif event_type == "final":
                sources = event_payload.get("sources") or []
                answer = str(event_payload.get("answer") or "")
                if answer:
                    answer_parts = [answer]
                status_line = "🏁 完成"
            elif event_type == "error":
                answer_parts = [f"❌ 执行出错：{event_payload.get('message')}"]
                status_line = "❌ 出错"
            elif event_type == "done":
                status_line = "🏁 完成"

            content = "".join(answer_parts) or "_（正在继续…）_"
            history[-1] = {
                "role": "assistant",
                "content": build_assistant_message(
                    content, sources, trace_lines if enable_trace else [], pending
                ),
            }
            yield history, session_id, status_line, gr.update(visible=False)
    except Exception as exc:  # noqa: BLE001
        history[-1] = {"role": "assistant", "content": f"❌ 继续执行失败：{exc}"}
        yield history, session_id, "❌ 失败", gr.update(visible=False)
        return

    if not answer_parts:
        answer_parts = ["_操作已处理，但没有收到后续回答_"]
    history[-1] = {
        "role": "assistant",
        "content": build_assistant_message(
            "".join(answer_parts), sources, trace_lines if enable_trace else []
        ),
    }
    yield history, session_id, f"🏁 {action}后的结果如上", gr.update(visible=False)


def _extract_resume_token(history: List[Dict[str, str]]) -> Optional[str]:
    """从对话历史里找回 resume_token。

    做法：确认按钮触发时无法直接携带 token（Gradio 事件签名限制），
    因此把它写进消息的 HTML 注释里，这里用正则取回。
    比额外维护一份 session state 更简单，也不会因为状态不同步而失效。
    """
    import re

    for message in reversed(history or []):
        content = str(message.get("content") or "")
        match = re.search(r"<!--resume_token:([0-9a-f]+)-->", content)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# 通道一：结构化行程表单（直接调用工具 + 富结构渲染）
# ---------------------------------------------------------------------------
# 偏好标签（与后端 trip_planner 子系统的 PREFERENCE_OPTIONS 对齐）
PREFERENCE_CHOICES = [
    "历史文化", "自然风光", "美食", "购物", "亲子", "摄影", "夜生活", "休闲度假",
]
BUDGET_LEVEL_CHOICES = ["经济", "中等", "豪华"]

# 档位 → 人均每天参考预算（与后端 LEVEL_DAILY_BUDGET 保持一致）
LEVEL_DAILY_BUDGET = {"经济": 500, "中等": 1200, "豪华": 3000}

# 渲染时最多展示多少个景点 / 酒店卡片
MAX_ATTRACTION_CARDS = 8
MAX_HOTEL_CARDS = 4


def calc_reference_budget(days: float, travelers: float, level: str) -> int:
    """按"天数 × 人数 × 档位日均"推算参考总预算。

    用户点「按档位推算」按钮时填充到预算输入框，避免自己算。
    """
    daily = LEVEL_DAILY_BUDGET.get(level or "中等", 1200)
    return int(daily * max(int(days or 1), 1) * max(int(travelers or 1), 1))


def _format_daily_card(day: Dict[str, Any]) -> str:
    """把一天的行程渲染成一张结构化卡片（Markdown）。"""
    lines = [
        f"#### 第 {day.get('day')} 天｜{day.get('date')}　{day.get('theme', '')}",
        "",
        f"- **上午**：{day.get('morning', '')}",
        f"- **下午**：{day.get('afternoon', '')}",
        f"- **晚间**：{day.get('evening', '')}",
        f"- **住宿**：{day.get('accommodation', '')}",
        f"- **交通**：{day.get('transportation', '')}",
    ]
    meals = day.get("meals") or []
    if meals:
        lines.append("- **餐饮**：")
        lines.extend(f"    - {meal}" for meal in meals)
    if day.get("weather_note"):
        lines.append(f"- **天气**：{day['weather_note']}")
    if day.get("estimated_cost"):
        lines.append(f"- **当日花费**：约 {day['estimated_cost']} 元")
    if day.get("tips"):
        lines.append(f"- **贴士**：{day['tips']}")
    return "\n".join(lines)


def _format_attractions(attractions: List[Dict[str, Any]]) -> str:
    """景点列表（折叠面板内容）。"""
    if not attractions:
        return ""
    lines: List[str] = []
    for item in attractions[:MAX_ATTRACTION_CARDS]:
        ticket = "免费" if not item.get("ticket_price") else f"{item['ticket_price']} 元"
        tags = "、".join(item.get("tags") or [])
        lines.append(
            f"**{item.get('name')}**　票价 {ticket}　建议 {item.get('duration_hours')} 小时"
            f"　评分 {item.get('rating')}　位置 {item.get('location', '')}\n\n"
            f"> {str(item.get('description', ''))[:200]}\n"
            + (f"\n<sub>标签：{tags}</sub>\n" if tags else "")
        )
    return "\n".join(lines)


def _format_hotels(hotels: List[Dict[str, Any]]) -> str:
    """酒店列表（折叠面板内容）。"""
    if not hotels:
        return ""
    lines: List[str] = []
    for item in hotels[:MAX_HOTEL_CARDS]:
        tags = "、".join(item.get("tags") or [])
        lines.append(
            f"**{item.get('name')}**　{item.get('price_per_night')} 元/晚　"
            f"评分 {item.get('rating')}　{item.get('level', '')}\n\n"
            f"- 位置：{item.get('location', '')}（{item.get('distance_to_center', '')}）\n"
            + (f"- 标签：{tags}\n" if tags else "")
        )
    return "\n".join(lines)


def _format_weather(weather: List[Dict[str, Any]]) -> str:
    """逐日天气（折叠面板内容）。"""
    if not weather:
        return ""
    lines = ["| 日期 | 星期 | 天气 | 温度 | 风力 | 建议 |", "| --- | --- | --- | --- | --- | --- |"]
    for item in weather:
        lines.append(
            f"| {item.get('date')} | {item.get('weekday', '')} | {item.get('condition', '')} | "
            f"{item.get('temp_min')}~{item.get('temp_max')}℃ | {item.get('wind', '')} | "
            f"{item.get('suggestion', '')} |"
        )
    return "\n".join(lines)


def _format_budget_table(breakdown: Dict[str, Any], budget: int, estimated_total: int) -> str:
    """预算明细表（含占比与结论）。"""
    if not breakdown:
        return ""
    total = breakdown.get("合计") or estimated_total or 1
    lines = ["| 项目 | 金额（元） | 占比 |", "| --- | --- | --- |"]
    for name, amount in breakdown.items():
        if name == "合计":
            continue
        try:
            ratio = f"{float(amount) / float(total) * 100:.1f}%"
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = "—"
        lines.append(f"| {name} | {amount} | {ratio} |")
    lines.append(f"| **合计** | **{total}** | 100% |")

    diff = int(estimated_total or total) - int(budget or 0)
    if budget:
        verdict = f"低于预算 {abs(diff)} 元" if diff <= 0 else f"超出预算 {diff} 元"
        lines.append(f"\n**用户预算**：{budget} 元　→　**{verdict}**")
    return "\n".join(lines)


def render_trip_plan(data: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """把工具返回的结构化行程渲染成完整页面（这是"不再单薄"的关键）。

    为什么不直接把 ``display`` 贴出来：``display`` 是给大模型看的紧凑文本，
    而结构化渲染能做出标题层级、表格、折叠面板，信息密度和可读性都更高。
    """
    if not data:
        return "❌ 没有拿到行程数据"

    daily_plans = data.get("daily_plans") or []
    header = (
        f"## 🗺 {data.get('destination')} {data.get('days')} 天行程\n\n"
        f"**出发日期**：{data.get('start_date')}　|　**返程日期**：{data.get('end_date')}　|　"
        f"**出行人数**：{data.get('travelers')} 人　|　**预算档位**：{data.get('budget_level')}\n\n"
        f"**总预算**：{data.get('budget')} 元　|　**预估总花费**：{data.get('estimated_total')} 元\n\n"
        + (
            f"**偏好**：{'、'.join(data.get('preferences') or [])}\n\n"
            if data.get("preferences")
            else ""
        )
        + f"> {data.get('summary', '')}"
    )

    sections: List[str] = [header]

    if data.get("budget_status"):
        sections.append(f"### 💰 预算结论\n\n{data['budget_status']}")

    if daily_plans:
        cards = "\n\n---\n\n".join(_format_daily_card(day) for day in daily_plans)
        sections.append(f"### 📅 逐日行程（{len(daily_plans)} 天）\n\n{cards}")
    else:
        sections.append("### 📅 逐日行程\n\n（未生成每日安排）")

    budget_table = _format_budget_table(
        data.get("budget_breakdown") or {},
        int(data.get("budget") or 0),
        int(data.get("estimated_total") or 0),
    )
    if budget_table:
        sections.append(f"### 📊 预算明细\n\n{budget_table}")

    attractions = _format_attractions(data.get("attractions") or [])
    if attractions:
        sections.append(
            f"<details><summary>🎫 推荐景点（{len(data.get('attractions') or [])} 个，点击展开）"
            f"</summary>\n\n{attractions}\n</details>"
        )

    hotels = _format_hotels(data.get("hotels") or [])
    if hotels:
        sections.append(
            f"<details><summary>🏨 推荐酒店（{len(data.get('hotels') or [])} 家，点击展开）"
            f"</summary>\n\n{hotels}\n</details>"
        )

    weather = _format_weather(data.get("weather") or [])
    if weather:
        sections.append(
            f"<details><summary>🌤 逐日天气（{len(data.get('weather') or [])} 天，点击展开）"
            f"</summary>\n\n{weather}\n</details>"
        )

    tips = data.get("tips") or []
    if tips:
        sections.append("### ⚠️ 注意事项\n\n" + "\n".join(f"- {tip}" for tip in tips))

    # 生成方式与子智能体轨迹（体现"内部是 4 个智能体在协作"）
    footer = [f"<sub>生成方式：{data.get('generated_by', '')}</sub>"]
    traces = (meta or {}).get("agent_traces") or []
    if traces:
        rows = "\n".join(
            f"| {t.get('agent')} | {t.get('status')} | {t.get('duration_ms')} ms | {t.get('summary', '')[:60]} |"
            for t in traces
        )
        footer.append(
            "<details><summary>🤖 子智能体协作轨迹（点击展开）</summary>\n\n"
            "| 智能体 | 状态 | 耗时 | 结果 |\n| --- | --- | --- | --- |\n" + rows + "\n</details>"
        )
    sections.append("\n\n".join(footer))

    return "\n\n".join(sections)


async def generate_trip_plan(
    destination: str,
    start_date: Any,
    days: float,
    travelers: float,
    budget: float,
    budget_level: str,
    preferences: Optional[List[str]],
    notes: str,
) -> AsyncIterator[Tuple[Any, ...]]:
    """通道一：按结构化参数直接调用 trip_planner 工具并渲染结果。

    为什么用异步生成器：行程规划要跑 4 个智能体（开启大模型时约 20~60 秒），
    必须让用户看到"正在做什么"，否则又是"点了没反应"。

    Yields:
        ``(进度说明, 富结构渲染结果)``
    """
    destination = (destination or "").strip()
    if not destination:
        yield "❌ 请先填写目的地", ""
        return

    # 日期归一化（Gradio 的 DatePicker 可能给 datetime.date 或字符串）
    date_text = ""
    if hasattr(start_date, "strftime"):
        date_text = start_date.strftime("%Y-%m-%d")
    elif start_date:
        date_text = str(start_date).strip()[:10]

    args: Dict[str, Any] = {
        "destination": destination,
        "days": int(days or 3),
        "travelers": int(travelers or 2),
        "budget": int(budget or calc_reference_budget(days, travelers, budget_level)),
        "budget_level": budget_level or "中等",
        "preferences": list(preferences or []),
    }
    if date_text:
        args["start_date"] = date_text
    if (notes or "").strip():
        args["notes"] = notes.strip()

    level_text = "、".join(args["preferences"]) or "无特别偏好"
    yield (
        f"⏳ 正在规划 {destination} {args['days']} 天行程"
        f"（{args['travelers']} 人，{args['budget_level']}预算 {args['budget']} 元，偏好：{level_text}）……\n\n"
        "> 内部会依次执行：景点搜索 → 天气查询 → 酒店推荐 → 行程规划（4 个智能体协作）",
        "",
    )

    try:
        async with _client(timeout=900.0) as client:
            response = await client.post(
                "/api/tools/trip_planner/invoke", json={"args": args}
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001
        yield f"❌ 调用失败：{exc}", ""
        return

    result = payload.get("result") or {}
    if not payload.get("ok"):
        error = result.get("error") or "工具执行失败"
        yield f"❌ {error}\n\n> 提示：请检查目的地写法，或改用对话框描述需求。", ""
        return

    data = result.get("data") or {}
    meta = result.get("meta") or {}
    parsed = meta.get("parsed") or {}
    degraded = result.get("degraded")

    progress = (
        f"✅ 已生成 {data.get('destination')} {data.get('days')} 天行程"
        f"（实际使用参数：{parsed.get('travelers')} 人 / {parsed.get('budget')} 元"
        f"（{parsed.get('budget_source', '')}）/ 档位 {parsed.get('budget_level')}）"
    )
    if degraded:
        progress += "\n\n> ⚠️ 本次未使用大模型（离线或未配置 Key），行程由本地规则引擎生成。"

    yield progress, render_trip_plan(data, meta)


async def ask_agent_about_trip(
    destination: str,
    days: float,
    travelers: float,
    budget: float,
    budget_level: str,
    preferences: Optional[List[str]],
) -> str:
    """把表单内容转成一句话，填进对话框（通道二的入口）。

    这样用户既可以用表单精确指定参数，也可以把需求丢进对话框让 Agent 处理
    （例如"顺便帮我查一下公司差旅标准"这类需要多工具协作的问题）。
    """
    parts = [f"帮我规划{destination or '（请填目的地）'}{int(days or 3)}天行程"]
    if travelers:
        parts.append(f"{int(travelers)}人出行")
    if budget:
        parts.append(f"总预算{int(budget)}元")
    if budget_level:
        parts.append(f"{budget_level}档")
    if preferences:
        parts.append("偏好" + "、".join(preferences))
    return "，".join(parts)


# ---------------------------------------------------------------------------
# 组装界面
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    """构建 Gradio 界面。"""
    with gr.Blocks(title="Agent 平台", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🤖 Agent 平台\n"
            "可插拔工具协议的企业级 Agent：**知识库检索**与**旅行规划**由 Planner 自主路由，"
            "带来源引用、执行轨迹与人工确认。"
        )
        status_box = gr.Markdown("正在连接后端…")

        session_state = gr.State("")

        with gr.Row():
            # ---------------- 左：知识库 ----------------
            with gr.Column(scale=3):
                gr.Markdown("### 📚 知识库")
                upload_box = gr.File(
                    label="上传文档（PDF / DOCX / TXT / MD，可多选）",
                    file_count="multiple",
                    file_types=ALLOWED_EXTENSIONS,
                )
                with gr.Row():
                    upload_button = gr.Button("上传并向量化", variant="primary")
                    refresh_button = gr.Button("刷新列表")
                gr.Markdown(
                    "<sub>若选完文件后没有自动开始，点上面的「上传并向量化」按钮即可。</sub>"
                )
                upload_result = gr.Markdown("")
                document_table = gr.Dataframe(
                    headers=["文件名", "块数", "字数", "页数", "入库时间"],
                    datatype=["str", "number", "number", "str", "str"],
                    label="已入库文档",
                    interactive=False,
                    wrap=True,
                )
                clear_button = gr.Button("🗑 清空知识库", variant="stop")

                gr.Markdown("### 🧰 工具与配置")
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=1, maximum=8, value=4, step=1, label="Agent 步数预算"
                    )
                trace_toggle = gr.Checkbox(value=True, label="在回答下方附带执行轨迹")

            # ---------------- 右：两条输入通道 ----------------
            with gr.Column(scale=7):
                with gr.Tabs():
                    # ============ 通道一：结构化行程规划表单 ============
                    with gr.Tab("🗺 行程规划（表单）"):
                        gr.Markdown(
                            "填好条件后**直接调用行程规划工具**，结果按结构呈现："
                            "逐日卡片 + 景点/酒店/天气列表 + 预算明细表。\n\n"
                            "<sub>这条路不经过对话，参数由你指定，最可控；"
                            "需要「顺便查一下公司差旅标准」这类跨工具协作时，再用旁边的对话通道。</sub>"
                        )
                        with gr.Row():
                            trip_destination = gr.Textbox(
                                label="目的地", value="北京", scale=2,
                                placeholder="例如：北京、成都、杭州",
                            )
                            trip_start_date = gr.DateTime(
                                label="出发日期", value=None, include_time=False, scale=2,
                            )
                        with gr.Row():
                            trip_days = gr.Slider(
                                minimum=1, maximum=15, value=3, step=1, label="天数"
                            )
                            trip_travelers = gr.Slider(
                                minimum=1, maximum=20, value=2, step=1, label="出行人数"
                            )
                        with gr.Row():
                            trip_budget_level = gr.Radio(
                                choices=BUDGET_LEVEL_CHOICES, value="中等", label="预算档位",
                                info="决定住宿与餐饮标准",
                            )
                        with gr.Row():
                            trip_budget = gr.Number(
                                label="总预算（元）", value=7200, precision=0, scale=3,
                            )
                            trip_budget_hint = gr.Button("按档位推算参考值", scale=1)
                        trip_preferences = gr.CheckboxGroup(
                            choices=PREFERENCE_CHOICES, value=["历史文化"], label="旅行偏好（可多选）"
                        )
                        trip_notes = gr.Textbox(
                            label="补充要求（可选）", lines=2,
                            placeholder="例如：带 70 岁老人同行，节奏慢一点；想安排一次烤鸭",
                        )
                        with gr.Row():
                            trip_submit = gr.Button("生成行程", variant="primary", scale=3)
                            trip_to_chat = gr.Button("填入对话框（走 Agent 处理）", scale=2)
                        trip_status = gr.Markdown("_填好条件后点「生成行程」_")
                        trip_output = gr.Markdown(
                            "<sub>生成的行程会显示在这里：逐日安排、景点、天气、酒店与预算明细。</sub>"
                        )

                    # ============ 通道二：对话 ============
                    with gr.Tab("💬 对话（Agent 自主路由）"):
                        gr.Markdown(
                            "<sub>直接说话即可：Agent 会自己判断该查知识库、规划行程，还是发邮件。"
                            "行程类问题在这里也能用，只是呈现为一段文字而不是结构化卡片。</sub>"
                        )
                        chatbot = gr.Chatbot(
                            label="对话",
                            type="messages",
                            height=430,
                            show_copy_button=True,
                            sanitize_html=False,   # 需要渲染 <details> 折叠面板
                            allow_tags=True,       # 显式声明：允许消息里带 HTML 标签
                        )
                        status_line = gr.Markdown("_就绪_")
                        question_box = gr.Textbox(
                            label="",
                            placeholder="试试：年假有几天 ／ 帮我规划北京三日游，喜欢历史文化 ／ 把行程发到 me@example.com",
                            lines=2,
                            show_label=False,
                        )
                        with gr.Row():
                            send_button = gr.Button("发送", variant="primary", scale=3)
                            confirm_button = gr.Button("✅ 确认执行", variant="primary", scale=1, visible=False)
                            reject_button = gr.Button("❌ 拒绝", variant="stop", scale=1, visible=False)
                        # 隐藏字段：保存挂起操作的 resume_token（随确认按钮一起提交）
                        resume_token_box = gr.Textbox(visible=False, value="")
                        gr.Markdown(
                            "<sub>提示：`send_email` 是有副作用的写操作，会先请求确认——"
                            "这是 human-in-the-loop 的演示。</sub>"
                        )

        # ---------------- 事件绑定 ----------------
        demo.load(refresh_status, outputs=status_box)
        demo.load(refresh_documents, outputs=document_table)

        upload_button.click(
            upload_documents, inputs=upload_box, outputs=[upload_result, document_table]
        ).then(refresh_status, outputs=status_box)

        # 文件选择后自动开始（Gradio 的 File 组件在部分版本里不会可靠触发这条链路，
        # 所以上面的按钮是主要入口，这一条只是"选完即开始"的便利路径）。
        upload_box.change(
            upload_documents, inputs=upload_box, outputs=[upload_result, document_table]
        )

        refresh_button.click(refresh_documents, outputs=document_table)
        clear_button.click(
            delete_all_documents, outputs=[upload_result, document_table]
        ).then(refresh_status, outputs=status_box)

        # ---------------- 通道一：行程表单 ----------------
        trip_inputs = [
            trip_destination,
            trip_start_date,
            trip_days,
            trip_travelers,
            trip_budget,
            trip_budget_level,
            trip_preferences,
            trip_notes,
        ]

        # 按档位推算参考预算：天数 / 人数 / 档位任一变化都可以重算
        trip_budget_hint.click(
            calc_reference_budget,
            inputs=[trip_days, trip_travelers, trip_budget_level],
            outputs=trip_budget,
        )

        # 档位 / 天数 / 人数变化时自动同步参考预算。
        # 为什么做这件事：用户很容易填出"豪华档 + 15000 元（3 人 4 天）"这种自相矛盾的组合，
        # 结果预算结论显示"超出预算 9180 元"，看起来像系统算错了。
        # 自动给出与档位匹配的参考值，用户仍可手动改。
        for component in (trip_budget_level, trip_days, trip_travelers):
            component.change(
                calc_reference_budget,
                inputs=[trip_days, trip_travelers, trip_budget_level],
                outputs=trip_budget,
            )

        # 生成行程（异步生成器：边跑边显示进度）
        trip_submit.click(
            generate_trip_plan, inputs=trip_inputs, outputs=[trip_status, trip_output]
        )

        # 把表单内容转成一句话填进对话框（走 Agent 通道，便于跨工具协作）
        trip_to_chat.click(
            ask_agent_about_trip,
            inputs=[
                trip_destination, trip_days, trip_travelers,
                trip_budget, trip_budget_level, trip_preferences,
            ],
            outputs=question_box,
        )

        # ---------------- 通道二：对话 ----------------
        chat_inputs = [question_box, chatbot, session_state, steps_slider, trace_toggle]
        chat_outputs = [chatbot, question_box, session_state, status_line, confirm_button, resume_token_box]

        send_button.click(chat_respond, inputs=chat_inputs, outputs=chat_outputs).then(
            refresh_documents, outputs=document_table
        )
        # 回车提交
        question_box.submit(chat_respond, inputs=chat_inputs, outputs=chat_outputs)

        # 确认 / 拒绝：带着隐藏字段里的 token 与**客户端保留的对话历史**继续执行，
        # 这样确认后的回答会追加在同一段对话里，而不是凭空多出一条。
        confirm_button.click(
            partial(confirm_action, True),
            inputs=[chatbot, session_state, resume_token_box, trace_toggle],
            outputs=[chatbot, session_state, status_line, confirm_button],
        ).then(lambda: gr.update(visible=False), outputs=[reject_button])
        reject_button.click(
            partial(confirm_action, False),
            inputs=[chatbot, session_state, resume_token_box, trace_toggle],
            outputs=[chatbot, session_state, status_line, confirm_button],
        ).then(lambda: gr.update(visible=False), outputs=[confirm_button])

    return demo


# ---------------------------------------------------------------------------
# 独立启动
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    demo = build_demo()
    demo.queue(default_concurrency_limit=8).launch(
        server_name="127.0.0.1", server_port=7860, show_api=False
    )
