"""API 路由包。

按资源拆分子路由，避免 ``app.py`` 随功能增长变成"什么都往里塞"的大文件：

* ``api.documents`` —— 文档上传与管理（B2）
* ``api.chat``      —— 知识库问答与检索预览（B3 / B4 的 rag|agent 两种模式）
* ``api.chat_stream`` —— 流式问答（SSE）与人工确认恢复（B5）
* ``api.tools``     —— 工具目录、直接调用与执行轨迹（B4）
* ``api.evaluation`` —— 评测集概况与历史报告（B6）
  （注意：模块名不能叫 ``eval``，会与项目根目录的顶层 ``eval`` 包冲突）
"""

from . import chat, chat_stream, documents, evaluation, tools  # noqa: F401

__all__ = ["chat", "chat_stream", "documents", "evaluation", "tools"]
