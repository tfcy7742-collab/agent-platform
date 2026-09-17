"""Agent 平台 —— 可插拔工具协议的企业级 Agent 系统。

本项目把两类能力（RAG 知识检索、多智能体任务编排）统一抽象为「工具」，
由 Planner 自主路由，并内置流式输出、超时熔断、失败降级、人工确认、
全链路 trace 与离线评测回归。

对外入口：
    ``app.py``       FastAPI 服务（含挂载的 Gradio UI）
    ``ui.py``        Gradio 界面（可独立启动）
    ``eval.run_eval`` 离线评测 CLI
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
