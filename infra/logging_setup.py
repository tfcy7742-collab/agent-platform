"""日志配置。

支持两种输出形态（``LOG_JSON``）：
* 文本（默认）：人读友好，本地开发用；
* JSON：一行一条，便于被 Filebeat / Loki 之类的采集器直接消费。

统一在 ``setup_logging()`` 里配置，避免各模块自行 ``basicConfig`` 导致格式不一致。
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

from config.settings import Settings, get_settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """把日志记录序列化成单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # 允许业务代码通过 logger.info("x", extra={"extra_fields": {...}}) 附加结构化字段
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(settings: Settings | None = None, force: bool = False) -> None:
    """初始化全局日志（幂等）。

    Args:
        settings: 配置；默认取全局配置。
        force: 为 True 时重新配置（测试中切换日志级别时使用）。
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = settings or get_settings()
    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # 第三方库的日志噪音降级，避免淹没自己的日志
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("日志初始化完成：level=%s json=%s", settings.log_level, settings.log_json)


def log_config_snapshot(settings: Settings | None = None) -> None:
    """打印脱敏配置快照（服务启动时的第一行关键日志，便于事后追溯）。"""
    settings = settings or get_settings()
    logger = logging.getLogger("startup")
    snapshot = settings.masked_snapshot()
    logger.info("配置快照：%s", json.dumps(snapshot, ensure_ascii=False))
    if not settings.llm_online:
        logger.warning(
            "当前处于离线模式（LLM_MODE=%s）：Planner 走规则路由、回答走模板拼接，"
            "检索/拒答/评测功能不受影响。配置 DEEPSEEK_API_KEY 后自动启用大模型。",
            settings.llm_mode,
        )
