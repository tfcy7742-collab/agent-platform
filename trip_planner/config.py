"""全局配置。

统一从环境变量（含 ``.env`` 文件）读取 DeepSeek 相关配置，
避免在多个模块里重复写 ``os.getenv``。

用法::

    from backend.config import get_settings
    settings = get_settings()
    settings.model        # "deepseek-flash"
    settings.base_url     # "https://api.deepseek.com"
    settings.has_api_key  # 是否配置了可用的 API Key
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

# 项目根目录（backend/ 的上一级），用于定位 .env
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# 尝试加载 .env（python-dotenv 未安装时静默跳过）
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(BACKEND_DIR / ".env")
except Exception:  # pragma: no cover
    pass


def _get_float(name: str, default: float) -> float:
    """读取浮点型环境变量，非法值回退默认值。"""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    """读取整型环境变量，非法值回退默认值。"""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """运行期配置。"""

    api_key: Optional[str]
    base_url: str
    model: str
    timeout: float
    max_retries: int
    max_tool_rounds: int
    cors_origins: list

    @property
    def has_api_key(self) -> bool:
        """是否配置了有效（非占位符）的 API Key。"""
        key = (self.api_key or "").strip()
        if not key:
            return False
        placeholder = {"your_key_here", "sk-xxx", "none", "null", "test"}
        return key.lower() not in placeholder

    @property
    def masked_key(self) -> str:
        """脱敏后的 Key，用于日志/健康检查展示。"""
        if not self.has_api_key:
            return "(未配置)"
        key = self.api_key or ""
        return f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一配置实例（带缓存）。"""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    origins = [item.strip() for item in origins_raw.split(",") if item.strip()]

    return Settings(
        api_key=api_key,
        # 需求指定：DeepSeek 官方 OpenAI 兼容地址
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        # 需求指定：模型 deepseek-flash
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        timeout=_get_float("DEEPSEEK_TIMEOUT", 60.0),
        max_retries=_get_int("DEEPSEEK_MAX_RETRIES", 2),
        max_tool_rounds=_get_int("MAX_TOOL_ROUNDS", 3),
        cors_origins=origins or ["*"],
    )
