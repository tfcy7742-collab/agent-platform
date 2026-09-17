"""send_email 工具：演示「危险操作需要人工确认」。

为什么需要这个工具
------------------
前两个工具都是**只读**的，无法体现工具协议里最重要的一条治理能力：
**有副作用的操作必须先获得人工确认**。

把它接进来后，平台就同时具备三类工具，正好覆盖 Planner 的三种决策维度：

======================  ==========  ==========  ==================
工具                     成本        耗时        是否需确认
======================  ==========  ==========  ==================
knowledge_search        medium      medium      否（只读检索）
trip_planner            high        high        否（只读生成）
send_email              low         low         **是**（有副作用）
======================  ==========  ==========  ==================

实现上它是**演示级的**：不真的发邮件，只校验参数、生成邮件正文并落一条日志，
返回"已发送"的说明。这样既能演示确认流程，又不会在演示中真的骚扰别人；
真要接 SMTP，把 ``_run`` 里的落日志换成发送即可（接口与确认闸门都不用改）。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# 简化的邮箱格式校验（够挡住明显错误的输入，不追求 RFC 完备）
EMAIL_PATTERN = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")

# 单次邮件正文长度上限（防止把整份行程塞进去导致内容异常）
MAX_BODY_CHARS = 8000


class SendEmailTool(BaseTool):
    """发送邮件工具（需要人工确认）。"""

    name = "send_email"
    description = (
        "把内容（例如生成的行程规划）以邮件形式发送给指定收件人。"
        "适用于：用户明确要求「发到我的邮箱」「把行程邮件发我」「发给同事」等。"
        "这是**有副作用的写操作**，执行前会要求用户确认。"
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "收件人邮箱地址"},
            "subject": {"type": "string", "description": "邮件主题，默认「旅行行程规划」"},
            "body": {
                "type": "string",
                "description": "邮件正文。用户说「把上面的行程发我」时，应把行程内容填进来",
            },
        },
        "required": ["to", "body"],
    }
    timeout_s = 15.0
    est_cost = "low"        # 不消耗大模型
    est_latency = "low"
    retryable = False       # 写操作不重试，避免重复发送
    requires_confirmation = True

    def __init__(self, sender: str = "agent-platform@example.com") -> None:
        super().__init__()
        self.sender = sender
        # 已"发送"的邮件记录（演示用，便于测试与 UI 展示）
        self.sent: list = []

    # ------------------------------------------------------------------
    def _run(self, to: str = "", subject: str = "", body: str = "", **_: Any) -> ToolResult:
        """校验参数并"发送"邮件。"""
        to = (to or "").strip()
        body = (body or "").strip()
        subject = (subject or "").strip() or "旅行行程规划"

        if not EMAIL_PATTERN.match(to):
            return ToolResult(
                ok=False,
                error=f"收件人邮箱格式不正确：{to!r}",
                error_type="tool_invalid_args",
                display=f"邮箱格式不正确：{to}",
            )
        if not body:
            return ToolResult(
                ok=False,
                error="邮件正文不能为空",
                error_type="tool_invalid_args",
                display="邮件正文为空，无法发送",
            )

        truncated = len(body) > MAX_BODY_CHARS
        body = body[:MAX_BODY_CHARS]

        record = {"to": to, "subject": subject, "chars": len(body), "truncated": truncated}
        self.sent.append(record)
        # 演示：真实场景在这里调用 SMTP / 邮件服务 API
        logger.info("[send_email] 已发送邮件 → %s（主题：%s，%s 字）", to, subject, len(body))

        warning = "（正文过长已截断）" if truncated else ""
        return ToolResult(
            ok=True,
            data={
                "to": to,
                "subject": subject,
                "chars": len(body),
                "truncated": truncated,
                "preview": body[:200],
                "simulated": True,
            },
            display=f"已把「{subject}」发送到 {to}{warning}（演示模式，未真实投递）",
            meta={"simulated": True, "sender": self.sender},
        )

    def stats(self) -> Dict[str, Any]:
        """额外暴露"已发送"计数，便于演示与测试断言。"""
        base = super().stats()
        base["sent_count"] = len(self.sent)
        return base


__all__ = ["EMAIL_PATTERN", "SendEmailTool"]
