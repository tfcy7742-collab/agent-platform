"""Agent 主循环（B4 核心）。

一次问答的执行流程::

    1. 加载会话历史（多轮）
    2. [可选] 问题改写：把"它的年假怎么算"补全为可独立检索的问句
    3. 循环（最多 AGENT_MAX_STEPS 步）：
         Planner 决策 ──▶ 工具执行 ──▶ 结果裁剪 ──▶ 观察累积
         退出条件：决策为 final_answer / 步数触顶 / 需要人工确认
    4. 汇总回答（有工具观察时用一次 LLM 归纳，离线则用规则汇总）
    5. 拒答处理：工具明确表示"资料中没有"时，答案必须是标准拒答话术
    6. 产出事件流 + 结果（供 API 落 trace 并返回）

三条硬约束（都在代码里强制，而不是只写在提示词里）
--------------------------------------------------
1. **步数预算**：触顶强制收敛，``status`` 标记 ``budget_exhausted``；
2. **不重复调用**：相同工具 + 相同参数只执行一次（Planner 后处理与执行器双重把关）；
3. **拒答优先**：只要知识库工具明确拒答，最终答案就必须是标准拒答话术，
   不允许模型用自身知识"补一个答案"。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config.settings import Settings, get_settings
from core.llm import LLMClient, get_llm_client
from core.prompts import REFUSAL_MESSAGE
from core.runtime.events import AgentEvent, EventEmitter, EventType
from core.runtime.executor import ExecutionOutcome, ToolExecutor, get_executor
from core.runtime.planner import Decision, Planner
from core.tools.registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)

# 回答归纳用的提示词（把工具观察整理成面向用户的答案）
SYNTHESIS_SYSTEM_PROMPT = """你是一个严谨的助手。请**只依据**下面提供的工具执行结果回答用户问题。

规则：
1. 不得使用工具结果之外的知识补充或推测；
2. 引用资料时用【文件名 第X页】格式标注来源；
3. 如果工具结果明确表示资料中没有相关信息，必须原样输出：{refusal}
4. **如果工具返回的是行程规划，必须完整呈现，不允许概括**：
   - 逐日展开，每天都写清 上午 / 下午 / 晚间 的具体安排与景点名称；
   - 保留住宿、交通、当日花费与天气提示；
   - 最后单独列出预算明细（分项金额）与预算结论；
   - **禁止**压缩成"第 1 天游览某景点"这类一句话概括——用户需要能照着走的详细行程；
   - 这类内容超过 300 字是允许且必要的。
5. 其他类型的问题保持简洁（300 字以内）。
"""

SYNTHESIS_USER_TEMPLATE = """【用户问题】
{question}

【工具执行结果】
{observations}

请给出最终回答。"""

# 送给大模型的工具结果长度上限。
#
# 第一版这里取 1500 字符，而行程规划的完整文本可能上万字，结果大模型只能
# 复述一个极简版本，用户反馈"比原项目单薄很多"。**这是适配层的偷懒**：
# 子系统本身返回的每一天都是完整内容，是我们在传递时把它截掉了。
# 现在放宽到 30000 字符，只有真正超长时才截断并明确标注。
OBSERVATION_TEXT_LIMIT = 30000


@dataclass
class AgentResult:
    """一次 Agent 执行的完整结果。"""

    answer: str
    run_id: str = ""
    session_id: str = ""
    question: str = ""
    rewritten_question: Optional[str] = None
    refused: bool = False
    refuse_reason: Optional[str] = None
    status: str = "success"
    steps: int = 0
    tools_used: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    decision_sources: List[str] = field(default_factory=list)
    degraded: bool = False
    total_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_est: float = 0.0
    events: List[AgentEvent] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    pending: Optional[Dict[str, Any]] = None
    # 已完成的工具观察（恢复执行时需要带上，避免从头重跑）
    observations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转成 API 响应结构（不含事件流，事件由 SSE 单独推送）。"""
        return {
            "answer": self.answer,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "refused": self.refused,
            "refuse_reason": self.refuse_reason,
            "status": self.status,
            "steps": self.steps,
            "tools_used": self.tools_used,
            "tool_calls": self.tool_calls,
            "degraded": self.degraded,
            "total_ms": self.total_ms,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cost_est": round(self.cost_est, 6),
            },
            "sources": self.sources,
            "pending": self.pending,
            "error": self.error,
            "error_type": self.error_type,
            "meta": self.meta,
        }


class AgentRuntime:
    """Agent 运行时。"""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        planner: Optional[Planner] = None,
        executor: Optional[ToolExecutor] = None,
        llm: Optional[LLMClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_registry()
        self.llm = llm or get_llm_client()
        self.planner = planner or Planner(
            registry=self.registry, llm=self.llm, use_llm=self.llm.available
        )
        self.executor = executor or get_executor(self.registry)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        session_id: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        max_steps: Optional[int] = None,
        emit: Optional[Callable[[AgentEvent], None]] = None,
        run_id: str = "",
        observations: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tools_used: Optional[List[str]] = None,
        rewritten_question: Optional[str] = None,
        resumed: bool = False,
    ) -> AgentResult:
        """执行一次完整的 Agent 流程。

        Args:
            question: 用户问题（恢复执行时应传**改写后**的问题）。
            session_id: 会话 id。
            history: 历史消息（多轮上下文）。
            max_steps: 覆盖默认步数预算。
            emit: 事件回调（SSE 用它做实时推送）。
            run_id: 外部传入的 trace id（由 API 层生成，便于与 trace 表对齐）。
            observations: **恢复执行时传入**已完成的工具观察，
                避免从头重跑（重复调用工具、重复消耗 token）。
            tool_calls: 恢复执行时传入已有的工具调用记录。
            tools_used: 恢复执行时传入已用过的工具名。
            rewritten_question: 恢复执行时传入上次的改写结果。
            resumed: 是否为"人工确认后恢复执行"（用于事件与 trace 标记）。

        Returns:
            ``AgentResult``。
        """
        started = time.perf_counter()
        question = (question or "").strip()
        history = history or []
        emitter = EventEmitter(run_id=run_id, callback=emit)
        budget = max_steps or self.settings.agent_max_steps

        result = AgentResult(answer="", run_id=run_id, session_id=session_id, question=question)
        result.tool_calls = list(tool_calls or [])
        result.tools_used = list(tools_used or [])
        result.rewritten_question = rewritten_question

        if not question:
            result.answer = "请先告诉我你想问什么。"
            result.status = "invalid"
            emitter.emit(EventType.FINAL, answer=result.answer)
            emitter.emit(EventType.DONE)
            result.events = emitter.events
            return result

        # ---- 1. 问题改写（多轮指代消解；恢复执行时跳过，沿用上次结果）----
        current_question = question
        if not resumed and self.settings.enable_query_rewrite and history:
            decision = self.planner.rewrite(question, history)
            self._account_tokens(result, decision)
            if decision.rewritten and decision.rewritten != question:
                current_question = decision.rewritten
                result.rewritten_question = current_question
                emitter.emit(
                    EventType.REWRITE,
                    original=question,
                    rewritten=current_question,
                    reason=decision.reason,
                )

        # ---- 2. 主循环 ----
        # 若本次是"人工确认后恢复"，且上一次挂起前 Planner 又提出了同一个待确认操作
        # （观察里会留下 tool_pending_confirmation），就把这条噪声观察滤掉：
        # 否则 Planner 会基于"已发送成功"的内容去拼邮件正文，生成出**不一样的参数**，
        # 从而被当成"新操作"再次要求确认（死循环）。
        # 滤掉之后，Planner 看到的状态与首次一致，会产出完全相同的参数，
        # 于是能命中执行器里"已获人工确认"的记录，直接执行。
        observations = list(observations or [])
        if resumed:
            filtered = [
                item
                for item in observations
                if item.get("error_type") != "tool_pending_confirmation"
            ]
            logger.warning(
                "恢复执行：观察 %s 条 → 过滤待确认噪声后 %s 条（resumed=%s）",
                len(observations), len(filtered), resumed,
            )
            observations = filtered
        pending_payload: Optional[Dict[str, Any]] = None

        for step in range(budget):
            decision = self.planner.decide(
                current_question,
                history=history,
                observations=observations,
                allow_rewrite=bool(history) and result.rewritten_question is None,
            )
            self._account_tokens(result, decision)
            result.decision_sources.append(decision.source)

            emitter.emit(
                EventType.PLAN,
                step=step + 1,
                thought=decision.thought,
                action=decision.action,
                tool=decision.tool,
                args=decision.args,
                source=decision.source,
                latency_ms=decision.latency_ms,
            )
            result.steps = step + 1

            # 2.1 需要改写（只在 LLM 路径下会发生）
            if decision.action == "rewrite" and decision.rewritten:
                current_question = decision.rewritten
                result.rewritten_question = current_question
                emitter.emit(
                    EventType.REWRITE,
                    original=question,
                    rewritten=current_question,
                    reason=decision.reason,
                )
                continue

            # 2.2 直接回答
            if decision.action == "final_answer":
                result.answer = decision.answer
                break

            # 2.3 调用工具
            if decision.action == "tool" and decision.tool:
                emitter.emit(
                    EventType.TOOL_START, tool=decision.tool, args=decision.args, step=step + 1
                )
                outcome = self.executor.execute(
                    decision.tool, decision.args, question=current_question
                )
                observation = outcome.to_observation()
                observations.append(observation)
                result.tool_calls.append(
                    {
                        "step": step + 1,
                        "tool": outcome.tool_name,
                        "args": outcome.args,
                        "ok": outcome.result.ok,
                        "latency_ms": outcome.result.latency_ms,
                        "error_type": outcome.result.error_type,
                        "retried": outcome.retried,
                        "attempts": outcome.attempts,
                    }
                )
                if outcome.tool_name not in result.tools_used:
                    result.tools_used.append(outcome.tool_name)
                if outcome.result.degraded:
                    result.degraded = True

                emitter.emit(
                    EventType.TOOL_END,
                    tool=outcome.tool_name,
                    ok=outcome.result.ok,
                    latency_ms=outcome.result.latency_ms,
                    degraded=outcome.result.degraded,
                    retried=outcome.retried,
                    error=outcome.result.error,
                    error_type=outcome.result.error_type,
                    hits=len((outcome.result.data or {}).get("sources", []))
                    if isinstance(outcome.result.data, dict)
                    else 0,
                )
                emitter.emit(
                    EventType.OBSERVATION,
                    summary=outcome.result.to_llm_text()[:200],
                    chars=len(outcome.result.to_llm_text()),
                    tool=outcome.tool_name,
                )

                # 人工确认：挂起整个流程，等用户确认后从 /api/chat/confirm 恢复
                if outcome.pending_confirmation:
                    pending_payload = outcome.result.meta.get("pending")
                    result.status = "pending_confirmation"
                    result.pending = pending_payload
                    emitter.emit(EventType.PENDING_CONFIRMATION, **(pending_payload or {}))
                    break
                continue
            # 2.4 兜底：未知动作
            logger.warning("未知决策动作：%s", decision.action)
            break
        else:
            # for-else：循环正常走完（没有 break）说明步数触顶
            result.status = "budget_exhausted"
            logger.info("步数预算触顶（%s 步），强制收敛", budget)

        # ---- 3. 汇总回答 ----
        if result.status == "pending_confirmation":
            # 挂起场景不会走到 _finalize_answer，但前端仍需要一个明确的"本轮结束"信号
            emitter.emit(
                EventType.FINAL,
                answer="",
                sources=[],
                refused=False,
                status=result.status,
                pending=result.pending,
            )
        else:
            self._finalize_answer(result, current_question, observations, emitter)

        result.total_ms = int((time.perf_counter() - started) * 1000)
        # 把累积的观察交回给调用方：人工确认恢复时需要它避免从头重跑
        result.observations = observations
        emitter.emit(EventType.DONE, status=result.status)
        result.events = emitter.events
        result.meta.setdefault("step_budget", budget)
        result.meta["planner_sources"] = result.decision_sources
        result.meta["resumed"] = resumed
        return result

    # ------------------------------------------------------------------
    # 回答生成
    # ------------------------------------------------------------------
    def _finalize_answer(
        self,
        result: AgentResult,
        question: str,
        observations: List[Dict[str, Any]],
        emitter: EventEmitter,
    ) -> None:
        """生成最终回答（含拒答处理与来源收集）。"""
        # ---- 拒答优先：知识库明确说"资料里没有" ----
        refused_observation = next(
            (item for item in observations if item.get("refused")), None
        )
        # 只有"知识库拒答"且没有其他成功工具时才拒答（避免行程工具成功却被知识库拒答带偏）
        successful_other = [
            item for item in observations
            if item.get("ok") and item.get("tool") != "knowledge_search"
        ]
        if refused_observation is not None and not successful_other:
            result.answer = REFUSAL_MESSAGE
            result.refused = True
            result.refuse_reason = str(
                (refused_observation.get("data") or {}).get("refuse_reason") or "below_threshold"
            )
            result.status = "refused"
            emitter.emit(EventType.REFUSE, reason=result.refuse_reason, answer=REFUSAL_MESSAGE)
            emitter.emit(EventType.FINAL, answer=REFUSAL_MESSAGE, sources=[], refused=True)
            return

        # ---- 收集来源 ----
        result.sources = self._collect_sources(observations)

        # ---- 已有答案（Planner 直接给出）----
        base_answer = (result.answer or "").strip()

        # ---- 有工具观察时，用一次 LLM 归纳（离线则用规则汇总）----
        if observations and self.llm.available:
            synthesized = self._synthesize(question, observations, result)
            if synthesized:
                base_answer = synthesized
        elif observations and not base_answer:
            base_answer = self._rule_summary(observations)
        elif observations:
            # 离线模式：Planner 已给出答案，但补上来源标注，便于前端展示
            base_answer = self._append_sources(base_answer, result.sources)

        if not base_answer:
            base_answer = "我暂时无法回答这个问题，请补充更多信息。"

        result.answer = base_answer
        emitter.emit(
            EventType.FINAL,
            answer=base_answer,
            sources=result.sources,
            refused=False,
            status=result.status,
        )

    def _synthesize(
        self,
        question: str,
        observations: List[Dict[str, Any]],
        result: AgentResult,
    ) -> Optional[str]:
        """用一次 LLM 调用把工具观察归纳成面向用户的答案。"""
        observation_text = "\n\n".join(
            f"[{index}] 工具 {item.get('tool')}（{'成功' if item.get('ok') else '失败'}）：\n"
            f"{self._observation_text(item)}"
            for index, item in enumerate(observations, start=1)
        )
        system_prompt = SYNTHESIS_SYSTEM_PROMPT.format(refusal=REFUSAL_MESSAGE)
        user_prompt = SYNTHESIS_USER_TEMPLATE.format(
            question=question, observations=observation_text
        )
        llm_result = self.llm.chat(system_prompt, user_prompt)
        result.prompt_tokens += llm_result.prompt_tokens
        result.completion_tokens += llm_result.completion_tokens
        result.total_tokens += llm_result.total_tokens
        result.cost_est += llm_result.cost_est
        if llm_result.ok and llm_result.text.strip():
            return llm_result.text.strip()

        # 归纳失败 → 退回规则汇总（并标记降级）
        result.degraded = True
        logger.info("回答归纳失败（%s），使用规则汇总", llm_result.error_type)
        return self._rule_summary(observations)

    @staticmethod
    def _observation_text(item: Dict[str, Any]) -> str:
        """取出单个工具观察的完整文本（供回答汇总使用）。

        为什么单独抽一个函数：这里曾经硬编码 ``[:1500]`` 截断，
        而行程规划的完整文本可能上万字，导致最终回答被"压扁"。
        截断长度现在由 ``OBSERVATION_TEXT_LIMIT`` 统一控制，并会在截断时明确标注。
        """
        text = str(item.get("text") or item.get("display") or "")
        if len(text) > OBSERVATION_TEXT_LIMIT:
            logger.warning(
                "工具 %s 的结果超长（%s 字符），已截断到 %s 字符",
                item.get("tool"), len(text), OBSERVATION_TEXT_LIMIT,
            )
            text = text[:OBSERVATION_TEXT_LIMIT] + "\n…（内容过长，已截断）"
        return text

    @staticmethod
    def _rule_summary(observations: List[Dict[str, Any]]) -> str:
        """规则汇总：把各工具 display 串起来（离线模式的回答）。"""
        parts: List[str] = []
        for index, item in enumerate(observations, start=1):
            display = str(item.get("display") or item.get("text") or "").strip()
            if display:
                parts.append(display)
        if not parts:
            return "工具没有返回可用内容。"
        return "\n\n".join(parts)

    @staticmethod
    def _collect_sources(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """收集所有工具返回的来源片段（去重，保留最高分）。"""
        collected: Dict[str, Dict[str, Any]] = {}
        for item in observations:
            data = item.get("data")
            if not isinstance(data, dict):
                continue
            for source in data.get("sources") or []:
                if not isinstance(source, dict):
                    continue
                key = str(source.get("chunk_id") or source.get("file_name") or "")
                if not key:
                    continue
                existing = collected.get(key)
                if existing is None or float(source.get("score") or 0) > float(existing.get("score") or 0):
                    collected[key] = source
        return sorted(collected.values(), key=lambda entry: -float(entry.get("score") or 0))[:5]

    @staticmethod
    def _append_sources(answer: str, sources: List[Dict[str, Any]]) -> str:
        """给答案补上来源标注（如果答案里还没有引用）。"""
        if not sources or "【" in answer:
            return answer
        marks = []
        for source in sources[:3]:
            page = f" 第 {source['page']} 页" if source.get("page") else ""
            marks.append(f"{source.get('file_name', '未知文档')}{page}")
        return f"{answer}\n\n来源：{'；'.join(marks)}"

    @staticmethod
    def _account_tokens(result: AgentResult, decision: Decision) -> None:
        """累计 Planner 决策消耗的 token。"""
        result.prompt_tokens += decision.prompt_tokens
        result.completion_tokens += decision.completion_tokens
        result.total_tokens += decision.prompt_tokens + decision.completion_tokens
        # Planner 的成本按与 LLM 客户端一致的粗略单价估算
        result.cost_est += (
            decision.prompt_tokens / 1000 * 0.001 + decision.completion_tokens / 1000 * 0.002
        )


_runtime_singleton: Optional[AgentRuntime] = None


def get_agent_runtime(reload: bool = False) -> AgentRuntime:
    """获取全局 Agent 运行时单例。"""
    global _runtime_singleton
    if _runtime_singleton is None or reload:
        _runtime_singleton = AgentRuntime()
    return _runtime_singleton


def reset_agent_runtime() -> None:
    """丢弃运行时单例（测试用）。"""
    global _runtime_singleton
    _runtime_singleton = None


__all__ = ["AgentResult", "AgentRuntime", "get_agent_runtime", "reset_agent_runtime"]
