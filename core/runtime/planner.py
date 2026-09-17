"""Planner：Agent 的决策中枢（B4）。

职责
----
每一步决定「下一步做什么」：

* ``tool``          调用某个工具（给出工具名与参数）
* ``final_answer``  信息足够，直接回答用户
* ``rewrite``       先改写问题（多轮指代消解）再重新决策

两条决策路径
------------
1. **LLM 路径（在线）**：把工具目录（含能力说明、成本、耗时）与已有观察一起给模型，
   要求输出严格 JSON。这是"自主路由"的真正体现——模型在等价工具间做成本/延迟权衡。
2. **规则路径（离线）**：按关键词 + 意图打分选择工具。离线模式没有模型可用，
   但**路由能力必须仍然可见**（否则 CI 与演示都无法验证多工具编排）。

两条路径都遵守同一组硬规则（在提示词与代码里双重保证）：
* 不重复用**相同参数**调用同一个工具（避免死循环）；
* 没有可用工具或工具无法回答时，如实说明而不是编造；
* 需要外部事实（公司制度、行程安排）时必须先用工具，不能凭记忆作答。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.llm import LLMClient, get_llm_client
from core.tools.base import ToolResult
from core.tools.registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 决策结构
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    """Planner 的一步决策。"""

    action: str = "final_answer"           # tool | final_answer | rewrite
    thought: str = ""
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    rewritten: str = ""
    reason: str = ""
    source: str = "rule"                   # llm | rule
    error: Optional[str] = None
    error_type: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """供 trace 与事件使用。"""
        return {
            "action": self.action,
            "thought": self.thought,
            "tool": self.tool,
            "args": self.args,
            "answer": self.answer,
            "rewritten": self.rewritten,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """你是一个 Agent 的决策中枢。你的任务是根据用户问题和已有观察，决定下一步动作。

可用工具：
{tools}

决策规则（严格遵守）：
1. 需要**外部事实**（公司制度、产品文档、运维手册里的信息）→ 调用 knowledge_search。
2. 用户想**规划旅行/出行/旅游行程** → 调用 trip_planner。
3. 已有观察足以回答用户问题 → 直接给出 final_answer，不要再多调工具。
4. **禁止**用完全相同的参数重复调用同一个工具。
5. 涉及"它""这个""上面说的"等指代且缺少上下文时，先用 rewrite 把问题补全。
6. 工具返回「根据现有资料，我无法回答这个问题」时，**必须如实告诉用户资料中没有**，
   绝不允许凭自己的知识编造答案。
7. **只要问题涉及外部事实（景点、制度、文档、数据），即使你认为自己知道答案，
   也必须先调用工具**。凭记忆作答属于严重错误——用户要的是有依据的答案。
   判断标准：如果你准备输出的内容不是来自"已有的工具观察"，
   而是来自你自己的知识，那就必须先调用工具。
8. 只输出 JSON，不要输出 Markdown 代码块或解释性文字。

输出 JSON 格式：
{{"thought": "一句话说明你的判断", "action": "tool 或 final_answer 或 rewrite",
  "tool": "工具名（action=tool 时必填）", "args": {{"参数名": "参数值"}},
  "answer": "action=final_answer 时的回答", "rewritten": "action=rewrite 时的改写后问题"}}
"""

PLANNER_USER_TEMPLATE = """【对话历史】
{history}

【用户问题】
{question}

【已有的工具观察】
{observations}

请输出下一步决策的 JSON。"""

REWRITE_SYSTEM_PROMPT = """你负责把多轮对话里的追问改写成可以独立检索的完整问题。

规则：
1. 把"它""这个""上面那个"等指代替换成历史中提到的具体名词；
2. 不改变用户原意，不添加历史中不存在的信息；
3. 问题本身已经完整时原样返回；
4. 只输出 JSON：{"rewritten": "改写后的问题", "reason": "改写原因，10 字以内"}
"""

REWRITE_USER_TEMPLATE = """【对话历史】
{history}

【用户最新问题】
{question}
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class Planner:
    """决策中枢。"""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        llm: Optional[LLMClient] = None,
        use_llm: bool = True,
    ) -> None:
        self.registry = registry or get_registry()
        self.llm = llm or get_llm_client()
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def decide(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        allow_rewrite: bool = True,
    ) -> Decision:
        """决定下一步动作。

        Args:
            question: 当前（可能已被改写的）问题。
            history: 对话历史（用于改写与上下文）。
            observations: 已完成的工具调用记录，形如
                ``[{"tool": "knowledge_search", "args": {...}, "ok": True, "text": "..."}]``
            allow_rewrite: 是否允许返回 rewrite 动作（避免无限改写）。

        Returns:
            ``Decision``。
        """
        observations = observations or []
        history = history or []

        # ---- LLM 路径 ----
        if self.use_llm and self.llm.available:
            decision = self._decide_with_llm(question, history, observations)
            if decision is not None:
                return self._post_process(decision, observations, allow_rewrite)

        # ---- 规则路径（离线或模型失败）----
        decision = self._decide_with_rules(question, observations)
        return self._post_process(decision, observations, allow_rewrite)

    # ------------------------------------------------------------------
    # LLM 决策
    # ------------------------------------------------------------------
    def _decide_with_llm(
        self,
        question: str,
        history: List[Dict[str, str]],
        observations: List[Dict[str, Any]],
    ) -> Optional[Decision]:
        """调用大模型做决策。"""
        system_prompt = PLANNER_SYSTEM_PROMPT.format(tools=self.registry.describe_for_prompt())
        user_prompt = PLANNER_USER_TEMPLATE.format(
            history=self._format_history(history),
            question=question,
            observations=self._format_observations(observations),
        )
        parsed, result = self.llm.chat_json(system_prompt, user_prompt)
        if not result.ok or not isinstance(parsed, dict):
            logger.warning("Planner 决策失败（%s），回退规则路由", result.error_type)
            return None

        action = str(parsed.get("action") or "").strip()
        tool = str(parsed.get("tool") or "").strip() or None
        args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}

        # 工具名兜底：模型可能给中文名或拼错，用注册表解析一次
        if action == "tool" and tool:
            resolved = self.registry.get(tool)
            if resolved is not None:
                tool = resolved.name
            else:
                logger.info("模型给出的工具 %r 不存在，回退规则路由", tool)
                return None

        if action not in {"tool", "final_answer", "rewrite"}:
            logger.info("模型返回未知 action=%r，回退规则路由", action)
            return None

        # tool 动作必须带工具名，否则视为无效决策
        if action == "tool" and not tool:
            return Decision(
                action="final_answer",
                thought="模型未给出工具名",
                answer="我需要更明确一些才能帮你查。请补充一下你想了解什么，或者想去哪座城市旅行？",
                source="llm",
                error="missing_tool",
                error_type="llm_bad_output",
            )

        return Decision(
            action=action,
            thought=str(parsed.get("thought") or "").strip(),
            tool=tool,
            args=dict(args),
            answer=str(parsed.get("answer") or "").strip(),
            rewritten=str(parsed.get("rewritten") or "").strip(),
            source="llm",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
        )

    # ------------------------------------------------------------------
    # 规则决策（离线兜底）
    # ------------------------------------------------------------------
    def _decide_with_rules(self, question: str, observations: List[Dict[str, Any]]) -> Decision:
        """基于关键词与意图的规则路由。

        设计原则：**先看已有观察**——如果工具已经给出了结论（成功或明确拒答），
        就直接回答，而不是反复调工具。
        """
        text = (question or "").strip()

        # 0) 邮件发送意图优先判断：
        #    它比"继续检索/继续规划"更具体，而且**不能被"已有观察就直接回答"的
        #    快捷路径吃掉**——用户说"把上面的行程发到 xxx@yy.com"时，
        #    上一步的行程结果恰恰是它要发送的内容，不能被当成"问题已经回答完了"。
        #
        #    但有一个例外：**已经成功发过一次就不再重发**。
        #    否则恢复执行时 Planner 会拿着"已发送成功"的结果再拼一封新正文，
        #    参数一变就被当成新操作、再次要求确认，形成死循环。
        already_sent = any(
            item.get("tool") == "send_email" and item.get("ok")
            for item in observations
        )
        email_args = None if already_sent else self._extract_email_args(text, observations)
        if email_args and self.registry.get("send_email") is not None:
            return Decision(
                action="tool",
                thought="识别为发送邮件意图（写操作，需要人工确认）",
                tool="send_email",
                args=email_args,
                source="rule",
            )

        # 1) 已有观察 → 是否可以直接回答
        if observations:
            latest = observations[-1]
            if latest.get("ok"):
                # 知识库明确拒答 → 如实告知，不再尝试其他工具
                if latest.get("refused"):
                    return Decision(
                        action="final_answer",
                        thought="工具已明确表示资料中没有相关信息",
                        answer=(
                            "根据现有资料，我无法回答这个问题。"
                            "如果需要，可以上传相关文档后我再帮你查。"
                        ),
                        source="rule",
                    )
                # 有其他工具成功返回 → 用其结果组织回答
                if latest.get("tool") != "knowledge_search":
                    return Decision(
                        action="final_answer",
                        thought="工具已返回足够信息",
                        answer=self._summarize_observation(latest),
                        source="rule",
                    )
                # knowledge_search 成功且有答案 → 直接引用
                answer = str((latest.get("data") or {}).get("answer") or "").strip()
                if answer:
                    return Decision(
                        action="final_answer",
                        thought="检索已给出答案",
                        answer=answer,
                        source="rule",
                    )

        # 2) 意图打分：选择工具
        #
        # 注意这里的优先级：**有正向信号才用知识库工具**。
        # 早期版本在"明确不是旅行意图"时会把所有问题都路由给 knowledge_search，
        # 结果连"你好"这种寒暄也去检索资料了；现在只有打分 > 0 才走工具，
        # 否则如实告知能力范围并请用户补充需求。
        trip_score = self._score_trip_intent(text)
        knowledge_score = self._score_knowledge_intent(text)

        if trip_score > 0 and trip_score >= knowledge_score:
            if self.registry.get("trip_planner") is not None:
                return Decision(
                    action="tool",
                    thought=f"识别为旅行规划意图（得分 {trip_score}）",
                    tool="trip_planner",
                    args=self._extract_trip_args(text),
                    source="rule",
                )

        if knowledge_score > 0 and self.registry.get("knowledge_search") is not None:
            return Decision(
                action="tool",
                thought=f"识别为知识库查询意图（得分 {knowledge_score}）",
                tool="knowledge_search",
                args={"query": text},
                source="rule",
            )

        # 3) 都没命中：
        #    - 寒暄类短输入 → 给能力引导（把"你好"送进知识库既浪费又体验差）
        #    - 其余输入 → 交给知识库试一次，由它的三层拒答决定"答"还是"明确说没有"
        if knowledge_score <= 0 and self.registry.get("knowledge_search") is not None:
            if not self._looks_like_smalltalk(text):
                return Decision(
                    action="tool",
                    thought="未命中明确意图，交给知识库尝试（检索不到会明确拒答）",
                    tool="knowledge_search",
                    args={"query": text},
                    source="rule",
                )

        return Decision(
            action="final_answer",
            thought=f"未命中任何工具意图（旅行 {trip_score} / 知识库 {knowledge_score}）",
            answer=(
                "我暂时无法判断你想问什么。我可以帮你做两件事：\n"
                "1. 查询知识库里的文档内容（公司制度、产品需求、运维手册等）；\n"
                "2. 规划旅行行程（例如「帮我规划北京三日游」）。\n"
                "请补充一下你的具体需求。"
            ),
            source="rule",
        )

    @staticmethod
    def _score_trip_intent(text: str) -> int:
        """旅行规划意图打分。

        注意两个刻意的设计：

        1. **城市名本身不是旅行意图**："今天北京的天气怎么样" 含"北京"，
           但用户是在问天气而不是要行程。因此城市名必须与"游/玩/行程/规划"
           这类动词同时出现才计分；
        2. **不把"天"当成天数信号**："今天""明天"里的"天"会造成大量误判，
           只有"\\d+天/日"或中文数字+"天/日"才算天数。
        """
        score = 0
        strong = ["旅游", "旅行", "行程", "游玩", "三日游", "自驾", "度假", "攻略", "出行"]
        medium = ["规划", "安排", "出去玩", "去玩", "游玩", "自由行", "跟团"]
        cities = ["北京", "上海", "成都", "西安", "杭州"]

        for keyword in strong:
            if keyword in text:
                score += 3
        for keyword in medium:
            if keyword in text:
                score += 1

        # 明确的天数表达（"3 天""三天"）才是旅行信号
        has_days = bool(
            re.search(r"\d{1,2}\s*[天日]", text)
            or re.search(r"[一二两三四五六七八九十]{1,3}\s*[天日]", text)
        )
        # 城市名 + 旅行动作（或天数）才计分
        has_city = any(city in text for city in cities)
        has_action = any(keyword in text for keyword in ("游", "玩", "行程", "规划", "安排", "度假"))
        if has_city and (has_action or has_days):
            score += 3
        return score

    @staticmethod
    def _score_knowledge_intent(text: str) -> int:
        """知识库查询意图打分。"""
        score = 0
        strong = [
            "制度", "规定", "手册", "文档", "资料", "政策", "条款", "流程", "标准",
            "报销", "年假", "考勤", "保密", "病假", "事假", "离职", "入职", "绩效",
            "需求", "接口", "部署", "备份", "运维", "故障", "排查", "配置", "迁移",
            "天气", "气温", "温度",
        ]
        medium = ["怎么", "如何", "什么", "为什么", "多少", "是否", "能不能", "谁", "哪", "几点"]
        # 纯疑问词：单独出现也表明"用户在问一个事实性问题"
        question_words = [
            "谁", "哪", "几点", "多少", "什么", "为什么", "怎么", "如何", "吗",
            "推荐", "介绍", "解释", "区别", "原因", "好处", "坏处",
        ]
        for keyword in strong:
            if keyword in text:
                score += 3
        for keyword in medium:
            if keyword in text:
                score += 1
        # 疑问句但没提文档 → 也大概率是想查资料
        if text.endswith(("?", "？")):
            score += 1
        # 带疑问词的事实性问题：即使没有命中领域关键词，也值得让知识库试一试。
        # 理由：知识库有"检索不到就明确拒答"的保护，**让工具先试一次比直接反问用户更好**；
        # 而且验收标准要求"文档外问题必须稳定拒答"，这依赖问题先被路由到知识库。
        if not score and any(word in text for word in question_words):
            score = 1
        return score

    @staticmethod
    def _looks_like_smalltalk(text: str) -> bool:
        """判断是不是"寒暄/没有具体诉求"的短输入。

        用途：规则路由在没有意图信号时，对**寒暄**给能力引导（"我可以帮你…"），
        对其余输入则交给知识库工具试一次（工具会拒答或给出资料）。

        为什么这么分：把"你好"送进知识库是浪费且体验差；
        而把"推荐几部科幻电影"直接回成能力引导，会让"文档外问题必须拒答"的
        验收标准失效（用户希望听到明确的"资料里没有"，而不是被反问）。
        """
        text = (text or "").strip()
        if not text:
            return True
        if len(text) > 8:
            return False
        greetings = {
            "你好", "您好", "hi", "hello", "哈喽", "在吗", "在不在", "嗨",
            "谢谢", "多谢", "感谢", "再见", "拜拜", "ok", "好的", "嗯",
            "随便聊聊", "聊聊", "测试", "test",
        }
        lowered = text.lower()
        return lowered in greetings or any(word == lowered for word in greetings)

    @staticmethod
    def _extract_trip_args(text: str) -> Dict[str, Any]:
        """从文本里抽取旅行规划参数（调用工具前先做一次轻量解析）。"""
        from core.tools.trip_tool import (
            parse_budget,
            parse_budget_level,
            parse_days,
            parse_preferences,
        )

        args: Dict[str, Any] = {"question": text}
        days = parse_days(text)
        if days:
            args["days"] = days
        level = parse_budget_level(text)
        if level != "中等":            # 只在明确提到档位时才传，避免覆盖工具默认值
            args["budget_level"] = level
        budget = parse_budget(text)
        if budget:
            args["budget"] = budget
        preferences = parse_preferences(text)
        if preferences:
            args["preferences"] = preferences
        return args

    # ------------------------------------------------------------------
    # 后处理：硬规则兜底
    # ------------------------------------------------------------------
    def _post_process(
        self,
        decision: Decision,
        observations: List[Dict[str, Any]],
        allow_rewrite: bool,
    ) -> Decision:
        """对决策做安全修正（防死循环、防无效动作）。"""
        # 1) 重复调用同一工具 + 同一参数 → 强制收敛为回答
        if decision.action == "tool" and decision.tool:
            for observation in observations:
                if observation.get("tool") == decision.tool and observation.get("args") == decision.args:
                    logger.info("检测到重复调用 %s，强制收敛", decision.tool)
                    return Decision(
                        action="final_answer",
                        thought="该工具已用相同参数调用过，避免重复执行",
                        answer=self._answer_from_observations(observations),
                        source=decision.source,
                    )

        # 2) 不允许改写时，把 rewrite 降级为直接回答
        if decision.action == "rewrite" and not allow_rewrite:
            return Decision(
                action="final_answer",
                thought="已达改写次数上限",
                answer=self._answer_from_observations(observations),
                source=decision.source,
            )

        # 3) 工具不存在 → 如实说明
        if decision.action == "tool" and decision.tool and self.registry.get(decision.tool) is None:
            available = "、".join(self.registry.names()) or "（无）"
            return Decision(
                action="final_answer",
                thought=f"工具 {decision.tool} 不存在",
                answer=f"当前没有可用的工具「{decision.tool}」（可用：{available}）。",
                source=decision.source,
                error="tool_not_found",
                error_type="tool_not_found",
            )

        return decision

    @staticmethod
    def _answer_from_observations(observations: List[Dict[str, Any]]) -> str:
        """没有更多信息可用时，用已有观察组织一个兜底回答。"""
        if not observations:
            return "我暂时没有足够信息回答这个问题，请补充更多细节。"
        latest = observations[-1]
        data = latest.get("data") or {}
        if isinstance(data, dict) and data.get("answer"):
            return str(data["answer"])
        return Planner._summarize_observation(latest)

    @staticmethod
    def _summarize_observation(observation: Dict[str, Any]) -> str:
        """把工具观察压缩成一句回答（工具自带 display 时优先用它）。"""
        display = str(observation.get("display") or "").strip()
        if display:
            return display
        data = observation.get("data")
        if isinstance(data, dict):
            for key in ("answer", "summary", "text"):
                if data.get(key):
                    return str(data[key])
        return "已获取相关结果。"

    @staticmethod
    def _extract_email_args(
        text: str, observations: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """从文本里抽取发送邮件所需的参数。

        触发条件（必须同时满足）：出现发送动作 **且** 出现邮箱地址。
        只出现邮箱或只说"发给我"都不触发——那更可能是知识库问答。

        Returns:
            ``{"to": ..., "subject": ..., "body": ...}``；不满足触发条件时返回 ``None``。
        """
        if not text:
            return None

        email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
        if email_match is None:
            return None

        action_words = ["发到", "发送", "发给", "发邮件", "邮件", "发我", "发一份", "寄到", "发过来"]
        if not any(word in text for word in action_words):
            return None

        to = email_match.group(0)
        # 主题：优先取上一轮工具结果里的内容（"把行程发我" 指的是上一次生成的行程）
        observations = observations or []
        previous = next(
            (item for item in reversed(observations) if item.get("ok") and item.get("display")), None
        )
        if previous is not None:
            body = str(previous.get("display") or "")
            subject = "为你准备的内容"
            if previous.get("tool") == "trip_planner":
                subject = "旅行行程规划"
            elif previous.get("tool") == "knowledge_search":
                subject = "知识库查询结果"
        else:
            # 没有上一轮结果：把用户原话作为正文，避免发空邮件
            body = f"（由 Agent 平台生成）用户请求：{text}"
            subject = "来自 Agent 平台"

        return {"to": to, "subject": subject, "body": body[:4000]}

    @staticmethod
    def _format_history(history: List[Dict[str, str]], limit: int = 6) -> str:
        """渲染对话历史（只保留最近若干轮）。"""
        if not history:
            return "（无历史）"
        lines = []
        for item in history[-limit:]:
            role = "用户" if item.get("role") == "user" else "助手"
            lines.append(f"{role}：{str(item.get('content', ''))[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _format_observations(observations: List[Dict[str, Any]], limit: int = 3) -> str:
        """渲染已有观察（截断，避免上下文过长）。"""
        if not observations:
            return "（还没有调用过工具）"
        lines = []
        for index, observation in enumerate(observations[-limit:], start=1):
            text = str(observation.get("text") or observation.get("display") or "")[:600]
            status = "成功" if observation.get("ok") else f"失败({observation.get('error_type')})"
            lines.append(f"[{index}] {observation.get('tool')}（{status}，参数 {observation.get('args')}）：\n{text}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # 问题改写
    # ------------------------------------------------------------------
    def rewrite(self, question: str, history: List[Dict[str, str]]) -> Decision:
        """多轮指代消解：把追问改写成可独立检索的问题。

        离线模式或没有历史时，原样返回（``action=final_answer`` 之外的语义由调用方处理）。
        """
        question = (question or "").strip()
        if not history or not (self.use_llm and self.llm.available):
            return Decision(action="rewrite", rewritten=question, reason="无历史或离线模式", source="rule")

        parsed, result = self.llm.chat_json(
            REWRITE_SYSTEM_PROMPT,
            REWRITE_USER_TEMPLATE.format(
                history=self._format_history(history), question=question
            ),
        )
        if not result.ok or not isinstance(parsed, dict):
            return Decision(action="rewrite", rewritten=question, reason="改写失败", source="rule")

        rewritten = str(parsed.get("rewritten") or "").strip() or question
        return Decision(
            action="rewrite",
            rewritten=rewritten,
            reason=str(parsed.get("reason") or ""),
            source="llm",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
        )


__all__ = ["Decision", "Planner"]
