"""上下文管理：把原始消息裁剪/压缩成「真正喂给 LLM 的有效输入」。

这是实验主题之一（质量 vs token 开销）。每种策略实现同一个接口：
    build(messages, state) -> effective_messages

策略：
- full: 全量消息（基线）
- sliding: 滑动窗口（只留最近 N 条 + 系统提示）
- summarize: 窗口外的旧消息滚动 LLM 摘要
- hier: 分层记忆（短窗口 + 事实槽注入）
- auto: 自动压缩——消息数或 token 预算超阈值时增量 LLM 摘要，否则退化为 sliding
"""
from __future__ import annotations

from typing import Any

from ..core.types import AgentState, ChatMessage


class ContextPolicy:
    """上下文策略基类。"""

    name = "full"

    def build(self, messages: list[ChatMessage], state: AgentState) -> list[ChatMessage]:
        raise NotImplementedError


class FullContext(ContextPolicy):
    name = "full"

    def build(self, messages: list[ChatMessage], state: AgentState) -> list[ChatMessage]:
        return messages


class SlidingWindowContext(ContextPolicy):
    """滑动窗口：保留系统消息 + 最近 N 条。

    注意：窗口不能切断「assistant(tool_calls) ↔ tool 响应」的配对，
    否则 LLM 会报 tool 消息没有前置 tool_calls。截断后若尾首是 tool 消息，
    向前回溯补上携带 tool_calls 的 assistant 消息。
    """

    name = "sliding"

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def build(self, messages: list[ChatMessage], state: AgentState) -> list[ChatMessage]:
        if len(messages) <= self.window_size:
            return messages
        # 保留 system 开头 + 最近 window_size 条
        head: list[ChatMessage] = []
        tail: list[ChatMessage] = messages
        if messages and messages[0].role == "system":
            head = [messages[0]]
            tail = messages[1:]
        window = tail[-self.window_size:]
        # 修复：若窗口首条是 tool 消息，向前找回带 tool_calls 的 assistant 配对
        if window and window[0].role == "tool":
            idx = len(tail) - self.window_size - 1
            while idx >= 0:
                m = tail[idx]
                if m.role == "assistant" and getattr(m, "tool_calls", None):
                    window = tail[idx:]  # 从配对的 assistant 开始（可能略超 window_size）
                    break
                idx -= 1
        return head + window


class SummarizeContext(ContextPolicy):
    """LLM 摘要：把窗口外旧消息压缩成一条摘要消息（cost 计 agent 桶）。"""

    name = "summarize"

    def __init__(self, window_size: int = 10, llm: Any = None):
        self.window_size = window_size
        self.llm = llm  # 懒注入

    async def build_async(self, messages: list[ChatMessage], state: AgentState) -> list[ChatMessage]:
        from ..core.llm import make_llm
        if len(messages) <= self.window_size or self.llm is None:
            return self._fallback(messages)
        head: list[ChatMessage] = []
        tail: list[ChatMessage] = messages
        if messages and messages[0].role == "system":
            head = [messages[0]]
            tail = messages[1:]
        if len(tail) <= self.window_size:
            return messages
        old = tail[:-self.window_size]
        recent = tail[-self.window_size:]
        llm = self.llm or make_llm("agent")
        summary_text = await llm.acomplete([
            {"role": "system", "content": "你负责压缩对话历史。用一句话概括旧对话的关键信息（用户意图、已确认的事实、未完成的事项）。"},
            {"role": "user", "content": "旧对话：\n" + "\n".join(f"{m.role}: {m.content[:200]}" for m in old)},
        ])
        summary_msg = ChatMessage(role="system", content=f"[历史摘要] {summary_text}")
        return head + [summary_msg] + recent

    def _fallback(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        return SlidingWindowContext(self.window_size).build(messages)


class HierarchicalContext(ContextPolicy):
    """分层记忆：短窗口 + 持久事实槽（工具抽取的用户画像/购物车/最近订单）。

    事实槽由 state["facts"] 提供（可在工具执行后更新），随轮注入 system 消息。
    """

    name = "hier"

    def __init__(self, window_size: int = 8):
        self.window_size = window_size

    def build(self, messages: list[ChatMessage], state: AgentState) -> list[ChatMessage]:
        facts = state.get("facts") or {}
        fact_lines = []
        if facts.get("user_id"):
            fact_lines.append(f"用户ID: {facts['user_id']}")
        if facts.get("recent_order"):
            fact_lines.append(f"最近订单: {facts['recent_order']}")
        if facts.get("active_cart"):
            fact_lines.append(f"购物车: {facts['active_cart']}")
        fact_block = "；".join(fact_lines)

        head: list[ChatMessage] = []
        tail: list[ChatMessage] = messages
        if messages and messages[0].role == "system":
            base_sys = messages[0]
            sys_content = base_sys.content
            if fact_block:
                sys_content = f"{sys_content}\n[用户事实] {fact_block}"
            head = [ChatMessage(role="system", content=sys_content)]
            tail = messages[1:]
        elif fact_block:
            head = [ChatMessage(role="system", content=f"[用户事实] {fact_block}")]
        return head + tail[-self.window_size:]


class AutoContext(ContextPolicy):
    """自动压缩：双触发（消息数 / token 预算），增量 LLM 摘要写回 state。

    设计：
    - 触发条件（任一满足即压缩）：
      a) 消息数 ≥ summary_trigger_messages（settings，默认 12）
      b) 输入 token 估算（含工具 schema）≥ auto_compress_tokens（默认 8000）
    - 压缩动作：把「系统消息之后、最近 window_size 条之前」的旧消息，
      用 LLM 压成一条 [历史摘要] system 消息。
    - 增量：摘要写回 state["summary"]（{text,msg_count,token_estimate}），
      下次只压"上次摘要之后新增的旧消息 + 既有摘要"，避免 O(n²) 重压缩。
    - 未触发：退化为 sliding 窗口。
    - 摘要随 state 持久化（sessions/store 已存完整 state），断线恢复/续聊带摘要。
    """

    name = "auto"

    def __init__(self, window_size: int = 10, llm: Any = None,
                 msg_threshold: int = 12, token_threshold: int = 8000):
        self.window_size = window_size
        self.llm = llm
        self.msg_threshold = msg_threshold
        self.token_threshold = token_threshold

    async def build_async(self, messages: list[ChatMessage], state: AgentState) -> list[ChatMessage]:
        from ..core.llm import make_llm

        # 分离系统头
        head: list[ChatMessage] = []
        tail: list[ChatMessage] = messages
        if messages and messages[0].role == "system":
            head = [messages[0]]
            tail = messages[1:]

        # ---- 触发判定：双条件任一满足即压缩 ----
        # a) 消息数达到阈值（非系统消息 > window 且有足够历史可压）
        # b) token 预算超阈值（只要非系统消息 > window 即可触发，不看消息数门槛）
        over_window = len(tail) > self.window_size
        over_msg_count = len(messages) >= self.msg_threshold
        over_token_budget = self._estimate_tokens(messages) >= self.token_threshold
        if not (over_window and (over_msg_count or over_token_budget)):
            # 未触发：sliding 窗口（保留 tool 配对）
            return SlidingWindowContext(self.window_size).build(messages, state)

        # ---- 增量摘要 ----
        summary = state.get("summary") or {}
        prev_count = int(summary.get("msg_count", 0) or 0)
        # 上次摘要之后新增的旧消息（不含最近 window 条）
        old = tail[:-self.window_size]
        new_old = old[prev_count:]
        if not summary:
            # 首次压缩：全量压 old
            new_old = old
        if new_old:
            llm = self.llm or make_llm("agent")
            summary_text = await self._build_summary(llm, summary, new_old)
            summary = {"text": summary_text, "msg_count": len(old),
                       "token_estimate": self._estimate_tokens(
                           head + ([ChatMessage(role="system", content=summary_text)] if summary_text else []))}
        # 无旧摘要且无 new_old（理论上不达此分支），沿用空摘要
        recent = tail[-self.window_size:]

        if summary.get("text"):
            summary_msg = ChatMessage(role="system", content=f"[历史摘要] {summary['text']}")
        else:
            summary_msg = None

        # 写回 state（随会话持久化）
        state["summary"] = summary

        out = head + ([summary_msg] if summary_msg else []) + recent
        return out

    @staticmethod
    def _estimate_tokens(messages: list[ChatMessage]) -> int:
        from ..core.llm import estimate_tokens
        # 内容 token + 工具 schema 估算（agent 节点会把 tools 附进每次 LLM 请求）
        content = sum(estimate_tokens(m.content) for m in messages)
        try:
            from ..tools.registry import registry
            schemas = registry.openai_schemas()
            tools_tok = sum(estimate_tokens(str(s)) for s in schemas)
        except Exception:
            tools_tok = 0
        return content + tools_tok

    @staticmethod
    async def _build_summary(llm: Any, prev_summary: dict | None, new_messages: list[ChatMessage]) -> str:
        parts: list[dict] = [{"role": "system",
                              "content": "你负责增量压缩对话历史。请合并旧摘要与新增消息，输出一句连贯的关键信息摘要"
                                         "（用户意图、已确认事实、未完成事项）。只输出摘要本体，不要前缀。"}]
        if prev_summary and prev_summary.get("text"):
            parts.append({"role": "user", "content": f"旧摘要：{prev_summary['text']}"})
        if new_messages:
            body = "\n".join(f"{m.role}: {m.content[:200]}" for m in new_messages)
            parts.append({"role": "user", "content": f"新增对话：\n{body}"})
        else:
            parts.append({"role": "user", "content": "无新增，请沿用旧摘要。"})
        text = await llm.acomplete(parts)
        return text.strip()[:500]


class CondenseContext(ContextPolicy):
    """工具返回压缩：把大段工具返回截断（正交叠加在其它策略之上）。"""

    name = "condense"

    def __init__(self, base: ContextPolicy | None = None, max_tool_chars: int = 500):
        self.base = base or SlidingWindowContext()
        self.max_tool_chars = max_tool_chars

    def build(self, messages: list[ChatMessage], state: AgentState) -> list[ChatMessage]:
        msgs = self.base.build(messages, state)
        out = []
        for m in msgs:
            if m.role == "tool" and len(m.content) > self.max_tool_chars:
                out.append(ChatMessage(role="tool", content=m.content[:self.max_tool_chars] + "...[截断]",
                                       name=m.name, tool_call_id=m.tool_call_id))
            else:
                out.append(m)
        return out


# ------------------------------------------------------------------
# 工厂
# ------------------------------------------------------------------
def build_context_policy(name: str, llm: Any = None) -> ContextPolicy:
    from ..core.settings import settings
    name = (name or "full").lower()
    win = int(settings.context_cfg.get("sliding_window_size", 10))
    if name == "sliding":
        return SlidingWindowContext(win)
    if name == "summarize":
        return SummarizeContext(win, llm=llm)
    if name == "hier":
        return HierarchicalContext(max(4, win - 2))
    if name == "condense":
        return CondenseContext(SlidingWindowContext(win))
    if name in ("auto", "automatic", "compress"):
        return AutoContext(
            window_size=win,
            llm=llm,
            msg_threshold=int(settings.context_cfg.get("summary_trigger_messages", 12)),
            token_threshold=int(settings.context_cfg.get("auto_compress_tokens", 8000)),
        )
    return FullContext()
