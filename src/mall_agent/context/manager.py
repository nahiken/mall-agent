"""ContextManager：按策略裁剪消息 → effective_messages。"""
from __future__ import annotations

from typing import Any

from ..core.types import AgentState, ChatMessage
from .policies import build_context_policy


class ContextManager:
    """上下文管理器：负责生成真正喂给 LLM 的 effective_messages。

    用法：
        cm = ContextManager("sliding")
        eff = await cm.build(messages, state)   # summarize 策略需 await
    """

    def __init__(self, policy: str | None = None, llm: Any = None):
        from ..core.settings import settings
        self.policy_name = policy or settings.default_context_policy
        self.policy = build_context_policy(self.policy_name, llm=llm)

    async def build(self, messages: list[ChatMessage], state: AgentState) -> list[ChatMessage]:
        """生成有效输入消息。同步策略直接调 build；summarize 走 build_async。"""
        build = getattr(self.policy, "build_async", None)
        if build is not None:
            return await build(messages, state)
        return self.policy.build(messages, state)

    @property
    def name(self) -> str:
        return self.policy.name
