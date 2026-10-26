"""评测 harness 基座：统一 TrialResult 生产 + 用户模拟器接口。

原则：harness 只产"原始 per-trial 数据"（TrialResult），
聚合只在 eval/metrics/* 层做。这样双 harness（custom/tau）不打架。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..golden.schema import GoldenTask


class BaseUserSimulator(ABC):
    """用户模拟器接口：根据 agent 回复生成用户下一条回复。"""

    @abstractmethod
    async def next_reply(self, agent_reply: str, task: GoldenTask, turn: int) -> str | None:
        """返回用户下一条消息；返回 None 表示对话结束。"""
        ...


class ScriptedSimulator(BaseUserSimulator):
    """脚本化模拟器：从 task.scripted_user_replies 顺序取回复。

    零 LLM、零额外成本、确定性高 —— 作为主评测模拟器。
    """

    async def next_reply(self, agent_reply: str, task: GoldenTask, turn: int) -> str | None:
        replies = task.scripted_user_replies
        idx = turn - 1  # 第 1 轮 agent 回复后对应第 1 条脚本回复
        if idx < len(replies):
            return replies[idx]
        return None


class LLMPersonaSimulator(BaseUserSimulator):
    """LLM persona 模拟器：用 persona + 对话历史生成下一条回复（耗 token，计 user_sim 桶）。"""

    def __init__(self, llm: Any = None):
        self.llm = llm  # 用 make_llm("user_sim") 惰性注入

    async def next_reply(self, agent_reply: str, task: GoldenTask, turn: int) -> str | None:
        if self.llm is None:
            from mall_agent.core.llm import make_llm
            self.llm = make_llm("user_sim")
        persona = task.persona or "你是商城普通用户，说话自然简洁。"
        prompt = (
            f"{persona}\n"
            f"任务背景：{task.opener}\n"
            f"Agent 刚才的回复：{agent_reply}\n"
            f"请以用户身份回复 Agent（一句话）。如果事情已解决或你想结束，回复'结束'。"
        )
        reply = await self.llm.acomplete([{"role": "user", "content": prompt}])
        if "结束" in reply:
            return None
        return reply


class HybridSimulator(BaseUserSimulator):
    """混合模拟器：前 max_scripted 轮用脚本回复（保证任务推进），之后用 LLM persona。

    兼顾确定性（脚本）与自然度（persona）：脚本保证关键轮次不跑偏，
    后续轮由 persona 自然延续，避免脚本耗尽就提前结束。
    """

    def __init__(self, max_scripted: int = 1, persona_llm: Any = None):
        self.max_scripted = max_scripted
        self._persona = LLMPersonaSimulator(persona_llm)

    async def next_reply(self, agent_reply: str, task: GoldenTask, turn: int) -> str | None:
        replies = task.scripted_user_replies
        if turn - 1 < min(self.max_scripted, len(replies)):
            return replies[turn - 1]
        return await self._persona.next_reply(agent_reply, task, turn)


class BaseHarness(ABC):
    """评测 harness 抽象。"""

    harness_name = "base"

    @abstractmethod
    async def run_trial(self, task: GoldenTask, agent: Any, context_policy: str) -> dict:
        """跑单条任务，返回 TrialResult dict。"""
        ...
