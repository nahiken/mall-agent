"""自建评测 runner：批量跑 agent × 黄金任务，产出统一 TrialResult。

流程：
1. begin_trial 开启 monitor
2. 用户 opener → agent 多轮 → ScriptedSimulator 回脚本回复
3. 每轮 agent.ainvoke 累积 state
4. end_trial 产出 TrialResult + 断言判定（probe 读业务状态）
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..golden.schema import GoldenTask
from .base import BaseHarness


class CustomHarness(BaseHarness):
    harness_name = "custom"

    def __init__(self, simulator: Any = None, probe: Any = None, max_turns: int = 6,
                 simulator_mode: str = "scripted", trial_timeout: float = 180.0,
                 reset_state: bool = True):
        # 单轮 agent 调用超时（秒）。上游 LLM/RAG 无响应时，超时记 error 而不是拖死整批。
        self.trial_timeout = trial_timeout
        # 每题开跑前重置 Mock 用户态：防跨题状态泄漏 + 保证「查订单」类任务有数据可查
        self.reset_state = reset_state
        # simulator 显式传入时优先；否则按 mode 选型
        if simulator is not None:
            self.simulator = simulator
        else:
            from .base import HybridSimulator, LLMPersonaSimulator, ScriptedSimulator
            if simulator_mode == "persona":
                self.simulator = LLMPersonaSimulator()
            elif simulator_mode == "hybrid":
                # hybrid：首轮脚本回复（保证任务推进），后续轮用 persona
                self.simulator = HybridSimulator()
            else:
                self.simulator = ScriptedSimulator()
        self.simulator_mode = simulator_mode
        self.probe = probe            # 业务状态探针（probe.py）
        self.max_turns = max_turns

    async def run_trial(self, task: GoldenTask, agent: Any, context_policy: str, agent_id: str = "1") -> dict:
        from mall_agent.monitoring.monitor import monitor
        from ..metrics.scoring import FAIL_REASONS
        from .probe import has_assertions

        # 每题重置用户态（夹具），否则上一题加的购物车/领的券会让下一题的断言白捡通过
        if self.reset_state and self.probe is not None and hasattr(self.probe, "reset"):
            self.probe.reset()
        # 会话用户必须与 mock 当前登录用户一致：套用 "1" 会让数据级权限判成
        # "订单不属于当前用户"（mock 的 uid 是 sso 自增出来的），表现为所有查单/退单被拦。
        eval_uid = getattr(self.probe, "_uid", None) or task.initial_state.get("user_id", "1")

        # 空断言任务（expected_state 全 None）不能用 probe 判通过，否则第一轮就 break
        task_has_assert = has_assertions(task.expected_state)

        monitor.begin_trial(harness="custom", agent_id=agent_id,
                            golden_task_id=task.task_id, context_policy=context_policy)

        state: dict[str, Any] = {
            "session_id": f"{task.task_id}-{monitor.current.trial_id}",
            "role": "cs" if task.domain in ("customer_service", "mixed") else "guide",
            "user_id": str(eval_uid),
            "scheme": agent.scheme,
            "messages": [],
            "effective_messages": [],
            "context_policy": context_policy,
            "tool_records": [],
            "steps": 0,
            "max_steps": 15,
            "pending_tool_calls": [],
            "permission_events": [],
            "flags": {},
            "done": False,
            "rag_enabled": getattr(agent, "rag_enabled", True),
            # 输出风格轴（改动七/T11）：由 build_agent 注入 agent，state 携带供 prompt 装配
            "prompt_style": getattr(agent, "prompt_style", None),
        }

        # 用户 opener
        from mall_agent.core.types import ChatMessage
        state["messages"].append(ChatMessage(role="user", content=task.opener))

        outcome = "success"
        last_reply = ""
        turns = 0
        fail_reason: str | None = None
        for turn in range(1, self.max_turns + 1):
            turns = turn
            # agent 跑一轮（其内部会做 agent→tools 直到无工具调用或达 max_steps）
            try:
                result = await agent.app.ainvoke(
                    dict(state),
                    config={"recursion_limit": state.get("max_steps", 15) * 2 + 10},
                )
            except Exception as e:
                result = dict(state)
                outcome = "error"
                fail_reason = f"{FAIL_REASONS['ERROR']}: {type(e).__name__}"
            state.update(result)
            last_reply = agent._extract_reply(state)

            # 检查是否达成 expected_state（仅当任务带真断言）
            if self.probe is not None and task_has_assert:
                ok, reason = await self.probe.check(task.expected_state)
                if ok:
                    outcome = "success"
                    fail_reason = None
                    break

            # 检查发散：超过 max_steps → 效率作为正确性的一部分（显式失败）
            max_steps = state.get("max_steps", 15)
            if state.get("steps", 0) >= max_steps:
                outcome = "fail"
                fail_reason = FAIL_REASONS["MAX_STEPS"]
                break

            # 用户模拟器回一条
            user_reply = await self.simulator.next_reply(last_reply, task, turn)
            if user_reply is None:
                # 用户结束对话
                if outcome == "error":
                    fail_reason = FAIL_REASONS["ERROR"]
                break
            state["messages"].append(ChatMessage(role="user", content=user_reply))

        # 用 probe 做最终判定（无断言的任务跳过：空断言恒 True，不构成证据）
        if self.probe is not None and task_has_assert and outcome == "success":
            ok, _ = await self.probe.check(task.expected_state)
            if ok:
                outcome = "success"
                fail_reason = None
            else:
                outcome = "fail"
                fail_reason = FAIL_REASONS["PROBE"]

        # 成功信号：统一由 scoring.compute_success_signal 构造
        from ..metrics.scoring import compute_success_signal
        success_signal = await compute_success_signal(self.probe, task, state, last_reply, outcome)

        # 期望步数：有黄金流则用其长度（≥1，避免 0 步任务被算成绕路），无则 0（不惩罚）
        expected_steps = max(1, len(task.golden_tool_flow)) if task.golden_tool_flow else 0

        result = monitor.end_trial(
            outcome=outcome,
            success_signal=success_signal,
            steps=state.get("steps", 0),
            expected_steps=expected_steps,
            meta={"model": "deepseek", "turns": turns, "task_type": task.task_type,
                  "fail_reason": fail_reason, "has_assertions": task_has_assert},
        )
        # 多轮时 last_reply 往往只是收尾客套话，答案在第一轮；判答必须看全部 assistant 回复
        replies = [m.content for m in state.get("messages", [])
                   if getattr(m, "role", "") == "assistant" and getattr(m, "content", "")]
        reply_all = chr(10).join(replies) if replies else last_reply

        # 答案级判答所需上下文（改动五.2）：reply + 工具记录 + 任务答案字段
        result["answer_ctx"] = {
            "question": task.opener,
            "reference_answer": getattr(task, "reference_answer", ""),
            "key_facts": list(getattr(task, "key_facts", []) or []),
            "rag_hub": getattr(task, "rag_hub", None),
            "category": getattr(task, "category", ""),
            "depends_on_private_kb": bool(getattr(task, "depends_on_private_kb", False)),
            "reply": last_reply,
            "reply_all": reply_all,
            "tool_records": [
                {"name": r.name, "result": r.result} for r in state.get("tool_records", [])
            ],
            "task_id": task.task_id,
            "rag_enabled": bool(state.get("rag_enabled", True)),
        }
        return result
