"""τ-Bench 风格适配：把 MallAgent 包装成 τ-Bench 的 Agent.solve() 接口。

借鉴 τ-Bench 的核心模式：
- Agent.solve(env, task_index) → SolveResult
- env 提供 tools / tasks / user simulator / rules（成功判定）
但不照搬其 airline/retail data 复制模型 —— 用我们的黄金数据集 + Mock 状态。

这样既保留"τ-Bench 风格评测"（agent 与 env 解耦、多轮对话、结果判定），
又复用 mall-agent 已有设施，双 harness 共享 TrialResult。
"""
from __future__ import annotations

from typing import Any

from ..golden.schema import GoldenTask


class TauStyleTask:
    """τ-Bench 风格任务包装：把 GoldenTask 暴露成类似 tau_bench Task 的结构。"""

    def __init__(self, golden: GoldenTask):
        self.golden = golden
        self.user = golden.opener            # 首条用户消息
        self.task_id = golden.task_id
        self.tools = [s.tool for s in golden.golden_tool_flow]
        # 成功判定统一走 env.evaluate → probe.check(expected_state)，
        # 不再维护独立的 success_conditions 列表。


class TauStyleEnv:
    """τ-Bench 风格 Env：绑定黄金数据集 + 状态探针。"""

    def __init__(self, tasks: list[GoldenTask], probe: Any):
        self.tasks = [TauStyleTask(t) for t in tasks]
        self._probe = probe

    async def reset(self, task_index: int) -> str:
        """返回首条用户消息。"""
        return self.tasks[task_index].user

    async def evaluate(self, task_index: int, state: dict) -> dict:
        """判定任务是否成功（用 expected_state + probe）。"""
        golden = self.tasks[task_index].golden
        ok, reason = await self._probe.check(golden.expected_state)
        return {"success": ok, "reason": reason}


class TauStyleAgent:
    """τ-Bench 风格 Agent：把 MallAgent 包成 solve() 接口。"""

    def __init__(self, mall_agent: Any, max_steps: int = 30):
        self.agent = mall_agent
        self.max_steps = max_steps

    async def solve(self, env: TauStyleEnv, task_index: int, context_policy: str | None = None) -> dict:
        """跑单个任务，返回统一结果（近似 SolveResult）。

        context_policy 必须与 agent 实际构建策略一致（由调用方传入）；
        传 None 时回退到 "sliding"（保持向后兼容）。
        """
        policy = context_policy or "sliding"
        task = env.tasks[task_index]
        from mall_agent.monitoring.monitor import monitor

        monitor.begin_trial(harness="tau", agent_id=self.agent.scheme,
                            golden_task_id=task.task_id, context_policy=policy)
        # 多轮：agent 回复 → 脚本用户回复 → agent 回复 ...
        opener = await env.reset(task_index)
        state: dict[str, Any] = {
            "session_id": f"{task.task_id}-tau",
            "role": "cs" if task.golden.domain in ("customer_service", "mixed") else "guide",
            "user_id": str(task.golden.initial_state.get("user_id", "1")),
            "scheme": self.agent.scheme,
            "messages": [],
            "effective_messages": [],
            "context_policy": policy,
            "tool_records": [], "steps": 0, "max_steps": self.max_steps,
            "pending_tool_calls": [], "permission_events": [], "flags": {},
        }
        from mall_agent.core.types import ChatMessage
        state["messages"].append(ChatMessage(role="user", content=opener))

        outcome = "fail"
        fail_reason: str | None = None
        from ..metrics.scoring import FAIL_REASONS
        for turn in range(1, self.max_steps):
            try:
                result = await self.agent.app.ainvoke(
                    dict(state),
                    config={"recursion_limit": min(self.max_steps * 2 + 10, 60)},
                )
            except Exception as e:
                result = dict(state)
                outcome = "error"
                fail_reason = f"{FAIL_REASONS['ERROR']}: {type(e).__name__}"
                break
            state.update(result)
            # 判定
            eval_res = await env.evaluate(task_index, state)
            if eval_res["success"]:
                outcome = "success"
                fail_reason = None
                break
            if state.get("steps", 0) >= 15:
                outcome = "fail"
                fail_reason = FAIL_REASONS["MAX_STEPS"]   # 效率作为正确性的一部分
                break
            # 脚本用户回复
            replies = task.golden.scripted_user_replies
            if turn - 1 < len(replies):
                state["messages"].append(ChatMessage(role="user", content=replies[turn - 1]))
            else:
                # 脚本回复用完仍未达成 → 判定失败（probe 最终再确认一次）
                fail_reason = FAIL_REASONS["SCRIPT"]
                break

        # 最终用 probe 确认（脚本结束后状态可能已达成）
        from ..metrics.scoring import compute_success_signal
        last_reply = self.agent._extract_reply(state)
        final_eval = await env.evaluate(task_index, state)
        if final_eval["success"]:
            outcome = "success"
            fail_reason = None
        elif outcome == "fail":
            fail_reason = fail_reason or FAIL_REASONS["PROBE"]
        elif outcome == "error":
            fail_reason = fail_reason or FAIL_REASONS["ERROR"]

        success_signal = await compute_success_signal(env._probe, task.golden, state, last_reply, outcome)

        # 期望步数：有黄金流用其长度（≥1），无则 0（不惩罚）
        expected_steps = max(1, len(task.golden.golden_tool_flow)) if task.golden.golden_tool_flow else 0

        return monitor.end_trial(
            outcome=outcome,
            success_signal=success_signal,
            steps=state.get("steps", 0),
            expected_steps=expected_steps,
            meta={"harness": "tau", "task_type": task.golden.task_type,
                  "fail_reason": fail_reason},
        )
