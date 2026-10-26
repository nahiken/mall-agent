"""extract_zh：中文/域无关的成功判定。

τ-Bench 官方 extract 是英文 + 域特定（retail/airline），中文任务必须重写。
本模块：基于 GoldenTask.expected_state + 状态探针做判定，与语言、域解耦。
"""
from __future__ import annotations

from typing import Any

from ..golden.schema import ExpectedState, GoldenTask


class ExtractZh:
    """中文成功判定器：校验 agent 轨迹后的业务状态是否达成 expected_state。"""

    def __init__(self, probe: Any):
        self.probe = probe  # MockStateProbe / SQLDataProbe

    async def extract(self, state: dict) -> dict:
        """从 agent 终态提取可判定的业务事实。"""
        tool_names = [r.name for r in state.get("tool_records", [])]
        return {
            "tools_called": tool_names,
            "steps": state.get("steps", 0),
        }

    async def get_success(self, task: GoldenTask, extracted: dict) -> tuple[bool, str]:
        """判定任务是否成功：expected_state 是否被探针满足。"""
        ok, reason = await self.probe.check(task.expected_state)
        return ok, reason

    async def get_reward(self, task: GoldenTask, state: dict) -> dict:
        """返回奖励信号（Pass@1 + 子目标召回）。"""
        extracted = await self.extract(state)
        success, reason = await self.get_success(task, extracted)
        executed = set(extracted["tools_called"])
        expected = {s.tool for s in task.golden_tool_flow}
        subgoal_recall = round(len(executed & expected) / len(expected), 2) if expected else 1.0
        return {
            "success": success,
            "reason": reason,
            "pass_at_1": 1.0 if success else 0.0,
            "subgoal_recall": subgoal_recall,
        }
