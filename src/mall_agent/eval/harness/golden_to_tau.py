"""golden_to_tau：把 GoldenTask 转成 τ-Bench 风格 Task。

提供与 τ-Bench 交互的桥接：若未来直接复用 tau_bench 仓库的 Env/Simulator，
可把我们的任务转换成它的格式（user 消息 + 成功条件）。
"""
from __future__ import annotations

from typing import Any

from ..golden.schema import GoldenTask


def to_tau_task(golden: GoldenTask) -> dict[str, Any]:
    """GoldenTask → τ-Bench Task dict。

    τ-Bench Task 核心字段：user（首条消息）、success_conditions（判定）、
    action_success_reward / action_fail_reward / step_reward（可选）。
    """
    success_conditions: list[str] = []
    es = golden.expected_state
    if es.order_created:
        success_conditions.append("The order was successfully created")
    if es.order_cancelled:
        success_conditions.append("The order was cancelled")
    if es.cart_items:
        success_conditions.append("The cart contains the expected items")
    if es.coupon_received:
        success_conditions.append("The coupon was received")
    if not success_conditions:
        success_conditions.append("A satisfactory response was provided to the user")
    # 补充轨迹约束
    for step in golden.golden_tool_flow:
        success_conditions.append(f"The agent called the tool {step.tool}")

    return {
        "task_id": golden.task_id,
        "domain": golden.domain,
        "user": golden.opener,
        "success_conditions": success_conditions,
        "canonical_action_sequence": [s.tool for s in golden.golden_tool_flow],
        "tags": list(golden.safety_tags),
    }


def to_tau_dataset(goldens: list[GoldenTask]) -> list[dict[str, Any]]:
    return [to_tau_task(g) for g in goldens]


if __name__ == "__main__":
    from ..golden.generator import load_dataset
    tasks = load_dataset()
    tau_tasks = to_tau_dataset(tasks)
    print(f"转换 {len(tau_tasks)} 条 → τ-Bench 格式")
    print("示例:", tau_tasks[0])
