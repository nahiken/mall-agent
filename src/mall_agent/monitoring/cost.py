"""成本计算与价格表。

单价以"人民币 / 1k token"计，可在 settings.yaml 的 metrics.cost_per_1k_* 调整。
实际部署可按模型/供应商维护一张价格表，这里先用统一费率 + 按 caller 分桶。
"""
from __future__ import annotations

from ..core.settings import settings


def compute_rmb(input_tokens: int, output_tokens: int) -> float:
    cost_in, cost_out = settings.cost_rates
    return round(input_tokens / 1000 * cost_in + output_tokens / 1000 * cost_out, 6)


def summarize_cost(result: dict) -> dict[str, float]:
    """从 TrialResult dict 汇总各分桶人民币成本。

    说明：`external_cost_rmb` 已包含全部 LLM 事件成本（agent + user_sim + judge + rag），
    因此 total_rmb 直接取其值，不再叠加分桶，避免双重计数。
    """
    c = result.get("cost", {})
    cost_in, cost_out = settings.cost_rates
    agent = round(c.get("agent_in", 0) / 1000 * cost_in + c.get("agent_out", 0) / 1000 * cost_out, 6)
    user_sim = round(c.get("user_sim_in", 0) / 1000 * cost_in + c.get("user_sim_out", 0) / 1000 * cost_out, 6)
    judge = round(c.get("judge_in", 0) / 1000 * cost_in + c.get("judge_out", 0) / 1000 * cost_out, 6)
    external = round(c.get("external_cost_rmb", 0), 6)
    return {
        "agent_rmb": agent,
        "user_sim_rmb": user_sim,
        "judge_rmb": judge,
        "external_rmb": external,
        "total_rmb": external,
    }
