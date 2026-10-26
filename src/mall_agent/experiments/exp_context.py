"""实验：上下文管理（质量 vs token 开销）。

对比 Full / SlidingWindow / Summarize / Hierarchical 策略，
控制变量（同一方案=1，同一黄金子集，同一模型），
产出「每轮输入 token（X）vs Pass@1/rubric（Y）」权衡曲线数据。

用法：python -m mall_agent.cli.exp_context --limit 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mall_agent.core.settings import settings


async def run(policies: list[str], limit: int) -> None:
    from mall_agent.tools.cs_tools import register_cs_tools
    from mall_agent.tools.guide_tools import register_guide_tools
    from mall_agent.tools.admin_tools import register_admin_tools
    register_cs_tools(); register_guide_tools(); register_admin_tools()

    from mall_agent.eval.golden.generator import load_dataset
    from mall_agent.agents.graph_factory import build_agent
    from mall_agent.eval.harness.custom_runner import CustomHarness
    from mall_agent.eval.harness.probe import MockStateProbe
    from mall_agent.eval.metrics.aggregate import aggregate_results
    from mall_agent.mall_api.client import get_mall_client

    tasks = load_dataset()
    if not tasks:
        print("⚠️ 无黄金数据集，先运行: python -m mall_agent.cli.eval --gen")
        return
    if limit:
        tasks = tasks[:limit]
    client = get_mall_client()
    mock = client.ensure_mock_sync()
    probe = MockStateProbe(mock, user_id="1")
    h = CustomHarness(probe=probe)

    print(f"上下文策略实验：{policies} × {len(tasks)} 任务（方案1 单Agent）")
    rows = []
    for policy in policies:
        agent = build_agent(scheme="1", context_policy=policy)
        results = []
        for t in tasks:
            r = await h.run_trial(t, agent, policy, agent_id="1")
            results.append(r)
        agg = aggregate_results(results)
        row = {
            "policy": policy,
            "pass_at_1": agg.get("pass_at_1", 0),
            "subgoal_recall": agg.get("avg_subgoal_recall", 0),
            "avg_input_tokens": agg.get("avg_input_tokens", 0),
            "avg_output_tokens": agg.get("avg_output_tokens", 0),
            "avg_steps": agg.get("avg_steps", 0),
            "success_rate": agg.get("success_rate", 0),
            "cost_rmb": agg.get("total_cost_rmb", 0),
        }
        rows.append(row)
        print(f"  {policy:10s} Pass@1={row['pass_at_1']:.2f} 子目标={row['subgoal_recall']:.2f} "
              f"输入token={row['avg_input_tokens']:.0f} 成本=¥{row['cost_rmb']:.5f}")

    # 落盘结果（供 report 画 Pareto 曲线）
    out = Path(settings.get("data_dir", "data")) / "reports" / "exp_context.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")
    print("\n👉 权衡曲线：把每策略点 (avg_input_tokens, pass_at_1) 画出来，找 Pareto 前沿")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="full,sliding,summarize,hier",
                        help="逗号分隔策略")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args.policy.split(","), args.limit))


if __name__ == "__main__":
    sys.exit(main())
