"""实验：全矩阵评测（3方案 × 上下文策略 × 双 harness）。

产出最终对比报告（含跨 harness 一致性）。

用法：python -m mall_agent.experiments.exp_matrix --scheme 1,2,3 --policy full,sliding --harness custom,tau --limit 6
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from mall_agent.core.settings import settings


async def run(schemes: list[str], policies: list[str], harnesses: list[str], limit: int) -> list[dict]:
    from mall_agent.tools.cs_tools import register_cs_tools
    from mall_agent.tools.guide_tools import register_guide_tools
    from mall_agent.tools.admin_tools import register_admin_tools
    register_cs_tools(); register_guide_tools(); register_admin_tools()

    from mall_agent.eval.golden.generator import load_dataset
    from mall_agent.eval.harness.probe import MockStateProbe
    from mall_agent.mall_api.client import get_mall_client

    tasks = load_dataset()
    if not tasks:
        print("⚠️ 无黄金数据集，先运行: python -m mall_agent.cli.eval --gen")
        return []
    if limit:
        tasks = tasks[:limit]
    client = get_mall_client()
    mock = client.ensure_mock_sync()
    probe = MockStateProbe(mock, user_id="1")
    print(f"全矩阵: {len(schemes)} 方案 × {len(policies)} 策略 × {len(harnesses)} harness × {len(tasks)} 任务")

    results: list[dict] = []
    for scheme in schemes:
        from mall_agent.agents.graph_factory import build_agent
        for policy in policies:
            agent = build_agent(scheme=scheme, context_policy=policy)
            for harness in harnesses:
                for task in tasks:
                    if harness == "custom":
                        from mall_agent.eval.harness.custom_runner import CustomHarness
                        h = CustomHarness(probe=probe)
                        r = await h.run_trial(task, agent, policy, agent_id=scheme)
                        r["meta"]["task_domain"] = task.domain
                    else:
                        from mall_agent.eval.harness.tau_adapter import TauStyleAgent, TauStyleEnv
                        env = TauStyleEnv(tasks, probe)
                        tau_agent = TauStyleAgent(agent)
                        idx = next(i for i, t in enumerate(env.tasks) if t.task_id == task.task_id)
                        r = await tau_agent.solve(env, idx, context_policy=policy)
                        r["harness"] = "tau"
                        r["agent_id"] = scheme
                        r["golden_task_id"] = task.task_id
                        r["meta"]["task_domain"] = task.domain
                    print(f"  [{scheme}|{policy}|{harness}] {task.task_id}: {r['outcome']} steps={r['steps']}")
                    results.append(r)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="1,2,3")
    parser.add_argument("--policy", default="sliding")
    parser.add_argument("--harness", default="custom,tau")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    results = asyncio.run(run(
        args.scheme.split(","), args.policy.split(","), args.harness.split(","), args.limit))

    from mall_agent.eval.report import generate_report, save_raw, save_report
    report = generate_report(results, title="mall-agent 全矩阵评测")
    print("\n" + "=" * 40)
    print(report[:1500])
    save_raw(results, "full_matrix_results.json")
    p = save_report(report, "full_matrix_report.md")
    print(f"\n✅ 全矩阵报告已保存: {p}")

    # 跨 harness 一致性（同一任务两 harness 的 outcome 是否一致）
    if len(set(r.get("harness") for r in results)) > 1:
        print("\n=== 跨 harness 一致性 ===")
        by_task: dict = {}
        for r in results:
            by_task.setdefault((r["golden_task_id"], r["agent_id"], r["context_policy"]), []).append(r)
        agree = 0
        total = 0
        for key, rs in by_task.items():
            if len(rs) >= 2:
                total += 1
                if rs[0]["outcome"] == rs[1]["outcome"]:
                    agree += 1
        if total:
            print(f"双 harness 结果一致率: {agree}/{total} = {agree / total:.2f}")


if __name__ == "__main__":
    sys.exit(main())
