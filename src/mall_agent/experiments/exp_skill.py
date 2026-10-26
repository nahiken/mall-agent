"""实验：Skill 懒加载（token 开销 vs 质量）。

对比三种子形态 + 跨方案对比：
- full_tools（方案1 全工具基线）
- lazy_rule（方案3 + 规则路由，零 LLM）
- lazy_llm（方案3 + LLM 路由，1 次调用）

量化：每轮工具 schema token 差、wrong_tool_rate、router_error_rate、混合任务切换成功率。

用法：python -m mall_agent.cli.exp_skill --limit 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mall_agent.core.settings import settings


async def run(limit: int) -> None:
    from mall_agent.tools.cs_tools import register_cs_tools
    from mall_agent.tools.guide_tools import register_guide_tools
    from mall_agent.tools.admin_tools import register_admin_tools
    register_cs_tools(); register_guide_tools(); register_admin_tools()

    from mall_agent.tools.registry import registry
    from mall_agent.agents.skills.skill_base import build_skills
    from mall_agent.eval.golden.generator import load_dataset
    from mall_agent.eval.harness.custom_runner import CustomHarness
    from mall_agent.eval.harness.probe import MockStateProbe
    from mall_agent.eval.metrics.aggregate import aggregate_results
    from mall_agent.mall_api.client import get_mall_client

    # 1. 工具 schema token 差（不烧 LLM，纯统计）
    print("=== 工具 schema token 对比 ===")
    full = registry.filter_by_roles({"cs", "guide", "supervisor"})
    skills = build_skills()
    cs_schema = full.filter_by_skill("cs", skills["cs"].tool_names)
    guide_schema = full.filter_by_skill("guide", skills["guide"].tool_names)
    full_tok = full.schema_token_estimate()
    cs_tok = cs_schema.schema_token_estimate()
    guide_tok = guide_schema.schema_token_estimate()
    print(f"  全工具: {full_tok} token（基线）")
    print(f"  仅客服skill: {cs_tok} token（省 {100 - cs_tok * 100 // full_tok}%）")
    print(f"  仅导购skill: {guide_tok} token（省 {100 - guide_tok * 100 // full_tok}%）")

    # 2. 实际评测对比（rule 路由 vs full_tools）
    print("\n=== 跨方案质量对比（黄金子集） ===")
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

    from mall_agent.agents.graph_factory import build_agent
    rows = []
    for scheme, label in [("1", "方案1 全工具"), ("3", "方案3 rule懒加载")]:
        agent = build_agent(scheme=scheme, context_policy="sliding")
        results = [await h.run_trial(t, agent, "sliding", agent_id=scheme) for t in tasks]
        agg = aggregate_results(results)
        row = {"scheme": scheme, "label": label,
               "success_rate": agg.get("success_rate", 0),
               "avg_input_tokens": agg.get("avg_input_tokens", 0),
               "avg_steps": agg.get("avg_steps", 0),
               "avg_tool_calls": agg.get("avg_tool_calls", 0)}
        rows.append(row)
        print(f"  {label}: 成功率={row['success_rate']:.2f} 输入token={row['avg_input_tokens']:.0f} "
              f"步数={row['avg_steps']:.1f} 工具={row['avg_tool_calls']:.1f}")

    out = Path(settings.get("data_dir", "data")) / "reports" / "exp_skill.json"
    out.write_text(json.dumps({
        "schema_tokens": {"full": full_tok, "cs": cs_tok, "guide": guide_tok,
                          "cs_save_pct": 100 - cs_tok * 100 // full_tok,
                          "guide_save_pct": 100 - guide_tok * 100 // full_tok},
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args.limit))


if __name__ == "__main__":
    sys.exit(main())
