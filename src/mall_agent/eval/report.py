"""评测报告生成：按维度输出聚合表（效率/延迟/成本/发散/质量/轨迹/安全）。

输出 Markdown 报告到 data/reports/。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics.aggregate import aggregate_results


def _fmt(v: Any, suffix: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}{suffix}"
    return f"{v}{suffix}"


def render_table(agg_by_agent: dict[str, dict], metrics: list[tuple[str, str]]) -> str:
    """按 agent 渲染指标表。"""
    agents = list(agg_by_agent.keys())
    if not agents:
        return "（无数据）"
    header = "| 指标 | " + " | ".join(agents) + " |"
    sep = "|------|" + "|".join(["------"] * len(agents)) + "|"
    lines = [header, sep]
    for label, key in metrics:
        row = [label]
        for a in agents:
            val = agg_by_agent[a]
            # 支持嵌套 key 如 latency.p95
            node: Any = val
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            row.append(_fmt(node))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_report(results: list[dict], title: str = "评测报告") -> str:
    """生成 Markdown 报告。"""
    from .metrics.aggregate import critical_pass, group_by_agent

    agg = group_by_agent(results)
    lines = [f"# {title}", ""]
    lines.append(f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- 总任务数: {len(results)}")
    lines.append("")

    # Pass^k：关键任务全过（缺省全部任务）
    cp = critical_pass(results)
    lines.append(f"- **Pass^k（关键任务全过）**：{cp['critical_pass_rate']:.3f} "
                 f"（{cp['critical_pass_rate'] == 1.0 and '[OK] 全过' or '[X] 未全过'}，"
                 f"{cp['critical_tasks']} 个关键任务）")
    lines.append("")

    lines.append("## 方案对比")
    lines.append("")
    lines.append(render_table(agg, [
        ("成功率", "success_rate"), ("Pass@1", "pass_at_1"),
        ("rubric 加权分", "avg_rubric"), ("功能正确性", "avg_functional_correct"),
        ("事实一致性", "avg_fact_consistency"),
        ("答案grounding", "avg_answer_grounding"), ("grounded率", "grounded_rate"),
        ("轨迹匹配", "avg_tool_flow_match"),
        ("参数准确度", "avg_param_accuracy"),
        ("子目标召回", "avg_subgoal_recall"),
        ("平均步数", "avg_steps"), ("步数超限失败率", "max_steps_fail_rate"),
        ("平均工具调用", "avg_tool_calls"), ("平均绕路", "avg_detour"),
        ("平均输入token", "avg_input_tokens"), ("平均输出token", "avg_output_tokens"),
        ("超token预算率", "over_budget_rate"),
        ("每任务成本(¥)", "avg_cost_rmb_per_task"),
        ("总延迟P95(ms)", "latency.p95"), ("总延迟P99(ms)", "latency.p99"),
        ("TTFT P95(ms)", "latency.p95_ttft"), ("TTFT P99(ms)", "latency.p99_ttft"),
        ("平均TPOT(ms)", "latency.avg_tpot_ms"),
        ("权限拦截率", "permission.block_rate"),
    ]))
    lines.append("")

    # 失败原因分布
    lines.append("### 失败原因分布")
    lines.append("")
    for a in agg:
        fr = agg[a].get("fail_reasons", {})
        fr_txt = "；".join(f"{k or 'success'}: {v}" for k, v in fr.items()) if fr else "（无失败）"
        lines.append(f"- 方案 {a}: {fr_txt}")
    lines.append("")

    # 业务可用性代理指标（有用性 / 投诉风险 / 人工兜底）
    from .metrics import safety as safety_metrics
    by_agent_raw: dict[str, list[dict]] = {}
    for r in results:
        by_agent_raw.setdefault(r.get("agent_id"), []).append(r)
    lines.append("### 业务可用性代理（评审观测用）")
    lines.append("")
    for a, rs in by_agent_raw.items():
        prof = safety_metrics.helpfulness_risk_profile(rs)
        lines.append(
            f"- 方案 {a}: 有用性={prof['helpfulness_score']:.3f} "
            f"投诉风险={prof['complaint_risk']}（{prof['complaint_risk_rate']:.2f}） "
            f"人工兜底率={prof['human_fallback_rate']:.2f}（{prof['human_fallback_tasks']} 任务）")
    lines.append("")

    # 工具耗时 Top
    lines.append("### 工具耗时 Top（avg_ms，跨任务聚合）")
    lines.append("")
    for a in agg:
        tl = agg[a].get("tool_latency_top", {})
        if not tl:
            continue
        tl_txt = "；".join(f"{k}: {v['avg_ms']}ms×{v['calls']}" for k, v in list(tl.items())[:6])
        lines.append(f"- 方案 {a}: {tl_txt}")
    lines.append("")

    # 分域
    for domain in ("customer_service", "shopping_guide", "mixed"):
        dom_results = [r for r in results if r.get("meta", {}).get("task_domain", "") == domain
                       or r.get("golden_task_id", "").startswith(
                           {"customer_service": "CS-", "shopping_guide": "GD-", "mixed": "MX-"}.get(domain, "ZZ"))]
        if dom_results:
            lines.append(f"## {domain} 子集")
            lines.append("")
            lines.append(render_table(group_by_agent(dom_results), [
                ("成功率", "success_rate"), ("平均步数", "avg_steps"),
                ("平均工具调用", "avg_tool_calls"), ("平均输入token", "avg_input_tokens"),
            ]))
            lines.append("")

    # 失败明细
    failed = [r for r in results if r.get("outcome") != "success"]
    if failed:
        lines.append("## 失败任务明细")
        lines.append("")
        for r in failed[:20]:
            lines.append(f"- `{r.get('golden_task_id')}` agent={r.get('agent_id')} "
                         f"outcome={r.get('outcome')} steps={r.get('steps')} "
                         f"tools={[t['name'] for t in r.get('trajectory', []) if t['type'] == 'tool'][:3]}")
        lines.append("")

    return "\n".join(lines)


def save_report(report: str, filename: str | None = None) -> Path:
    from mall_agent.core.settings import settings
    reports_dir = Path(settings.get("data_dir", "data")) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    p = reports_dir / (filename or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    p.write_text(report, encoding="utf-8")
    return p


def save_raw(results: list[dict], filename: str | None = None) -> Path:
    from mall_agent.core.settings import settings
    reports_dir = Path(settings.get("data_dir", "data")) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    p = reports_dir / (filename or "raw_results.json")
    p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
