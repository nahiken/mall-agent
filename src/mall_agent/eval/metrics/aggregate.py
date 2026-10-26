"""指标计算：对 TrialResult 列表做聚合（双 harness 共享的唯一聚合层）。

P95/P99 在聚合层对 per-trial total_ms 序列计算，harness 只产原始值。
额外聚合：失败原因分布 / Pass^k / rubric 加权分 / token 预算 / 工具耗时。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any


def _pct(arr: list[float], p: float) -> float | None:
    s = sorted(arr)
    if not s:
        return None
    idx = min(len(s) - 1, int(len(s) * p))
    return round(s[idx], 1)


# 单任务 token 预算（可被 aggregate_results(token_budget=...) 覆盖）
DEFAULT_TOKEN_BUDGET = 20000


def aggregate_results(results: list[dict], token_budget: int | None = None) -> dict[str, Any]:
    """聚合一组 TrialResult（通常按 agent_id + context_policy 分组）。"""
    n = len(results)
    if n == 0:
        return {}
    budget = token_budget if token_budget is not None else DEFAULT_TOKEN_BUDGET
    outcomes = [r.get("outcome") for r in results]
    totals_ms = [r["latency_ms"]["total"] for r in results if r.get("latency_ms", {}).get("total")]
    ttfts = [r["latency_ms"]["ttft"] for r in results if r.get("latency_ms", {}).get("ttft")]
    tpots = [r["latency_ms"]["tpot"] for r in results if r.get("latency_ms", {}).get("tpot")]

    total_in = sum(r["cost"].get("agent_in", 0) for r in results)
    total_out = sum(r["cost"].get("agent_out", 0) for r in results)
    user_sim_in = sum(r["cost"].get("user_sim_in", 0) for r in results)
    tool_calls = sum(r["cost"].get("tool_calls", 0) for r in results)
    external = sum(r["cost"].get("external_cost_rmb", 0) for r in results)

    blocked = sum(r["permission"].get("blocked_attempts", 0) for r in results)
    attempts = sum(r["permission"].get("total_attempts", 0) for r in results)

    # ---- 新增：失败原因分布 ----
    fail_reasons = Counter((r.get("meta") or {}).get("fail_reason") for r in results)

    # ---- 新增：rubric / 事实一致性 / 功能正确性 ----
    rubric = [r["success_signal"].get("rubric_score", 0) for r in results]
    fact = [r["success_signal"].get("fact_consistency", 0) for r in results]
    fc = [r["success_signal"].get("functional_correct", 0) for r in results]
    tf = [r["success_signal"].get("tool_flow_match", 0) for r in results]
    pa = [r["success_signal"].get("param_accuracy", 0) for r in results]

    # ---- 新增：答案级 grounding / grounded（改动二遗留 / P1）----
    # avg_answer_grounding 只对有值的题求均（有检索命中+key_facts 的任务）；
    # 全部 None（无 RAG 语义任务）→ None → 报告渲染 "-"（N/A 显式）。
    ag = [r["success_signal"].get("answer_grounding") for r in results]
    ag_val = [x for x in ag if x is not None]
    grounded = sum(1 for r in results if r["success_signal"].get("grounded"))

    # ---- 新增：token 预算 ----
    over_budget = sum(1 for r in results if r["cost"].get("agent_in", 0) + r["cost"].get("agent_out", 0) > budget)

    # ---- 新增：工具耗时（跨任务聚合，基于原始延迟序列） ----
    tool_agg: dict[str, dict[str, Any]] = {}
    for r in results:
        for name, lats in (r.get("tool_latency_ms") or {}).items():
            base = tool_agg.setdefault(name, {"calls": 0, "lats": []})
            base["calls"] += len(lats)
            base["lats"].extend(lats)
    tool_latency: dict[str, dict[str, float]] = {}
    for name, base in tool_agg.items():
        srt = sorted(base["lats"])
        tool_latency[name] = {
            "calls": base["calls"],
            "avg_ms": round(sum(srt) / len(srt), 1) if srt else 0,
            "p95_ms": round(srt[min(len(srt) - 1, int(len(srt) * 0.95))], 1) if srt else 0,
            "max_ms": round(srt[-1], 1) if srt else 0,
        }
    tool_latency_top = dict(sorted(tool_latency.items(), key=lambda kv: kv[1]["avg_ms"], reverse=True)[:10])

    return {
        "n": n,
        "success_rate": round(sum(1 for o in outcomes if o == "success") / n, 3),
        "fail_rate": round(sum(1 for o in outcomes if o == "fail") / n, 3),
        "error_rate": round(sum(1 for o in outcomes if o == "error") / n, 3),
        "blocked_rate": round(sum(1 for o in outcomes if o == "blocked") / n, 3),
        # 失败原因分布（含步数超限=效率作为正确性的一部分）
        "fail_reasons": dict(fail_reasons),
        "max_steps_fail_rate": round(fail_reasons.get("max_steps_exceeded", 0) / n, 3),
        # 质量指标
        "pass_at_1": round(sum(1 for r in results if r["success_signal"].get("pass_at_1") == 1.0) / n, 3),
        "avg_rubric": round(mean(rubric), 3) if rubric else 0,
        "avg_fact_consistency": round(mean(fact), 3) if fact else 0,
        "avg_functional_correct": round(mean(fc), 3) if fc else 0,
        "avg_tool_flow_match": round(mean(tf), 3) if tf else 0,
        "avg_param_accuracy": round(mean(pa), 3) if pa else 0,
        # 答案级（改动二/P1）：无检索命中/无 key_facts 的任务 grounding=None → N/A 排除
        "avg_answer_grounding": round(mean(ag_val), 3) if ag_val else None,
        "grounded_rate": round(grounded / n, 3) if n else None,
        # 效率/成本
        "avg_steps": round(mean([r.get("steps", 0) for r in results]), 2),
        "avg_tool_calls": round(tool_calls / n, 2),
        "avg_detour": round(mean([r.get("detour_actions", 0) for r in results]), 2),
        "avg_input_tokens": round(total_in / n, 1),
        "avg_output_tokens": round(total_out / n, 1),
        "user_sim_in_tokens": user_sim_in,
        "total_cost_rmb": round(external, 5),
        "avg_cost_rmb_per_task": round(external / n, 6) if n else 0,
        # token 预算
        "token_budget": budget,
        "over_budget_rate": round(over_budget / n, 3) if n else 0,
        "latency": {
            "avg_total_ms": round(mean(totals_ms), 1) if totals_ms else None,
            "p50": _pct(totals_ms, 0.5),
            "p95": _pct(totals_ms, 0.95),
            "p99": _pct(totals_ms, 0.99),
            "avg_ttft_ms": round(mean(ttfts), 1) if ttfts else None,
            "p95_ttft": _pct(ttfts, 0.95),
            "p99_ttft": _pct(ttfts, 0.99),
            "avg_tpot_ms": round(mean(tpots), 1) if tpots else None,
        },
        "tool_latency_top": tool_latency_top,
        "permission": {
            "blocked_attempts": blocked,
            "total_attempts": attempts,
            "block_rate": round(blocked / attempts, 3) if attempts else 0,
        },
        "avg_subgoal_recall": round(mean(
            [r["success_signal"].get("subgoal_recall", 0) for r in results]), 3),
    }


def group_by_agent_and_policy(results: list[dict]) -> dict[str, dict[str, Any]]:
    """按 (agent_id, context_policy) 分组聚合。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        key = f"{r.get('agent_id')}|{r.get('context_policy')}"
        groups[key].append(r)
    return {k: aggregate_results(v) for k, v in groups.items()}


def critical_pass(results: list[dict], critical_ids: set[str] | None = None) -> dict[str, Any]:
    """Pass^k：关键任务子集是否全部通过。

    语义：critical_ids 缺省时取全部任务（即"所有任务全过"）；
    通过率 = 关键任务中成功数 / 关键任务总数（全过才算 1.0）。
    """
    subset = [r for r in results if critical_ids is None or r.get("golden_task_id") in critical_ids]
    if not subset:
        return {"critical_tasks": 0, "critical_pass_rate": 0.0, "critical_all_pass": False}
    passed = sum(1 for r in subset if r.get("outcome") == "success")
    return {
        "critical_tasks": len(subset),
        "critical_pass_rate": round(passed / len(subset), 3),
        "critical_all_pass": passed == len(subset),
    }


def group_by_agent(results: list[dict]) -> dict[str, dict[str, Any]]:
    """按 agent_id 聚合。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r.get("agent_id")].append(r)
    return {k: aggregate_results(v) for k, v in groups.items()}
