"""CLI：安全边界测试。

让 Agent 处理"不存在商品/要求降价/申请他人信息"等边界指令，
用 SafetyJudge 判定是否正确处理（拒绝越权/不编造/不泄露），量化安全指标。

用法：
    python -m mall_agent.cli.safety_test --scheme 1,2,3 --limit 12
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mall_agent.core.settings import settings


async def run(schemes: list[str], limit: int) -> None:
    from mall_agent.tools.cs_tools import register_cs_tools
    from mall_agent.tools.guide_tools import register_guide_tools
    from mall_agent.tools.admin_tools import register_admin_tools
    register_cs_tools(); register_guide_tools(); register_admin_tools()

    from mall_agent.eval.golden.safety_tasks import safety_tasks
    from mall_agent.eval.harness.safety_judge import SafetyJudge
    from mall_agent.agents.graph_factory import build_agent
    from mall_agent.core.types import ChatMessage

    tasks = safety_tasks()
    if not tasks:
        print("⚠️ 无安全任务数据，先运行: python -m mall_agent.eval.golden.safety_tasks")
        return
    if limit:
        tasks = tasks[:limit]
    print(f"安全边界测试：{len(tasks)} 条任务 × 方案 {schemes}")
    print(f"维度覆盖: {sorted({t['expect'] for t in tasks})}")
    print("=" * 70)

    # 按方案收集结果
    all_results: dict[str, list[dict]] = {s: [] for s in schemes}

    for scheme in schemes:
        agent = build_agent(scheme=scheme, context_policy="sliding")
        print(f"\n--- 方案{scheme} ---")
        for task in tasks:
            # 手动跑一轮（不用完整 harness，安全判定特殊）
            state: dict = {
                "session_id": f"sec-{task['task_id']}",
                "role": "cs" if task["domain"] == "customer_service" else "guide",
                "user_id": "1", "scheme": scheme,
                "messages": [ChatMessage(role="user", content=task["opener"])],
                "effective_messages": [], "context_policy": "sliding",
                "tool_records": [], "steps": 0, "max_steps": 15,
                "pending_tool_calls": [], "permission_events": [], "flags": {},
            }
            try:
                result = await agent.app.ainvoke(
                    dict(state), config={"recursion_limit": 40})
                state.update(result)
            except Exception as e:
                state["flags"]["error"] = str(e)[:80]

            # 若 agent 在等确认，给一次脚本回复
            if len(state.get("messages", [])) <= 4 and task.get("scripted_user_replies"):
                reply = task["scripted_user_replies"][0]
                state["messages"].append(ChatMessage(role="user", content=reply))
                try:
                    result = await agent.app.ainvoke(
                        dict(state), config={"recursion_limit": 40})
                    state.update(result)
                except Exception:
                    pass

            verdict = SafetyJudge.judge(task, state)
            all_results[scheme].append({"task_id": task["task_id"], **verdict})

            mark = {"pass": "✅", "fail": "❌", "uncertain": "⚠️"}.get(verdict["outcome"], "?")
            print(f"  {mark} {task['task_id']} [{task['expect']}] {task['opener'][:36]}")
            print(f"     → {verdict['reason']}")

    # 汇总
    print("\n" + "=" * 70)
    print("安全指标汇总")
    from mall_agent.eval.metrics import safety as safety_metrics
    for scheme in schemes:
        rs = all_results[scheme]
        n = len(rs)
        passed = sum(1 for r in rs if r["outcome"] == "pass")
        failed = sum(1 for r in rs if r["outcome"] == "fail")
        uncertain = sum(1 for r in rs if r["outcome"] == "uncertain")
        # 按维度
        by_dim = {}
        for r in rs:
            d = r["dimension"]
            by_dim.setdefault(d, {"pass": 0, "fail": 0, "uncertain": 0})
            by_dim[d][r["outcome"]] = by_dim[d].get(r["outcome"], 0) + 1
        print(f"  方案{scheme}: pass={passed}/{n} fail={failed} uncertain={uncertain} "
              f"通过率={passed/n:.2f}")
        for d, cnt in by_dim.items():
            print(f"    {d:20s} pass={cnt['pass']} fail={cnt['fail']} uncertain={cnt['uncertain']}")

        # S2 误拒率（合法基线子集，走统一 metrics）
        legit = [r for r in rs if r["dimension"] == "legit_accept"]
        if legit:
            # 构造最小 TrialResult 供 false_refusal_rate 使用
            legit_trials = [{
                "outcome": "success" if r["outcome"] == "pass" else "fail",
                "messages": [{"role": "assistant", "content": r.get("reply", "")}],
            } for r in legit]
            fr = safety_metrics.false_refusal_rate(legit_trials)
            print(f"    误拒率 false_refusal_rate={fr['false_refusals']}/{fr['legit_tasks']} "
                  f"= {fr['false_refusal_rate']:.2f}")

        # S1 guardrail 漏拦率（按调用级）
        gl = safety_metrics.guardrail_leak_rate(rs)
        print(f"    guardrail 漏拦率={gl['leaked_calls']}/{gl['attempted_calls']} "
              f"= {gl['guardrail_leak_rate']:.2f}")

        # S4 毒性：检查所有判定的回复
        toxic = sum(1 for r in rs if safety_metrics.toxicity_check(r.get("reply", ""))["toxic"])
        if toxic:
            print(f"    输出毒性命中={toxic}")

        # S5 审计字段完整性（对判定结果构造最小 TrialResult）
        trials = [{
            "trial_id": f"sec-{r.get('task_id')}", "golden_task_id": r.get("task_id"),
            "context_policy": "sliding", "outcome": r.get("outcome", "uncertain"),
            "steps": 0, "cost": {"external_cost_rmb": 0.0},
            "trajectory": [{"type": "tool", "name": t, "ok": True} for t in r.get("tools", [])],
        } for r in rs]
        ac = safety_metrics.audit_completeness(trials)
        print(f"    审计字段完整性={ac['complete']}/{ac['tasks']} = {ac['audit_completeness']:.2f}")

    # 落盘（合并已有结果，避免每次运行覆盖）
    out = Path(settings.get("data_dir", "data")) / "reports" / "safety_test.json"
    merged = {}
    if out.exists():
        try:
            merged = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(all_results)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="1,2,3")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args.scheme.split(","), args.limit))


if __name__ == "__main__":
    sys.exit(main())
