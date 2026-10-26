"""CLI：运行评测（黄金数据集 × 方案 × 上下文策略）。

用法：
    python -m mall_agent.cli.eval --scheme 1,2,3 --policy full,sliding --limit 3
    python -m mall_agent.cli.eval --scheme 1 --dry-run      # 只生成数据集+检查
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from mall_agent.core.settings import settings


def _register_tools() -> None:
    from mall_agent.tools.cs_tools import register_cs_tools
    from mall_agent.tools.guide_tools import register_guide_tools
    from mall_agent.tools.admin_tools import register_admin_tools
    register_cs_tools()
    register_guide_tools()
    register_admin_tools()


async def run(schemes: list[str], policies: list[str], limit: int, harness: str = "custom",
              token_budget: int | None = None, concurrency: int = 1,
              simulator: str = "scripted", prefix: str | None = None,
              category: str | None = None, dataset: str | None = None,
              trial_timeout: float = 180.0, raw_out: str | None = None,
              rag_enabled: bool = True, reset_state: bool = True) -> list[dict]:
    _register_tools()
    from mall_agent.eval.golden.generator import load_dataset
    from mall_agent.agents.graph_factory import build_agent

    from pathlib import Path as _Path
    # 数据集选择：显式 --dataset 优先；否则默认 v1（14 条）——曾因默认落到 v3(103 条)
    # 导致「跑一次全量」实际是 309 trials，看起来像卡死。
    from mall_agent.eval.golden.generator_v3 import load_v3
    if dataset:
        ds_path = _Path(dataset)
    else:
        ds_path = _Path("data/golden/golden_v1.jsonl")
    if not ds_path.exists():
        print(f"! 数据集不存在: {ds_path}")
        return []
    tasks = load_v3(ds_path)
    print(f"数据集: {ds_path}（{len(tasks)} 条）")
    if not tasks:
        print("! 无黄金数据集，先运行: python -m mall_agent.cli.eval --gen")
        return []
    # 任务过滤（改动二：--prefix 按 task_id 前缀，--category 按 category 字段）
    if prefix:
        prefixes = {p.strip() for p in prefix.split(",") if p.strip()}
        tasks = [t for t in tasks if t.task_id.split("-")[0] in prefixes]
    if category:
        cats = {c.strip() for c in category.split(",") if c.strip()}
        tasks = [t for t in tasks if getattr(t, "category", "") in cats]
    if limit:
        tasks = tasks[:limit]
    print(f"加载 {len(tasks)} 条黄金任务，方案 {schemes}，策略 {policies}")

    # 构建 harness
    if harness == "custom":
        from mall_agent.eval.harness.custom_runner import CustomHarness
        from mall_agent.eval.harness.probe import MockStateProbe
        from mall_agent.mall_api.client import get_mall_client
        client = get_mall_client()
        mock = client.ensure_mock_sync()
        probe = MockStateProbe(mock, user_id="1")
        harness_inst = CustomHarness(probe=probe, simulator_mode=simulator,
                                     trial_timeout=trial_timeout, reset_state=reset_state)
    elif harness == "tau":
        from mall_agent.eval.harness.tau_adapter import TauStyleAgent, TauStyleEnv
        from mall_agent.eval.harness.probe import MockStateProbe
        from mall_agent.mall_api.client import get_mall_client
        client = get_mall_client()
        mock = client.ensure_mock_sync()
        probe = MockStateProbe(mock, user_id="1")
        env = TauStyleEnv(tasks, probe)

        async def run_tau_trial(task, agent, policy, agent_id="1"):
            tau_agent = TauStyleAgent(agent)
            idx = next(i for i, t in enumerate(env.tasks) if t.task_id == task.task_id)
            result = await tau_agent.solve(env, idx, context_policy=policy)
            result["harness"] = "tau"
            result["agent_id"] = agent_id
            result["golden_task_id"] = task.task_id
            return result
        harness_inst = type("TauHarness", (), {"run_trial": run_tau_trial})()
    else:
        raise ValueError(f"未知 harness: {harness}")

    results: list[dict] = []
    _raw_fh = open(raw_out, "w", encoding="utf-8") if raw_out else None
    for scheme in schemes:
        for policy in policies:
            # 每个 policy 重新构建 agent —— context_policy 必须真实生效，而非只进元数据
            agent = build_agent(scheme=scheme, context_policy=policy, rag_enabled=rag_enabled)

            async def run_one(task: Any) -> dict:
                result = await harness_inst.run_trial(task, agent, policy, agent_id=scheme)
                result["meta"]["task_domain"] = task.domain
                return result

            if concurrency and concurrency > 1:
                # 并发压测：同一 (agent, policy) 内并发跑任务，验证吞吐/排队下的 P95/P99
                batch = await asyncio.gather(*[run_one(t) for t in tasks])
                for result in batch:
                    print(f"  -> agent={scheme} policy={policy} task={result['golden_task_id']} "
                          f"outcome={result['outcome']} steps={result['steps']} "
                          f"tools={result['cost']['tool_calls']}")
                results.extend(batch)
                if _raw_fh:
                    for r in batch:
                        _raw_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    _raw_fh.flush()
            else:
                for task in tasks:
                    print(f"  -> agent={scheme} policy={policy} task={task.task_id}", end=" ")
                    result = await run_one(task)
                    print(f"outcome={result['outcome']} steps={result['steps']} "
                          f"tools={result['cost']['tool_calls']}")
                    results.append(result)
                    # 逐条落盘：即使进程在收尾阶段挂起/被杀，数据也不丢
                    if _raw_fh:
                        _raw_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                        _raw_fh.flush()
    if _raw_fh:
        _raw_fh.close()
    return results


def _apply_strict(results: list[dict], sg_th: float, tf_th: float) -> int:
    """把「probe 默认通过」的结果改判为严格口径：子目标召回与工具流匹配都要达标。

    动机：多数黄金任务 expected_state 为空，probe 空断言直接 True，任务第一轮就判成功。
    严格口径用 success_signal 里的 subgoal_recall / tool_flow_match 复核真实完成度。
    """
    changed = 0
    for r in results:
        sig = r.setdefault("success_signal", {})
        sg = sig.get("subgoal_recall") or 0.0
        tf = sig.get("tool_flow_match") or 0.0
        ok = sg >= sg_th and tf >= tf_th
        legacy = r.get("outcome")
        r.setdefault("meta", {})["outcome_legacy"] = legacy
        new = "success" if ok else "fail"
        if legacy != new:
            changed += 1
        r["outcome"] = new
        sig["pass_at_1"] = 1.0 if ok else 0.0
    return changed


def _report_from(path: str, strict: bool = False, sg_th: float = 0.8, tf_th: float = 0.8) -> None:
    """从 raw_*.jsonl 重建报告（评测跑完后进程挂住时，用它补报告）。"""
    from mall_agent.eval.metrics.aggregate import critical_pass
    from mall_agent.eval.report import generate_report, save_raw, save_report

    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    if strict:
        n = _apply_strict(results, sg_th, tf_th)
        print(f"[strict] 改判 {n} 条（subgoal>={sg_th} 且 tool_flow>={tf_th}）")
    print(f"载入 {len(results)} 条 trial: {path}")
    report = generate_report(results, title="mall-agent 三方案评测")
    print("\n" + "=" * 40)
    print(report)
    cp = critical_pass(results, None)
    print(f"\n--- Pass^k --- {cp['critical_tasks']} 个任务，通过率 {cp['critical_pass_rate']:.3f}")
    p = save_report(report)
    save_raw(results)
    print(f"[OK] 报告已保存: {p}")
    sys.stdout.flush()
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="mall-agent 评测 CLI")
    parser.add_argument("--scheme", default=None, help="逗号分隔方案，如 1,2,3")
    parser.add_argument("--policy", default=None, help="逗号分隔策略，如 full,sliding")
    parser.add_argument("--limit", type=int, default=0, help="限制任务数（调试用）")
    parser.add_argument("--harness", default="custom", help="custom|tau")
    parser.add_argument("--token-budget", type=int, default=None,
                        help="单任务 token 预算（超预算率指标用）")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发压测：同一 (方案,策略) 内并发跑任务（默认 1=串行）")
    parser.add_argument("--critical-ids", default=None,
                        help="关键任务子集（逗号分隔 task_id，用于 Pass^k 门槛，默认全部任务）")
    parser.add_argument("--simulator", default="scripted",
                        choices=["scripted", "persona", "hybrid"],
                        help="用户模拟器：scripted(脚本，默认) | persona(LLM) | hybrid(先脚本后LLM)")
    parser.add_argument("--prefix", default=None,
                        help="按 task_id 前缀过滤任务（逗号分隔，如 PQ,NUM,APP）")
    parser.add_argument("--category", default=None,
                        help="按 category 字段过滤任务（逗号分隔，如 PQ,NUM,SECURITY）")
    parser.add_argument("--dataset", default=None,
                        help="指定数据集 jsonl（默认 v3；v1=golden_v1.jsonl、v2=golden_v2.jsonl）")
    parser.add_argument("--trial-timeout", type=float, default=180.0,
                        help="单轮 agent 调用超时（秒），超时记 error 不拖死整批")
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="整批墙钟上限（分钟，0=不限）；到点落盘已有结果并退出")
    parser.add_argument("--no-rag", action="store_true",
                        help="关闭 RAG 检索工具（DB 域任务用它可去掉 MCP 冷启动与检索噪声）")
    parser.add_argument("--no-reset", action="store_true",
                        help="不重置每题用户态（默认重置；关闭后与旧口径可比）")
    parser.add_argument("--strict", action="store_true",
                        help="严格口径：probe 默认通过的任务改用 subgoal/tool_flow 复核")
    parser.add_argument("--strict-sg", type=float, default=0.8, help="严格口径 subgoal 阈值")
    parser.add_argument("--strict-tf", type=float, default=0.8, help="严格口径 tool_flow 阈值")
    parser.add_argument("--report-from", default=None,
                        help="从已落盘的 raw_*.jsonl 重新生成报告（不重跑）")
    parser.add_argument("--gen", action="store_true", help="重新生成黄金数据集")
    parser.add_argument("--dry-run", action="store_true", help="只生成数据集+列任务，不跑")
    args = parser.parse_args()

    if args.report_from:
        _report_from(args.report_from, args.strict, args.strict_sg, args.strict_tf)
        return

    if args.gen:
        from mall_agent.eval.golden.generator_v3 import generate_v3, write_v3
        ts = generate_v3()
        p = write_v3(ts)
        print(f"[OK] 生成 {len(ts)} 条黄金任务 → {p}")
        return

    schemes = (args.scheme or settings.default_scheme).split(",")
    policies = (args.policy or "sliding").split(",")

    if args.dry_run:
        from mall_agent.eval.golden.generator import load_dataset
        tasks = load_dataset()
        print(f"黄金任务数: {len(tasks)}")
        for t in tasks:
            print(f"  {t.task_id} [{t.domain}] {t.opener[:40]}")
        return

    if args.max_minutes and args.max_minutes > 0:
        # 看门狗：LangGraph/anyio 不允许从外部取消协程（会抛 cancel scope 跨任务错误），
        # 所以超时只能整进程退出——已落盘的 raw_*.jsonl 不会丢，用 --report-from 补报告。
        import threading

        def _watchdog():
            time.sleep(args.max_minutes * 60)
            print("[WATCHDOG] 超过", args.max_minutes, "分钟，落盘后退出", flush=True)
            os._exit(2)

        threading.Thread(target=_watchdog, daemon=True).start()
        print(f"看门狗: {args.max_minutes} 分钟后强制退出")

    from pathlib import Path as _P
    _raw_dir = _P(settings.get("data_dir", "data")) / "reports"
    _raw_dir.mkdir(parents=True, exist_ok=True)
    raw_out = str(_raw_dir / f"raw_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    print(f"逐条落盘: {raw_out}")
    results = asyncio.run(run(schemes, policies, args.limit, args.harness, args.token_budget,
                              args.concurrency, args.simulator, args.prefix, args.category,
                              args.dataset, args.trial_timeout, raw_out,
                              rag_enabled=not args.no_rag, reset_state=not args.no_reset))
    if results:
        if args.strict:
            n = _apply_strict(results, args.strict_sg, args.strict_tf)
            print("[strict] 改判", n, "条")
        from collections import defaultdict
        from mall_agent.eval.metrics.aggregate import aggregate_results
        from mall_agent.eval.report import generate_report, save_raw, save_report
        report = generate_report(results, title="mall-agent 三方案评测")
        # 应用 token 预算（若指定）到聚合
        if args.token_budget:
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in results:
                groups[r.get("agent_id")].append(r)
            print(f"\n--- 超 token 预算率（预算 {args.token_budget}）---")
            for k, v in groups.items():
                a = aggregate_results(v, token_budget=args.token_budget)
                print(f"  方案 {k}: {a.get('over_budget_rate', 0):.2f}")
        print("\n" + "=" * 40)
        print(report)
        # Pass^k：关键任务子集全过（--critical-ids 指定，缺省全部）
        from mall_agent.eval.metrics.aggregate import critical_pass
        critical_ids = {c.strip() for c in args.critical_ids.split(",") if c.strip()} \
            if args.critical_ids else None
        cp = critical_pass(results, critical_ids)
        print(f"\n--- Pass^k（关键任务全过）---")
        print(f"  关键任务 {cp['critical_tasks']} 个：通过率 {cp['critical_pass_rate']:.3f} "
              f"{'[OK] 全过' if cp['critical_all_pass'] else '[X] 未全过'}")
        save_raw(results)
        p = save_report(report)
        print(f"\n[OK] 报告已保存: {p}")
        sys.stdout.flush()
        # RAG MCP 的 stdio 子进程会让解释器在退出阶段挂住；数据已落盘，直接硬退出
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
