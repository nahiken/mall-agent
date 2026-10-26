"""CLI：RAG × provider × style 配对实验骨架（改动五.2/改动七，冒烟优先）。

8 格 = {RAG ON/OFF} × {provider: deepseek/ollama} × {style: standard/concise}。
每格对同题配对跑（answer 池子集），记录 answer_correct(rule+judge)/out_token/cost/
归因等；落 data/reports/exp_rag_eval/{summary.json,summary.md,per_task.jsonl}。
style 轴由改动七(PROMPT_STYLE)装配：_run_cell 按 CELLS 的 style 经 build_agent 注入，
concise 格生效（system prompt 追加简洁约束，只作用 agent 主调用）。

RAG OFF 定义（R1）：屏蔽检索工具（domain="rag"），零外部知识注入。

provider 说明：ollama 走本地 11451（qwen2.5:7b / llama3.2:3b），需 OLLAMA_MODEL /
OLLAMA_BASE_URL env 或 settings.user_sim 级覆盖；deepseek 走 LLM_*。当前 CLI 以
**单一事件循环**运行（RagMcpClient 单例绑定 loop，避免多 asyncio.run 断连）。

用法（冒烟）：
    python -m mall_agent.cli.exp_rag_eval --cells ds_rag_on,ds_rag_off --cats PQ,APP --limit 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mall_agent.core.llm import LLMClient, make_llm
from mall_agent.core.settings import settings

DATA_DIR = Path(settings.get("data_dir", "data"))
REPORT_DIR = DATA_DIR / "reports" / "exp_rag_eval"

# 格代号 → (provider, rag, style)
CELLS = {
    "ds_rag_on": ("deepseek", True, "standard"),
    "ds_rag_off": ("deepseek", False, "standard"),
    "ol_rag_on": ("ollama", True, "standard"),
    "ol_rag_off": ("ollama", False, "standard"),
    "ds_rag_on_concise": ("deepseek", True, "concise"),
    "ds_rag_off_concise": ("deepseek", False, "concise"),
    "ol_rag_on_concise": ("ollama", True, "concise"),
    "ol_rag_off_concise": ("ollama", False, "concise"),
}

# provider → llm 覆盖来源（读对应 env）
_PROVIDER_LLM = {
    "deepseek": dict(model=settings.env("LLM_MODEL", settings.llm.get("model", "deepseek-chat")),
                     base_url=settings.env("LLM_BASE_URL", settings.llm.get("base_url"))),
    "ollama": dict(model=settings.env("OLLAMA_MODEL", "qwen2.5:7b"),
                   base_url=settings.env("OLLAMA_BASE_URL", "http://127.0.0.1:11451/v1"),
                   api_key=settings.env("OLLAMA_API_KEY", "ollama")),
}


def _register_tools() -> None:
    from mall_agent.tools.cs_tools import register_cs_tools
    from mall_agent.tools.guide_tools import register_guide_tools
    from mall_agent.tools.admin_tools import register_admin_tools
    register_cs_tools()
    register_guide_tools()
    register_admin_tools()


class _TaskLike:
    """judge/rule 需要的 task 最小视图。"""
    def __init__(self, ctx: dict):
        self.key_facts = ctx.get("key_facts", [])
        self.reference_answer = ctx.get("reference_answer", "")
        self.category = ctx.get("category", "")
        self.task_id = ctx.get("task_id", "")


async def _run_cell(cell: str, tasks: list, scheme: str, policy: str, simulator: str) -> list[dict]:
    from mall_agent.agents.graph_factory import build_agent
    from mall_agent.eval.harness.custom_runner import CustomHarness
    from mall_agent.eval.harness.probe import MockStateProbe
    from mall_agent.mall_api.client import get_mall_client

    provider, rag, style = CELLS[cell]
    client = get_mall_client()
    mock = client.ensure_mock_sync()
    probe = MockStateProbe(mock, user_id="1")
    harness = CustomHarness(probe=probe, simulator_mode=simulator)

    pconf = _PROVIDER_LLM[provider]
    llm = LLMClient(model=pconf["model"], base_url=pconf["base_url"],
                    api_key=pconf.get("api_key"), caller="agent")
    # 改动七：style 轴随 build_agent 注入 → state["prompt_style"]（standard=零改动基线）
    agent = build_agent(scheme=scheme, context_policy=policy, llm=llm,
                        rag_enabled=rag, prompt_style=style)

    rows = []
    for task in tasks:
        try:
            r = await harness.run_trial(task, agent, policy, agent_id=cell)
            r["cell"] = cell
            rows.append(r)
        except Exception as e:
            ctx = task.model_dump()
            rows.append({
                "cell": cell, "golden_task_id": task.task_id, "outcome": "error",
                "error": f"{type(e).__name__}: {e}",
                "answer_ctx": {
                    "task_id": task.task_id, "question": task.opener,
                    "reference_answer": getattr(task, "reference_answer", ""),
                    "key_facts": list(getattr(task, "key_facts", []) or []),
                    "category": getattr(task, "category", ""),
                    "rag_enabled": rag, "reply": "", "tool_records": [],
                },
            })
    return rows


async def _judge_rows(rows: list[dict], judge_llm: Any) -> list[dict]:
    from mall_agent.judge.answer_judge import async_answer_correct

    out = []
    for r in rows:
        ctx = r.get("answer_ctx") or {}
        row = dict(r)
        if not ctx.get("reference_answer"):
            row["judge"] = {"skip": "no_reference"}
            out.append(row)
            continue
        tool_results = [(t["name"], t["result"]) for t in ctx.get("tool_records", [])]
        try:
            sig = await async_answer_correct(ctx["question"], _TaskLike(ctx), ctx["reply"],
                                             tool_results, judge_llm)
            row["judge"] = sig
        except Exception as e:
            row["judge"] = {"skip": "judge_error", "err": f"{type(e).__name__}: {e}"}
        out.append(row)
    return out


def _parse_cat_limit(s: str | None) -> dict[str, int] | None:
    """解析 'PQ:12,APP:8,MH:8,NUM:8' → {类别: 上限}；'PQ'（无 :n）表示该类全取。"""
    if not s:
        return None
    d: dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip()] = int(v.strip())
        else:
            d[part] = 10**9  # 无 ':' → 该类全取（上限极大）
    return d


def _load_tasks(cats: list[str], prefixes: list[str], limit: int,
                cat_limit: dict[str, int] | None = None) -> list:
    """加载 answer 池任务。

    cat_limit（如 {'PQ':12,'APP':8,'MH':8,'NUM':8}）：按类别分别采样——
    每类别内先取非私域(常识)题、再取私域题，填满至该类上限，保证分层对照组不被序截断。
    limit：向后兼容的全局截断（cat_limit 优先）。
    """
    from mall_agent.eval.golden.policy_dataset import answer_pool, load_v4
    tasks = answer_pool(load_v4())
    if cats:
        tasks = [t for t in tasks if t.category in cats]
    if prefixes:
        tasks = [t for t in tasks if t.task_id.split("-")[0] in prefixes]
    if cat_limit:
        by_cat: dict[str, list] = defaultdict(list)
        for t in tasks:
            by_cat[t.category].append(t)
        out: list = []
        for cat, k in cat_limit.items():
            bucket = by_cat.get(cat, [])
            if k >= len(bucket):
                out.extend(bucket)
                continue
            nonp = [t for t in bucket if not getattr(t, "depends_on_private_kb", False)]
            priv = [t for t in bucket if getattr(t, "depends_on_private_kb", False)]
            out.extend((nonp + priv)[:k])
        return out
    if limit:
        tasks = tasks[:limit]
    return tasks


# 改动六：扩展聚合键（对已落盘 per_task.jsonl 只读聚合，不重跑/不加新采集）
# 在既有 5 列(answer_correct/grounded/rule/judge/disagree)之上补：
#   token 效率: avg_in/avg_out_token、每任务 cost¥、avg_tool_calls、avg_steps
#   时延: total p50/p99、TTFT p50/p99、TPOT p50
#   归因分布: judge_attr 计数; judge-rule disagree 率
#   分层: 按 category 与 depends_on_private_kb 分 answer_correct（private_kb 分层 Δ 报告）
# cost 列分 provider(R7): DeepSeek 有意义(¥), Ollama 本地≈0 —— 由 cell 前缀 ds_/ol_ 区分,
#   本地不报 ¥/答对（避免误导 0 成本除零），改用 out-token/答对作性价比。
_ATTR_ORDER = ["正确", "漏要点", "错口径", "编造", "答非所问", "拒绝成功", "拒绝不当"]


def _avg(xs):
    return round(sum(xs) / len(xs), 1) if xs else None


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return round(s[k], 1)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """把一批 per-task 行聚合为 summary（含改动六扩展列）。"""
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[r.get("cell") or "?"].append(r)
    out = []
    for cell in sorted(by_cell):
        rs = by_cell[cell]
        js = [r["judge"] for r in rs if isinstance(r.get("judge"), dict) and "answer_correct" in r.get("judge", {})]
        if not js:
            out.append({"cell": cell, "n": len(rs), "note": "no_judged"})
            continue
        sc = [j["answer_correct"] for j in js]
        ac = round(sum(sc) / len(sc), 3)
        # R5：逐 rep 均值 + spread（per-task rows 带 rep 标记）
        rep_ac: dict[int, list[float]] = defaultdict(list)
        for r in rs:
            j = r.get("judge")
            if isinstance(j, dict) and "answer_correct" in j:
                rep_ac[r.get("rep", 1)].append(j["answer_correct"])
        rep_means = [round(sum(v) / len(v), 3) for v in rep_ac.values()]
        spread = round(max(rep_means) - min(rep_means), 3) if len(rep_means) > 1 else None
        gd = [j["grounded"] for j in js if "grounded" in j]
        rl = [j["rule_score"] for j in js if j.get("rule_score") is not None]
        jv = [j["judge_score"] for j in js if "judge_score" in j]
        # token / cost / 时延
        in_tok = [r.get("cost", {}).get("agent_in", 0) for r in rs]
        out_tok = [r.get("cost", {}).get("agent_out", 0) for r in rs]
        rmb = [r.get("cost", {}).get("external_cost_rmb", 0) or 0 for r in rs]
        tool_n = [r.get("cost", {}).get("tool_calls", 0) for r in rs]
        steps = [r.get("steps", 0) for r in rs]
        total_ms = [r.get("latency_ms", {}).get("total", 0) for r in rs]
        ttft_ms = [r.get("latency_ms", {}).get("ttft", 0) for r in rs]
        tpot_ms = [r.get("latency_ms", {}).get("tpot", 0) for r in rs]
        avg_rmb = round(sum(rmb) / len(rmb), 4) if rmb else None
        # 归因分布（judge_attr）
        attr_counter = Counter(j.get("judge_attr", "") or "" for j in js)
        attr = {k: attr_counter.get(k, 0) for k in _ATTR_ORDER if attr_counter.get(k, 0)}
        # 分层：category / private_kb（按行自带 judge，防 skip 行错位）
        by_cat: dict[str, list[float]] = defaultdict(list)
        by_priv: dict[bool, list[float]] = defaultdict(list)
        for r in rs:
            j = r.get("judge")
            if not (isinstance(j, dict) and "answer_correct" in j):
                continue
            ctx = r.get("answer_ctx", {}) or {}
            by_cat[ctx.get("category", "?")].append(j["answer_correct"])
            by_priv[bool(ctx.get("depends_on_private_kb"))].append(j["answer_correct"])
        s = {
            "cell": cell, "n": len(rs), "n_reps": len(rep_means),
            "rep_answer_correct": rep_means,
            "spread": spread,  # R5：跨 rep 均值极差（None=单次运行）
            "answer_correct_avg": ac,
            "grounded_rate": round(sum(1 for v in gd if v) / len(gd), 3) if gd else None,
            "rule_avg": round(sum(rl) / len(rl), 3) if rl else None,
            "judge_avg": round(sum(jv) / len(jv), 3) if jv else None,
            "disagree": sum(1 for j in js if j.get("judge_rule_disagree")),
            "disagree_rate": round(sum(1 for j in js if j.get("judge_rule_disagree")) / len(js), 3),
            # token/成本
            "avg_in_token": _avg(in_tok), "avg_out_token": _avg(out_tok),
            "avg_rmb_per_task": avg_rmb,
            "avg_tool_calls": _avg(tool_n), "avg_steps": _avg(steps),
            # 时延
            "total_ms_p50": _pct(total_ms, 0.5), "total_ms_p99": _pct(total_ms, 0.99),
            "ttft_ms_p50": _pct(ttft_ms, 0.5), "ttft_ms_p99": _pct(ttft_ms, 0.99),
            "tpot_ms_p50": _pct(tpot_ms, 0.5),
            # 归因
            "attr_dist": attr,
            # 分层
            "answer_correct_by_cat": {k: round(sum(v) / len(v), 3) for k, v in by_cat.items()},
            "answer_correct_by_private_kb": {
                "private": round(sum(by_priv[True]) / len(by_priv[True]), 3) if by_priv[True] else None,
                "non_private": round(sum(by_priv[False]) / len(by_priv[False]), 3) if by_priv[False] else None,
            },
        }
        # 性价比：token/答对 全格通用；¥/答对 仅付费格(ds_)有意义（R7）
        provider = "deepseek" if cell.startswith("ds_") else ("ollama" if cell.startswith("ol_") else "?")
        if ac and ac > 0:
            s["out_token_per_correct"] = round((sum(out_tok) / len(out_tok)) / ac, 1) if out_tok else None
        else:
            s["out_token_per_correct"] = None
        if provider == "deepseek" and ac and ac > 0:
            s["rmb_per_correct"] = round((sum(rmb) / len(rmb)) / ac, 4)
        else:
            s["rmb_per_correct"] = None  # Ollama 本地≈0，不报 ¥/答对（R7）
        out.append(s)
    return out


def _report_from_rows(summary_rows: list[dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sp = REPORT_DIR / "summary.json"
    sp.write_text(json.dumps({"summary": summary_rows, "cells": CELLS},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# exp_rag_eval 结果", "",
          "| 格 | n(rep) | 答对均 | spread | grounded | rule | judge | disagree | avg_in | avg_out | ¥/题 | 工具 | 步 | total_p50 | ttft_p50 |",
          "|----|---|------|------|------|------|------|------|------|------|------|------|------|------|------|"]
    for s in summary_rows:
        if "note" in s:
            md.append(f"| {s['cell']} | {s['n']} | - (no judge) |")
            continue
        def f(x, nd=2):
            return "-" if x is None else f"{x:.{nd}f}"
        n_lbl = f"{s['n']}" + (f"(rep{s['n_reps']})" if s.get("n_reps", 1) > 1 else "")
        rep_s = "/".join(f"{m:.2f}" for m in s.get("rep_answer_correct", []))
        spread_s = f"{s['spread']:.3f}" if s.get("spread") is not None else f"{s['spread']}"
        md.append(f"| {s['cell']} | {n_lbl} | {f(s['answer_correct_avg'])} | {spread_s} | {f(s['grounded_rate'])} | "
                  f"{f(s['rule_avg'])} | {f(s['judge_avg'])} | {s['disagree']} | {f(s['avg_in_token'],0)} | "
                  f"{f(s['avg_out_token'],0)} | {f(s['avg_rmb_per_task'],4)} | {f(s['avg_tool_calls'],1)} | "
                  f"{f(s['avg_steps'],1)} | {f(s['total_ms_p50'],0)} | {f(s['ttft_ms_p50'],0)} |"
                  + (f"<br><sub>rep均:{rep_s}</sub>" if s.get("n_reps", 1) > 1 else ""))
    md += ["", "### 分层（按 depends_on_private_kb）", "",
           "| 格 | private 答对均 | 常识(非私域) 答对均 | out-token/答对 | ¥/答对 |",
           "|----|------|------|------|------|"]
    for s in summary_rows:
        if "note" in s:
            continue
        pk = s["answer_correct_by_private_kb"]
        md.append(f"| {s['cell']} | {f(pk['private'])} | {f(pk['non_private'])} | "
                  f"{f(s['out_token_per_correct'],1)} | {f(s['rmb_per_correct'],4)} |")
    md += ["", "### 归因分布（judge_attr 计数）", ""]
    for s in summary_rows:
        if "note" in s:
            continue
        md.append(f"- {s['cell']}: " + ("、".join(f"{k}={v}" for k, v in s["attr_dist"].items()) or "(空)"))
    md += ["", "### ON−OFF Δ（同 provider×style 内，RAG 开关增益）", "",
           "| 对比 | Δ答对 | Δgrounded | Δavg_out_token | Δ¥/题 | Δ工具/Δ步 |",
           "|------|------|------|------|------|------|"]
    # 组 (provider,style) → {on,off}; 用 CELLS 映射 cell→(provider,rag,style)
    cellmeta = {cell: meta for cell, meta in CELLS.items()}
    groups: dict[tuple, dict] = {}
    for s in summary_rows:
        meta = cellmeta.get(s.get("cell"))
        if not meta:
            continue
        key = (meta[0], meta[2])   # (provider, style)
        groups.setdefault(key, {})["on" if meta[1] else "off"] = s
    for (prov, style), m in sorted(groups.items()):
        on, off = m.get("on"), m.get("off")
        if not (on and off):
            continue
        def _d(on_v, off_v):
            if on_v is None or off_v is None:
                return "-"
            return f"{on_v - off_v:+.3f}"
        style_lbl = "" if style == "standard" else f"-{style}"
        md.append(f"| {prov}{style_lbl} | "
                  f"{_d(on['answer_correct_avg'], off['answer_correct_avg'])} | "
                  f"{_d(on['grounded_rate'], off['grounded_rate'])} | "
                  f"{_d(on['avg_out_token'], off['avg_out_token'])} | "
                  f"{(on['avg_rmb_per_task'] or 0) - (off['avg_rmb_per_task'] or 0):+.4f} | "
                  f"{(on['avg_tool_calls'] or 0) - (off['avg_tool_calls'] or 0):+.1f}/"
                  f"{(on['avg_steps'] or 0) - (off['avg_steps'] or 0):+.1f} |")
    (REPORT_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"[OK] {sp}")
    return sp


def report_only() -> Path:
    """只读产物聚合：读现有 per_task.jsonl 重算 summary（改动六，不重跑/不加采集）。"""
    per_path = REPORT_DIR / "per_task.jsonl"
    if not per_path.exists():
        raise SystemExit(f"! {per_path} 不存在（先跑评测或改路径）")
    rows = []
    with open(per_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return _report_from_rows(aggregate_rows(rows))


async def _main_async(args: argparse.Namespace) -> None:
    _register_tools()
    cells = [c for c in args.cells.split(",") if c in CELLS]
    cat_limit = _parse_cat_limit(args.cat_limit)
    tasks = _load_tasks([c for c in (args.cats or "").split(",") if c],
                        [p for p in (args.prefix or "").split(",") if p], args.limit,
                        cat_limit)
    if not tasks:
        print("! 任务为空（检查 --cats/--prefix 或先生成 v4）")
        return
    # 分层采样后仍保留私域/常识分布信息
    priv_n = sum(1 for t in tasks if getattr(t, "depends_on_private_kb", False))
    print(f"任务 {len(tasks)} 条（{priv_n} 私域 / {len(tasks)-priv_n} 常识）→ 格 {cells}"
          f" × repeats={args.repeats}")

    # judge 固定同一模型（R2）：make_llm("judge") 单例全格一致
    judge_llm = make_llm("judge")

    per_path = REPORT_DIR / "per_task.jsonl"
    per_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    with open(per_path, "w", encoding="utf-8") as f:  # 新跑清空
        pass
    for cell in cells:
        for rep in range(1, args.repeats + 1):
            print(f"\n=== 跑 {cell} rep{rep}（{CELLS[cell]}）===")
            t0 = time.perf_counter()
            rows = await _run_cell(cell, tasks, args.scheme, args.policy, args.simulator)
            rows = await _judge_rows(rows, judge_llm)
            dt = time.perf_counter() - t0
            for r in rows:
                r["rep"] = rep  # R5：重复标记（aggregate 按 rep 分均值+spread）
            all_rows.extend(rows)
            with open(per_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            ok = sum(1 for r in rows if r.get("outcome") == "success")
            print(f"  {cell} rep{rep}: {ok}/{len(rows)} success，耗时 {dt:.0f}s")
    _report_from_rows(aggregate_rows(all_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG×provider×style 配对实验")
    parser.add_argument("--cells", default="ds_rag_on,ds_rag_off", help="格代号逗号分隔")
    parser.add_argument("--cats", default="PQ,APP", help="category 过滤（逗号分隔）")
    parser.add_argument("--prefix", default=None, help="task_id 前缀过滤")
    parser.add_argument("--limit", type=int, default=3, help="全局任务数上限（冒烟，cat_limit 优先）")
    parser.add_argument("--cat-limit", default=None,
                        help="按类别分层采样上限，如 'PQ:12,APP:8,MH:8,NUM:8'（无 :n 的类别全取；"
                             "每类别先取非私域常识题保对照组）")
    parser.add_argument("--repeats", type=int, default=1, help="每格重复跑次数（R5，>1 报均值+spread）")
    parser.add_argument("--scheme", default="1")
    parser.add_argument("--policy", default="full")
    parser.add_argument("--simulator", default="scripted")
    parser.add_argument("--report-only", action="store_true",
                        help="只读聚合现有 per_task.jsonl 重算 summary（改动六，不跑评测）")
    args = parser.parse_args()
    if args.report_only:
        report_only()
        return
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
