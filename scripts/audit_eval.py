"""审计评测结果：区分「真实通过」与「默认通过」。

背景：`custom_runner` 在 `probe.check(expected_state)` 返回 True 时立即判成功；
而黄金任务里多数 `expected_state` 全为 null，probe 空断言直接返回 True，
于是这些任务**第一轮就默认通过**，与 Agent 实际表现无关。

本脚本做三件事：
1. 统计每条任务是否有非空断言（断言覆盖率）；
2. 用 success_signal 里的 subgoal_recall / tool_flow_match 重算「严格通过」；
3. 对比 outcome（旧口径）与严格口径，暴露差异。

用法：
    python scripts/audit_eval.py \
        --dataset data/golden/golden_v1.jsonl \
        --raw data/reports/raw_20261021_124216.jsonl

    # 多个 raw 一起看
    python scripts/audit_eval.py --dataset data/golden/golden_v2.jsonl \
        --raw data/reports/raw_a.jsonl data/reports/raw_b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

STRICT_SG = 0.8       # 子目标召回阈值
STRICT_TF = 0.8       # 工具流匹配阈值


def load_tasks(path: str) -> dict[str, dict]:
    tasks = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            tasks[t["task_id"]] = t
    return tasks


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def asserted_fields(task: dict) -> list[str]:
    es = task.get("expected_state") or {}
    return [k for k, v in es.items() if v is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description="评测结果审计（默认通过 vs 严格通过）")
    ap.add_argument("--dataset", required=True, help="黄金任务 jsonl")
    ap.add_argument("--raw", required=True, nargs="+", help="raw_*.jsonl（可多个）")
    ap.add_argument("--sg", type=float, default=STRICT_SG, help="subgoal_recall 阈值")
    ap.add_argument("--tf", type=float, default=STRICT_TF, help="tool_flow_match 阈值")
    args = ap.parse_args()

    tasks = load_tasks(args.dataset)
    rows = load_rows(args.raw)
    if not rows:
        print("没有 trial 数据")
        return 1

    # 1. 断言覆盖率
    vacuous = [tid for tid, t in tasks.items() if not asserted_fields(t)]
    print(f"数据集: {args.dataset}（{len(tasks)} 条）")
    print(f"带非空状态断言: {len(tasks) - len(vacuous)} 条；"
          f"空断言（会默认通过）: {len(vacuous)} 条")
    if vacuous:
        print("  空断言任务:", ", ".join(sorted(vacuous)))

    # 2. 按方案重算
    agg = defaultdict(lambda: defaultdict(list))
    per_task = defaultdict(dict)
    for r in rows:
        a = str(r.get("agent_id"))
        sig = r.get("success_signal") or {}
        sg = sig.get("subgoal_recall") or 0.0
        tf = sig.get("tool_flow_match") or 0.0
        strict = sg >= args.sg and tf >= args.tf
        agg[a]["pass"].append(1 if r.get("outcome") == "success" else 0)
        agg[a]["strict"].append(1 if strict else 0)
        agg[a]["sg"].append(sg)
        agg[a]["tf"].append(tf)
        per_task[str(r.get("golden_task_id"))][a] = (r.get("outcome") == "success", strict)

    print("\n方案 | n | 旧口径通过率 | 严格通过率 | 平均子目标召回 | 平均工具流匹配")
    print("---|---|---|---|---|---")
    for a in sorted(agg):
        d = agg[a]
        n = len(d["pass"])
        print(f"{a} | {n} | {sum(d['pass'])/n:.3f} | "
              f"**{sum(d['strict'])/n:.3f}** | {sum(d['sg'])/n:.2f} | {sum(d['tf'])/n:.2f}")

    # 3. 差异最大的任务（旧口径过、严格口径不过）
    print("\n旧口径判过但严格口径判不过的任务（默认通过的证据）:")
    for tid in sorted(per_task):
        cells = per_task[tid]
        bad = [a for a, (p, s) in cells.items() if p and not s]
        if bad:
            print(f"  {tid}: 方案 {', '.join(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
