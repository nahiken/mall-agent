"""评测审计：按「判定口径」分层重算 + 数据集质量统计 + 区间估计。

解决的问题：旧报告只用 probe 的 outcome 判通过，而 probe 在空断言下恒真，
于是 0.857 这种数字测的是「agent 有没有说话」，不是「任务有没有完成」。

本脚本做四件事：
1. 分层判定 L0..L3（越往后越严格），同一批 raw 数据算出四组成功率，
   用来回答「结论是否依赖判定口径」——敏感性分析。
2. Wilson 区间 + 配对 McNemar 精确检验 + 配对 bootstrap，替代「两个点估计比大小」。
3. 题目质量：难度分布、死题（全过/全不过）、近重复、断言覆盖率。
4. 成本口径交叉核对：LLM 调用数（比 steps 在方案间可比）、工具调用数、token、
   单题答对成本。

用法：
    python scripts/eval_audit.py --dataset data/golden/golden_v5.jsonl \
        --raw data/reports/raw_20261022_xxxx.jsonl [--raw ...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------- 统计工具
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """配对二分类的精确 McNemar（双侧二项检验）。b/c 为两方案不一致的格数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_diff(a: list[int], b: list[int], iters: int = 5000,
                   seed: int = 7) -> tuple[float, float, float]:
    """配对 bootstrap：同一批题上两方案差值（a-b）的 95% 区间。"""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    point = sum(a[i] - b[i] for i in range(n)) / n
    return point, diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]


def cronbach_alpha(matrix: list[list[int]]) -> float:
    """题目间一致性（K-R 20 近似）。matrix: 题 × 受测（方案）。"""
    if not matrix or not matrix[0]:
        return 0.0
    k = len(matrix)
    var_items = [sum((x - sum(it) / len(it)) ** 2 for x in it) / len(it) for it in matrix]
    totals = [sum(matrix[i][j] for i in range(k)) for j in range(len(matrix[0]))]
    m = sum(totals) / len(totals)
    var_total = sum((t - m) ** 2 for t in totals) / len(totals)
    if var_total == 0:
        return 0.0
    return (k / (k - 1)) * (1 - sum(var_items) / var_total)


# ---------------------------------------------------------------- 数据集质量
def _bigrams(s: str) -> set[str]:
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def dup_clusters(tasks: list[dict], thr: float = 0.6) -> list[list[str]]:
    """按 opener 字符二元组 Jaccard 聚类，识别模板刷题造成的近重复。"""
    sets = {t["task_id"]: _bigrams(t.get("opener", "")) for t in tasks}
    parent = {k: k for k in sets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = list(sets)
    for a, b in combinations(ids, 2):
        A, B = sets[a], sets[b]
        if not A or not B:
            continue
        j = len(A & B) / len(A | B)
        if j >= thr:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def dataset_stats(tasks: list[dict]) -> dict:
    n = len(tasks)
    has_state = [t for t in tasks if any(v is not None for v in (t.get("expected_state") or {}).values())]
    has_facts = [t for t in tasks if t.get("key_facts")]
    judgeable = {t["task_id"] for t in has_state + has_facts}
    no_flow = [t for t in tasks if not t.get("golden_tool_flow")]
    dups = dup_clusters(tasks)
    dup_items = sum(len(g) for g in dups)
    return {
        "n": n,
        "judgeable": len(judgeable),
        "judgeable_rate": round(len(judgeable) / n, 3),
        "state_assertions": len(has_state),
        "key_facts": len(has_facts),
        "no_golden_flow": len(no_flow),
        "cover_domain": dict(Counter(t.get("domain") for t in tasks)),
        "cover_type": dict(Counter(t.get("task_type") for t in tasks)),
        "cover_category": dict(Counter(t.get("category") or "-" for t in tasks)),
        "dup_clusters": dups,
        "dup_rate": round(dup_items / n, 3),
        "subgoals_per_task": round(sum(len(t.get("required_subgoals") or []) for t in tasks) / n, 2),
        "flow_len_avg": round(sum(len(t.get("golden_tool_flow") or []) for t in tasks) / n, 2),
    }


# ---------------------------------------------------------------- 分层判定
def _fact_hit(fact: str, reply: str) -> bool:
    text = re.sub(r"[,\s]", "", reply)
    fact = re.sub(r"[,\s]", "", str(fact))
    if fact in text:
        return True
    nums = re.findall(r"\d+(?:\.\d+)?", fact)
    if nums:
        return all(n in text for n in nums)
    return False


def judge_trial(task: dict, r: dict) -> dict:
    """返回 L0..L3 的 0/1。含义见模块 docstring。"""
    sig = r.get("success_signal") or {}
    sg = sig.get("subgoal_recall") or 0.0
    tf = sig.get("tool_flow_match") or 0.0
    pa = sig.get("param_accuracy")
    pa = 1.0 if pa is None else pa
    fc = sig.get("functional_correct")
    ac = r.get("answer_ctx") or {}
    reply = ac.get("reply_all") or ac.get("reply") or ""
    tools = [x.get("name") for x in (ac.get("tool_records") or [])]

    l0 = 1 if r.get("outcome") == "success" else 0
    l1 = 1 if (sg >= 1.0 and tf >= 0.8) else 0
    l2 = 1 if (l1 and pa >= 1.0) else 0

    state = task.get("expected_state") or {}
    need_state = any(v is not None for v in state.values())
    facts = task.get("key_facts") or []
    forbidden = [c.split(":", 1)[1] for c in (task.get("constraints") or [])
                 if c.startswith("forbidden_tool:")]

    l3 = l2
    if l3 and need_state and fc is not None and fc < 1.0:
        l3 = 0
    if l3 and facts and not all(_fact_hit(f, reply) for f in facts):
        l3 = 0
    if l3 and any(t in forbidden for t in tools):
        l3 = 0
    # L4：任务级判定，只看「做成了没有」，不要求走我的黄金路径。
    # 换路径但结果正确（例如 search_product 已带价格、没再调 detail）不该判失败——
    # 那是数据集过拟合单一路径，属于构念效度问题。
    l4 = 1
    if need_state and (fc or 0) < 1.0:
        l4 = 0
    if facts and not all(_fact_hit(f, reply) for f in facts):
        l4 = 0
    if any(t in forbidden for t in tools):
        l4 = 0
    if not need_state and not facts and (sg < 1.0):
        l4 = 0

    # 无断言、无 key_facts、无禁止工具的任务：L3 退化为 L2（并单独统计占比）
    return {"L0": l0, "L1": l1, "L2": l2, "L3": l3, "L4": l4}


def load_rows(paths: list[str]) -> list[dict]:
    """读多份 raw；同一 (task, scheme) 出现多次时后者覆盖前者（补跑覆盖旧结果）。"""
    latest: dict[tuple[str, str], dict] = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            latest[(r.get("golden_task_id"), str(r.get("agent_id")))] = r
    return list(latest.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw", action="append", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    by_id = {t["task_id"]: t for t in tasks}
    rows = [r for r in load_rows(args.raw) if r.get("golden_task_id") in by_id]

    print(f"数据集 {Path(args.dataset).name}：{len(tasks)} 条；trial {len(rows)} 条")
    ds = dataset_stats(tasks)
    print("\n## 数据集质量")
    for k, v in ds.items():
        if k == "dup_clusters":
            print(f"  dup_clusters: {len(v)} 组，示例 {v[:3]}")
        else:
            print(f"  {k}: {v}")

    # 每题 × 每方案
    per: dict[tuple[str, str], dict] = {}
    for r in rows:
        per[(r["golden_task_id"], str(r["agent_id"]))] = judge_trial(by_id[r["golden_task_id"]], r)
    schemes = sorted({str(r["agent_id"]) for r in rows})
    common = [t["task_id"] for t in tasks
              if all((t["task_id"], s) in per for s in schemes)]
    print(f"\n共同完成的任务：{len(common)} 条 × 方案 {schemes}")

    print("\n## 分层成功率（同一批 trial，只改判定口径）")
    print("| 方案 | L0 probe | L1 工具流 | L2 +参数 | L3 +答案/状态 | L4 任务级(不看路径) |")
    print("|---|---|---|---|---|---|")
    for s in schemes:
        cells = []
        for lv in ("L0", "L1", "L2", "L3", "L4"):
            k = sum(per[(t, s)][lv] for t in common)
            lo, hi = wilson(k, len(common))
            cells.append(f"{k}/{len(common)} ({k/len(common):.3f}) [{lo:.2f},{hi:.2f}]")
        print(f"| {s} | " + " | ".join(cells) + " |")

    print("\n## 两两配对比较（L3 口径）")
    for a, b in combinations(schemes, 2):
        va = [per[(t, a)]["L4"] for t in common]
        vb = [per[(t, b)]["L4"] for t in common]
        b01 = sum(1 for x, y in zip(va, vb) if x == 0 and y == 1)
        c10 = sum(1 for x, y in zip(va, vb) if x == 1 and y == 0)
        p = mcnemar_exact(b01, c10)
        pt, lo, hi = bootstrap_diff(va, vb)
        print(f"  {a} vs {b}: 差 {pt:+.3f} [95% {lo:+.3f},{hi:+.3f}]；"
              f"不一致 {b01}/{c10}；McNemar p={p:.3g}")

    print("\n## 题目难度与区分度（L3）")
    hard, easy, dead = [], [], []
    for t in common:
        vals = [per[(t, s)]["L4"] for s in schemes]
        rate = sum(vals) / len(vals)
        if rate == 0:
            dead.append(t)
        elif rate == 1:
            easy.append(t)
        elif rate < 0.5:
            hard.append(t)
    print(f"  全不过(0/{len(schemes)}) {len(dead)}：{dead}")
    print(f"  全过({len(schemes)}/{len(schemes)}) {len(easy)}：{easy}")
    print(f"  部分通过({len(hard)})：{hard}")
    mat = [[per[(t, s)]["L4"] for s in schemes] for t in common]
    print(f"  题目间一致性 KR-20 ≈ {cronbach_alpha(mat):.2f}（3 个受测太少，仅作参考）")

    # 成本口径
    print("\n## 成本/轨迹口径（L3 通过题）")
    for s in schemes:
        rs = [r for r in rows if str(r["agent_id"]) == s
              and per.get((r["golden_task_id"], s), {}).get("L4") == 1]
        spend_all = sum(r["cost"].get("external_cost_rmb", 0) for r in rows
                        if str(r["agent_id"]) == s)
        if not rs:
            print(f"  方案 {s}: 无通过题")
            continue
        llm_calls = [sum(1 for x in (r.get("trajectory") or []) if x.get("type") == "llm") for r in rs]
        tool_calls = [r["cost"].get("tool_calls", 0) for r in rs]
        tin = [r["cost"].get("agent_in", 0) for r in rs]
        tout = [r["cost"].get("agent_out", 0) for r in rs]
        cost = [r["cost"].get("external_cost_rmb", 0) for r in rs]
        lat = [r["latency_ms"].get("total") or 0 for r in rs]
        n = len(rs)
        print(f"  方案 {s}: n={n} LLM调用 {sum(llm_calls)/n:.2f} 工具 {sum(tool_calls)/n:.2f} "
              f"输入token {sum(tin)/n:.0f} 输出token {sum(tout)/n:.0f} "
              f"时延 {sum(lat)/n/1000:.1f}s 单题成本 ¥{sum(cost)/n:.4f} "
              f"单题答对成本 ¥{spend_all/max(1,len(rs)):.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"dataset": ds, "per_task": {f"{k[0]}|{k[1]}": v for k, v in per.items()}},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n明细 → {args.out}")


if __name__ == "__main__":
    main()
