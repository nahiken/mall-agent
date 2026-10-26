"""RAG 增益实验的审计：口径核对 + 判分一致性 + 配对区间。

三个要核的点：
1. summary.md 里叫「答对率」的列其实是 judge 的**部分分均值**（0/0.5/1），
   不是二分类正确率。同一批数据还能用确定性 rule_score（key_facts 覆盖率）算一遍，
   两个口径都要给，否则读者无法判断增益是来自检索还是来自判分宽松。
2. 「grounded 0→0.94」在代码里是 `answer_grounding is not None`，即
   "检索有没有命中且任务有 key_facts"——这是**检索可用率**，不是答案有据率。
   真正衡量答案有据的应该是 answer_grounding 的取值本身。
3. 每格 72 条来自 36 题 × 2 rep，rep 之间不独立。配对检验要在题层面做，
   或者按 rep 配对并在结论里标注聚类。

用法：python scripts/audit_rag_eval.py [per_task.jsonl]
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

PATH = Path("data/reports/exp_rag_eval/per_task.jsonl")


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, min(1.0, (c + h) / d))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_mean_diff(a: list[float], b: list[float], iters: int = 4000,
                        seed: int = 11) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(a[i] - b[i] for i in idx) / n)
    vals.sort()
    return (sum(x - y for x, y in zip(a, b)) / n, vals[int(0.025 * iters)],
            vals[int(0.975 * iters) - 1])


def weighted_kappa(x: list[int], y: list[int], levels: int = 3) -> float:
    """二次加权 Cohen's kappa（有序三档：0 / 0.5 / 1）。"""
    n = len(x)
    if n == 0:
        return 0.0
    obs = [[0] * levels for _ in range(levels)]
    for a, b in zip(x, y):
        obs[a][b] += 1
    row = [sum(obs[i]) for i in range(levels)]
    col = [sum(obs[i][j] for i in range(levels)) for j in range(levels)]
    num = den = 0.0
    for i in range(levels):
        for j in range(levels):
            w = ((i - j) ** 2) / ((levels - 1) ** 2)
            exp = row[i] * col[j] / n
            num += w * obs[i][j]
            den += w * exp
    return 1 - num / den if den else 0.0


def lvl(score: float) -> int:
    return 0 if score < 0.25 else (1 if score < 0.75 else 2)


def main(path: Path) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    cells = sorted({r["cell"] for r in rows})
    print(f"{path}：{len(rows)} 条，{len({r['golden_task_id'] for r in rows})} 题 × 2 rep")

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)

    print("\n## 三种口径下的每格表现（均值，括号为二分类 Wilson 95% CI）")
    print("| cell | n | judge 部分分均值 | judge 判正(>=0.5) | rule 确定分均值 | rule 判正 | 检索命中率 | 平均 grounding 值 |")
    print("|---|---|---|---|---|---|---|---|")
    stats = {}
    for c in cells:
        rs = by_cell[c]
        j = [r["judge"]["judge_score"] for r in rs]
        ru = [r["judge"]["rule_score"] for r in rs]
        jb = [1 if v >= 0.5 else 0 for v in j]
        rb = [1 if v >= 0.5 else 0 for v in ru]
        gr = [r["success_signal"].get("answer_grounding") for r in rs]
        hit = [1 if g is not None else 0 for g in gr]
        gvals = [g for g in gr if g is not None]
        lo, hi = wilson(sum(jb), len(jb))
        lo2, hi2 = wilson(sum(rb), len(rb))
        stats[c] = dict(jm=sum(j) / len(j), rm=sum(ru) / len(ru),
                        jb=jb, rb=rb, hit=sum(hit) / len(hit),
                        gv=sum(gvals) / len(gvals) if gvals else 0.0)
        print(f"| {c} | {len(rs)} | {sum(j)/len(j):.3f} | {sum(jb)/len(jb):.3f} "
              f"[{lo:.2f},{hi:.2f}] | {sum(ru)/len(ru):.3f} | {sum(rb)/len(rb):.3f} "
              f"[{lo2:.2f},{hi2:.2f}] | {sum(hit)/len(hit):.2f} | "
              f"{stats[c]['gv']:.3f} |")

    print("\n## 判分一致性（rule vs judge，二次加权 kappa）")
    agree = [r["judge"]["judge_rule_disagree"] for r in rows]
    print(f"  显式不一致条数：{sum(agree)}/{len(agree)} = {sum(agree)/len(agree):.0%}")
    for c in cells:
        rs = by_cell[c]
        x = [lvl(r["judge"]["rule_score"]) for r in rs]
        y = [lvl(r["judge"]["judge_score"]) for r in rs]
        print(f"  {c}: kappa={weighted_kappa(x, y):+.2f}")

    print("\n## RAG 开关的配对增益（同 provider×style）")
    pairs = [("ds_rag_off", "ds_rag_on"), ("ds_rag_off_concise", "ds_rag_on_concise")]
    # 以 (task, rep) 为配对单位
    index = defaultdict(dict)
    for r in rows:
        index[r["golden_task_id"]][f"{r['cell']}_r{r['rep']}"] = r
    for off, on in pairs:
        keys = sorted({k for v in index.values() for k in v if k.startswith(off)})
        a, b, jd = [], [], []
        for t, v in index.items():
            for k in v:
                if k.startswith(off):
                    suf = k[len(off):]
                    ok_on = f"{on}{suf}"
                    if ok_on in v:
                        a.append(v[ok_on]["judge"]["judge_score"])
                        b.append(v[k]["judge"]["judge_score"])
                        jd.append((v[ok_on]["judge"]["rule_score"], v[k]["judge"]["rule_score"]))
        if not a:
            continue
        pt, lo, hi = bootstrap_mean_diff(a, b)
        jb_on = [1 if x >= 0.5 else 0 for x, _ in jd]
        jb_off = [1 if y >= 0.5 else 0 for _, y in jd]
        b01 = sum(1 for x, y in zip(jb_on, jb_off) if x == 0 and y == 1)
        c10 = sum(1 for x, y in zip(jb_on, jb_off) if x == 1 and y == 0)
        r_pt, r_lo, r_hi = bootstrap_mean_diff([x for x, _ in jd], [y for _, y in jd])
        print(f"  {off} → {on}：judge Δ={pt:+.3f} [95% {lo:+.3f},{hi:+.3f}]；"
              f"rule Δ={r_pt:+.3f} [95% {r_lo:+.3f},{r_hi:+.3f}]；"
              f"McNemar(judge 判正) p={mcnemar_exact(b01, c10):.2g}，不一致 {b01}/{c10}，n={len(a)}")

    print("\n## 单题答对成本（元；按 judge 部分分折算，bootstrap 95% CI）")
    for c in cells:
        rs = by_cell[c]
        spend = sum(r["cost"].get("external_cost_rmb", 0) for r in rs)
        credit = sum(r["judge"]["judge_score"] for r in rs)
        cpc = spend / credit if credit else float("nan")
        rng = random.Random(3)
        vals = []
        for _ in range(2000):
            idx = [rng.randrange(len(rs)) for _ in range(len(rs))]
            s = sum(rs[i]["cost"].get("external_cost_rmb", 0) for i in idx)
            k = sum(rs[i]["judge"]["judge_score"] for i in idx)
            vals.append(s / k if k else 0)
        vals.sort()
        print(f"  {c}: ¥{cpc:.4f} [95% ¥{vals[50]:.4f},{vals[1950]:.4f}]，"
              f"总花费 ¥{spend:.2f}／有效答对数 {credit:.1f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PATH)
