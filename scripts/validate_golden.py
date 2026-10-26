"""黄金集「标签有效性」校验：把 golden_tool_flow 直接重放一遍，看断言是否真能成立。

为什么需要它：一个任务如果**照着标准答案做也过不了**（订单不存在、参数指向的商品
不存在、key_facts 在工具返回里根本没有），那它测的不是 agent 能力，而是环境缺陷。
这种题在统计上表现为「所有方案都失败」，会被误读成「架构能力不足」。

判据：
- 每步工具必须存在、可调用、返回非空且非「未实现/不存在」类错误
- 事务类：重放后 expected_state 逐字段成立
- 问答类：key_facts 里的数字必须出现在重放得到的工具返回文本中

用法：python scripts/validate_golden.py [dataset ...]
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mall_agent.cli.eval import _register_tools  # noqa: E402
from mall_agent.eval.harness.fixtures import reset_mock_state  # noqa: E402
from mall_agent.eval.harness.probe import MockStateProbe  # noqa: E402
from mall_agent.eval.golden.generator_v3 import load_v3  # noqa: E402
from mall_agent.tools import registry as R  # noqa: E402

ERROR_MARKERS = ("未实现", "不存在", "失败", "错误", "Error", "error")


async def replay_one(task, probe) -> dict:
    reset_mock_state(probe._mock, probe._uid)
    outs: list[str] = []
    problems: list[str] = []
    skipped: list[str] = []
    for st in task.golden_tool_flow:
        spec = R.registry.get(st.tool)
        if spec is None:
            problems.append(f"工具不存在:{st.tool}")
            continue
        if getattr(spec.scope, "domain", "") == "rag":
            # RAG 工具走 MCP stdio 子进程，冷启动可能 1-2 分钟；标签校验只关心
            # 结构化数据（DB）类步骤，检索类步骤跳过并单独计数。
            skipped.append(st.tool)
            continue
        try:
            out = await spec.func(**st.params)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{st.tool} 抛错:{type(e).__name__}:{e}")
            continue
        text = str(out or "")
        outs.append(text)
        if not text.strip():
            problems.append(f"{st.tool} 返回空")
        elif any(m in text for m in ERROR_MARKERS):
            problems.append(f"{st.tool} 返回错误:{text[:40]}")

    joined = " ".join(outs)
    fields = await probe.partial_check(task.expected_state)
    state_fail = [k for k, v in fields.items() if not v]

    facts = list(getattr(task, "key_facts", []) or [])
    missing = []
    compact = joined.replace(",", "").replace("¥", "")
    for f in facts:
        digits = re.findall(r"\d+(?:\.\d+)?", str(f))
        if digits:
            if not all(d in compact for d in digits):
                missing.append(f)
        elif "".join(str(f).split()) not in "".join(joined.split()):
            missing.append(f)

    return {
        "task_id": task.task_id,
        "steps": len(task.golden_tool_flow),
        "no_flow": len(task.golden_tool_flow) == 0,
        "state_fail": state_fail,
        "facts_missing": missing,
        "facts_unverified": bool(missing) and bool(skipped),
        "problems": problems,
        "skipped_rag": skipped,
        "has_assert": bool(fields),
        "has_facts": bool(facts),
    }


async def main(paths: list[str]) -> None:
    _register_tools()
    from mall_agent.mall_api.client import get_mall_client
    mock = get_mall_client().ensure_mock_sync()
    probe = MockStateProbe(mock, user_id="1")

    for p in paths:
        tasks = load_v3(Path(p))
        rows = [await replay_one(t, probe) for t in tasks]
        bad = [r for r in rows
               if r["state_fail"] or r["problems"]
               or (r["facts_missing"] and not r.get("facts_unverified"))]
        unver = [r for r in rows if r.get("facts_unverified")]
        noflow = [r for r in rows if r["no_flow"]]
        unjudge = [r for r in rows if not r["has_assert"] and not r["has_facts"]]
        print(f"\n=== {Path(p).name}：{len(rows)} 条")
        print(f"  可判(有断言或 key_facts)：{len(rows) - len(unjudge)}"
              f"  不可判：{len(unjudge)}")
        print(f"  无黄金流：{len(noflow)}   重放不通过：{len(bad)}"
              f"   检索类未验证(kv)：{len(unver)}")
        for r in bad:
            print(f"   ! {r['task_id']:10s} state_fail={r['state_fail']} "
                  f"facts_missing={r['facts_missing']} problems={r['problems']}")
        out = Path("data/reports") / f"golden_validation_{Path(p).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  明细 → {out}")


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "data/golden/golden_v1.jsonl", "data/golden/golden_v2.jsonl",
        "data/golden/golden_v3.jsonl", "data/golden/golden_v5.jsonl",
    ]
    asyncio.run(main(args))
