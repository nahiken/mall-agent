"""CLI：检索门禁（改动三）——重摄取 + 逐集合跑 RAG-MCP eval_mall，汇总 hit_rate/MRR。

流程：
1. 可选先调 ingest_kb 摄取（--ingest）。
2. 对 cs_kb / policy_kb 各跑 RAG-MCP `scripts/eval_mall.py`（fake embedding 离线确定性），
   从 `data/eval_rag/golden_retrieval.json` 的 expected_sources 反查 chunks.jsonl 得 expected_chunk_ids。
3. 汇总输出到 data/reports/eval_rag/summary.md。

用法：
    python -m mall_agent.cli.eval_rag --ingest            # 先摄取再评测
    python -m mall_agent.cli.eval_rag                     # 直接评测已摄取的集合
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mall_agent.core.settings import settings

COLLECTIONS = ("cs_kb", "policy_kb")   # 政策/客服知识门禁集合（商品 product_kb 不含 PQ golden）
DATA_DIR = Path(settings.get("data_dir", "data"))
GOLDEN_RETRIEVAL = DATA_DIR / "eval_rag" / "golden_retrieval.json"
REPORT_DIR = DATA_DIR / "reports" / "eval_rag"


def _rag_python() -> str | None:
    rag = settings.rag
    return settings.env("RAG_MCP_PYTHON") or rag.get("python")


def run_eval_mall(collection: str, strategy: str | None = "hybrid") -> dict | None:
    """跑一个集合的检索门禁，返回 report dict。"""
    rag = settings.rag
    python = _rag_python()
    if not python:
        print("  ⚠️ RAG_MCP_PYTHON 未配置，跳过门禁")
        return None
    rag_cwd = Path(rag.get("cwd") or ".")
    index_dir = (rag_cwd / rag.get("index_dir", "data/index")).resolve()
    script = rag_cwd / "scripts" / "eval_mall.py"
    cmd = [python, str(script), "--collection", collection,
           "--test-set", str(GOLDEN_RETRIEVAL.resolve()),
           "--index-dir", str(index_dir), "--embedding-provider", "fake"]
    if strategy:
        cmd += ["--strategy", strategy]
    cmd += ["--json"]
    print(f"  门禁 {collection} ...")
    import os
    result = subprocess.run(
        cmd, cwd=str(rag_cwd),
        env={"PYTHONPATH": rag.get("src"), "PYTHONIOENCODING": "utf-8", **{k: v for k, v in os.environ.items()}},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print("  stderr:", result.stderr[-600:])
        return None
    try:
        return json.loads(result.stdout)
    except Exception:
        print("  stdout:", result.stdout[-600:])
        return None


def main() -> None:
    args = sys.argv[1:]
    if "--ingest" in args:
        from mall_agent.cli import ingest_kb
        ingest_kb.main()
        print("摄取完成。")

    if not GOLDEN_RETRIEVAL.exists():
        print(f"! 检索 golden 不存在：{GOLDEN_RETRIEVAL}")
        print("  请先生成: python -m mall_agent.eval.golden.generator_v3")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    md_lines = ["# 检索门禁报告（RAG-MCP × mall 政策语料）", "",
                f"- golden: {GOLDEN_RETRIEVAL.name}（由 _PQ_TOPICS 导出）", ""]
    for coll in COLLECTIONS:
        rep = run_eval_mall(coll)
        if not rep:
            md_lines.append(f"## {coll}\n- 门禁失败/未运行\n")
            continue
        rows.append((coll, rep["hit_rate"], rep["mrr"], len(rep.get("per_query", []))))
        md_lines.append(f"## {coll}")
        md_lines.append(f"- **hit_rate={rep['hit_rate']:.3f}**  mrr={rep['mrr']:.3f}  "
                        f"({len(rep['per_query'])} 条)")
        fails = [p for p in rep.get("per_query", []) if not p["hit"]]
        for p in rep["per_query"]:
            mark = "PASS" if p["hit"] else "FAIL"
            md_lines.append(f"  - [{mark}] {p['query'][:36]} (expected={p['expected_sources']})")
        md_lines.append("")
    md_lines.insert(2, "| 集合 | hit_rate | MRR | 用例数 |\n|------|---------|-----|--------|\n"
                     + "\n".join(f"| {c} | {h:.3f} | {m:.3f} | {n} |" for c, h, m, n in rows))
    summary_path = REPORT_DIR / "summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    print(f"\n[OK] 报告已保存: {summary_path}")


if __name__ == "__main__":
    main()
