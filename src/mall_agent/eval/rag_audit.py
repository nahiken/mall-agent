"""RAGAS 离线审计层（改动八 / T12）：作 rule+judge 之外的第三交叉信号。

读取 exp_rag_eval 的 per_task.jsonl（含 question/reply/reference_answer/检索 tool_records），
按格聚成 RAGAS 指标，不重跑 agent 链。RAGAS 装在**隔离 venv**（.venv-ragas，锁 ragas==0.4.3），
本模块只 import 标准库 + 可选 ragas，绝不 import mall_agent 重型链路，故可在该 venv 直接跑。

指标语义（对照 RAGAS 文档）：
- FA  faithfulness     回复是否忠于检索到的上下文（无外部编造）—— 需 retrieved_contexts
- AR  answer_relevancy 回复是否切题（问答一致性）—— 无需 context
- CP  context_precision 检索上下文与参考答案的相关精度 —— 需 reference_answer
- CR  context_recall    黄金上下文被检索命中的比例 —— 需 reference_contexts(gold)
- AC  answer_correctness 回复与参考答案的语义+事实正确性 —— 无需 context

RAG OFF（无检索 tool_records → retrieved_contexts 空）：只产 AR / AC（FA/CP 缺 contexts 语义 N/A，
CR 缺检索不适用）；OFF 格的 AR/AC 恰是观察"无检索下回答是否仍正确/切题（先验/幻觉）"。

用法（隔离 venv，mall-agent 根）：
    .venv-ragas/Scripts/python.exe -m mall_agent.eval.rag_audit \\
        --per-task data/reports/exp_rag_eval/per_task.jsonl \\
        [--gold data/eval_rag/golden_retrieval.json] [--kb data/kb_src] \\
        [--cells ds_rag_on] [--limit 3]
输出：data/reports/exp_rag_eval/ragas_audit.{json,md}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = ROOT / "data" / "reports" / "exp_rag_eval"

RAG_TOOLS = {"query_policy", "query_cs_knowledge", "query_knowledge_hub", "verify_policy"}
# 每类任务希望跑的指标
ON_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness")
OFF_METRICS = ("answer_relevancy", "answer_correctness")

# ----------------------------------------------------------------------
# per_task.jsonl 解析
# ----------------------------------------------------------------------
def load_per_task(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _extract_citations(text: str) -> list[str]:
    """从 RAG 工具返回文本里抽 citations[].snippet（JSON 块）作为检索到的上下文。"""
    out: list[str] = []
    if not text:
        return out
    # 结尾 References (JSON) 块里的 citations
    for m in re.finditer(r'"snippet"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        out.append(m.group(1).encode().decode("unicode_escape", errors="replace"))
    if out:
        return out
    # 兜底：抽取 "> 引用原文" 行
    for m in re.finditer(r"^>\s*(.+)$", text, re.M):
        line = m.group(1).strip()
        if len(line) >= 8:
            out.append(line)
    return out


def retrieved_contexts(row: dict) -> list[str]:
    """该行 agent 实际检索到的上下文（RAG tool 的命中 snippet，去重保序）。"""
    ctx = row.get("answer_ctx") or {}
    seen: list[str] = []
    for tr in ctx.get("tool_records", []) or []:
        if tr.get("name") not in RAG_TOOLS:
            continue
        for snip in _extract_citations(tr.get("result") or ""):
            if snip and snip not in seen:
                seen.append(snip)
    return seen


def _hub_dir(collection: str) -> str:
    return collection[:-3] if collection.endswith("_kb") else collection  # cs_kb -> cs


def reference_contexts(row: dict, gold: dict | None, kb_root: Path | None) -> list[str]:
    """黄金上下文（CR 用）：golden_retrieval 的 expected_sources → kb_src 原文全文。"""
    ctx = row.get("answer_ctx") or {}
    task_id = ctx.get("task_id") or row.get("golden_task_id", "")
    if not gold or kb_root is None or not task_id:
        return []
    case = next((c for c in gold if c.get("task_id") == task_id), None)
    if not case:
        return []
    hub = _hub_dir(case.get("collection", ""))
    texts = []
    for src in case.get("expected_sources", []) or []:
        p = kb_root / hub / src
        if p.exists():
            texts.append(p.read_text(encoding="utf-8").strip())
    return texts


# ----------------------------------------------------------------------
# RAGAS 装配（隔离 venv 里有 ragas + langchain_ollama）
# ----------------------------------------------------------------------
def build_runtime() -> tuple[object, object, dict]:
    """返回 (llm, embeddings, metric_registry)。metric_registry: name -> SingleTurn metric 实例。

    仅 ON 指标装配（OFF 只需 AR/AC，两者在 ON_METRICS 内）。
    """
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        import os
        load_dotenv(ROOT / ".env")
        from ragas.llms import llm_factory
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from langchain_ollama import OllamaEmbeddings
    except ImportError as e:
        raise SystemExit(
            f"RAGAS 依赖缺失（应在 .venv-ragas 隔离 venv 跑）: {e}\n"
            "安装: ./.venv-ragas/Scripts/python.exe -m pip install ragas==0.4.3 langchain-ollama"
        ) from e

    key = os.getenv("LLM_API_KEY", "")
    if not key:
        raise SystemExit("! .env 缺 LLM_API_KEY（DeepSeek judge 判分用）")
    llm = llm_factory(model="deepseek-chat",
                      client=OpenAI(api_key=key, base_url="https://api.deepseek.com"))
    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11451"))

    # FA/AR/CP/CR 直接用单例 metric 挂 llm/embeddings
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.metrics._answer_correctness import AnswerCorrectness
    from ragas.metrics._answer_similarity import AnswerSimilarity
    faithfulness.llm = llm
    answer_relevancy.llm, answer_relevancy.embeddings = llm, emb
    context_precision.llm = llm
    context_recall.llm = llm
    as_ = AnswerSimilarity(); as_.llm, as_.embeddings = llm, emb
    ac = AnswerCorrectness(answer_similarity=as_); ac.llm, ac.embeddings = llm, emb
    registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": ac,
    }
    return llm, emb, registry


def _make_sample(row: dict, retrieved: list[str], reference_ctx: list[str]):
    from ragas.dataset_schema import SingleTurnSample
    ctx = row.get("answer_ctx") or {}
    return SingleTurnSample(
        user_input=ctx.get("question", ""),
        response=ctx.get("reply", "") or "",
        reference=ctx.get("reference_answer", "") or None,
        retrieved_contexts=retrieved or None,
        reference_contexts=reference_ctx or None,
    )


# ----------------------------------------------------------------------
# 单行评分
# ----------------------------------------------------------------------
def score_row(row: dict, metrics: dict, gold: dict | None, kb_root: Path | None) -> dict:
    """返回 {task_id, cell, rag_enabled, metric: value|None}。rag_off 只评 AR/AC。"""
    rag_enabled = bool((row.get("answer_ctx") or {}).get("rag_enabled", True))
    which = ON_METRICS if rag_enabled else OFF_METRICS
    retrieved = retrieved_contexts(row)
    ref_ctx = reference_contexts(row, gold, kb_root)
    sample = _make_sample(row, retrieved, ref_ctx)
    out: dict = {
        "task_id": row.get("golden_task_id") or (row.get("answer_ctx") or {}).get("task_id", "?"),
        "cell": row.get("cell", "?"), "rag_enabled": rag_enabled,
        "n_retrieved": len(retrieved), "n_reference_ctx": len(ref_ctx),
    }
    for name in which:
        m = metrics[name]
        try:
            out[name] = round(float(m.single_turn_score(sample)), 3)
        except Exception as e:  # 单条失败不拖垮整格
            out[name] = None
            out.setdefault("errors", []).append(f"{name}: {type(e).__name__}: {e}")
    return out


def _task_meta_avg(results: list[dict], key: str) -> float | None:
    vals = [r[key] for r in results]
    return round(sum(vals) / len(vals), 2) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS 离线审计（T12）")
    ap.add_argument("--per-task", type=Path, default=DEFAULT_REPORT_DIR / "per_task.jsonl")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "eval_rag" / "golden_retrieval.json")
    ap.add_argument("--kb", type=Path, default=ROOT / "data" / "kb_src")
    ap.add_argument("--cells", default="ds_rag_on,ds_rag_on_concise,ds_rag_off,ds_rag_off_concise",
                    help="只审计这些格（默认 4 个 DeepSeek 格，省 API）")
    ap.add_argument("--limit", type=int, default=0, help="每格最多审计任务数（冒烟）")
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    args = ap.parse_args()

    if not args.per_task.exists():
        raise SystemExit(f"! {args.per_task} 不存在（先跑 T11 产物）")
    cells_wanted = {c.strip() for c in args.cells.split(",") if c.strip()}
    rows = [r for r in load_per_task(args.per_task) if (r.get("cell") or "") in cells_wanted]
    if args.limit:
        # 每格截断
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            c = r["cell"]
            if seen.get(c, 0) < args.limit:
                kept.append(r)
                seen[c] = seen.get(c, 0) + 1
        rows = kept
    if not rows:
        raise SystemExit("! 匹配格为空（检查 --cells）")
    print(f"审计 {len(rows)} 行（格 {sorted(cells_wanted)}）...")

    gold = json.loads(args.gold.read_text(encoding="utf-8")) if args.gold.exists() else None
    _, _, registry = build_runtime()

    results = []
    for i, row in enumerate(rows, 1):
        r = score_row(row, registry, gold, args.kb if args.kb.exists() else None)
        results.append(r)
        print(f"  [{i}/{len(rows)}] {r['cell']} {r['task_id']} 检索={r['n_retrieved']} "
              + " ".join(f"{m}={r.get(m)}" for m in (ON_METRICS if r['rag_enabled'] else OFF_METRICS)))
        sys.stdout.flush()

    args.out.mkdir(parents=True, exist_ok=True)
    # 每格聚合（含任务粒度原始值）
    by_cell: dict[str, list] = {}
    for r in results:
        by_cell.setdefault(r["cell"], []).append(r)
    summary = {}
    for cell, rs in by_cell.items():
        agg_row: dict = {"n": len(rs)}
        for name in ON_METRICS:
            vals = [x[name] for x in rs if name in x and x[name] is not None]
            agg_row[name] = round(sum(vals) / len(vals), 3) if vals else None
            agg_row[f"{name}_n"] = len(vals)
        agg_row["avg_retrieved_ctx"] = _task_meta_avg(rs, "n_retrieved")
        summary[cell] = agg_row
    json_path = args.out / "ragas_audit.json"
    json_path.write_text(json.dumps({
        "source": str(args.per_task), "metrics": list(ON_METRICS),
        "summary": summary, "detail": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # Markdown
    md = ["# RAGAS 离线审计（T12 第三交叉信号）", "",
          f"- 数据源: `{args.per_task}`（T11 per_task，离线重算）",
          "- judge LLM=DeepSeek-chat；embeddings=Ollama nomic-embed-text（本地 11451）",
          "- RAG ON 产 FA/AR/CP/CR/AC；OFF 无 contexts 只产 AR/AC（FA/CP/CR N/A）", ""]
    md.append("## 每格聚合")
    md.append("")
    md.append("| 格 | n | FA | AR | CP | CR | AC | 检索ctx均 |")
    md.append("|----|---|----|----|----|----|----|----|")
    for cell in sorted(summary):
        s = summary[cell]
        def f(k):
            return "-" if s.get(k) is None else f"{s[k]:.3f}"
        md.append(f"| {cell} | {s['n']} | {f('faithfulness')} | {f('answer_relevancy')} | "
                  f"{f('context_precision')} | {f('context_recall')} | {f('answer_correctness')} | "
                  f"{s.get('avg_retrieved_ctx') or '-'} |")
    md.append("")
    md.append("## 任务粒度")
    md.append("")
    md.append("| 格 | task | rag | FA | AR | CP | CR | AC | 检索ctx |")
    md.append("|----|------|-----|----|----|----|----|----|----|")
    for r in results:
        def f2(k):
            v = r.get(k)
            return "-" if v is None else f"{v:.3f}"
        md.append(f"| {r['cell']} | {r['task_id']} | {'ON' if r['rag_enabled'] else 'OFF'} | "
                  f"{f2('faithfulness')} | {f2('answer_relevancy')} | {f2('context_precision')} | "
                  f"{f2('context_recall')} | {f2('answer_correctness')} | {r['n_retrieved']} |")
    md_path = args.out / "ragas_audit.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[OK] {json_path}\n[OK] {md_path}")


if __name__ == "__main__":
    main()
