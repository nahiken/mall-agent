"""答案级判答（改动五.1）：规则粗筛 + LLM judge 仲裁 → answer_correct。

双信号：
- `rule_grounding(task, reply, tool_results)` → float|None：key_facts 在回复中的覆盖比（粗筛）。
  **粗筛定位**：自由文本口径（"3-5个工作日"vs"3-5日"）字符串比对会漏报，不做细粒度裁决。
- `llm_judge(question, reference_answer, reply, judge_llm)` → {"verdict", "reason"}：
  correct/partial/wrong + 一句理由。judge 模型=固定同一模型同一 prompt（R2 硬约束，
  由调用方传入 make_llm("judge") 且全格一致）。
- `answer_correct(...)` 融合：rule 与 judge 一致取该值；不一致以 judge 仲裁为准，
  并记 judge_rule_disagree 计数（rule 对确定数字口径/拒答可行使否决，默认关）。

consistency：同题复判差记录作可信度下限（可选，run 两次比对）。
"""
from __future__ import annotations

import json
from typing import Any

from mall_agent.eval.metrics import scoring as _sc  # 复用 _fact_in_reply / 数字归一

# 归因标签（改动六新采集用）
ATTR_FABRICATE = "编造"            # 回复含检索外/无依据断言
ATTR_WRONG_NUM = "错口径"          # 关键数字口径错
ATTR_MISS_POINT = "漏要点"         # 缺 key_facts 覆盖
ATTR_OFF_TOPIC = "答非所问"
ATTR_REFUSE_OK = "拒绝成功"        # 安全/敏感正确地拒绝
ATTR_REFUSE_BAD = "拒绝不当"        # 过度拒绝（本可正常回答）

_VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_SYSTEM = (
    "你是严格的答案判分器。对照【参考答案】评判【客服回复】，只输出 JSON："
    '{"verdict": "correct|partial|wrong", "reason": "一句话中文理由", "attr": "归因"}。\n'
    "归因取值：编造 / 错口径 / 漏要点 / 答非所问 / 拒绝成功 / 拒绝不当 / 正确。\n"
    "规则：数字口径（如天数/工作日/金额）必须与参考答案一致，答成其它数值=wrong。"
    "客服回复需覆盖参考的关键要点；漏任一关键点=partial 或 wrong。"
    "若回复恰当拒绝（如涉及隐私/越权）且参考为拒绝类，则 correct。"
)
_JUDGE_USER = "【问题】\n{question}\n\n【参考答案】\n{reference}\n\n【客服回复】\n{reply}\n\n输出 JSON："


# ------------------------------------------------------------------
# 规则粗筛
# ------------------------------------------------------------------
def rule_grounding(task: Any, reply: str, tool_results: list[tuple[str, str]] | None = None) -> float | None:
    """key_facts 在回复中的覆盖比（无 key_facts→None）。数值口径用 _fact_in_reply 宽松匹配。

    复用 scoring.answer_grounding 的核心，但不要求"必须有 RAG 命中"作为前提——本函数纯粹
    看**回复内容是否覆盖参考要点**（RAG OFF 也判覆盖比，供 OFF 格答对率）。
    """
    key_facts = list(getattr(task, "key_facts", []) or [])
    if not key_facts:
        return None
    reply = reply or ""
    covered = 0
    for kf in key_facts:
        if _sc._fact_in_reply(kf, reply):
            covered += 1
    return round(covered / len(key_facts), 3)


# ------------------------------------------------------------------
# LLM judge
# ------------------------------------------------------------------
def _parse_judge(text: str) -> dict | None:
    try:
        # 剥可能的 ```json 围栏
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(s)
        if data.get("verdict") in _VERDICT_SCORE:
            return {"verdict": data["verdict"], "reason": str(data.get("reason", "")),
                    "attr": str(data.get("attr", ""))}
    except Exception:
        pass
    return None


async def llm_judge(question: str, reference_answer: str, reply: str,
                    judge_llm: Any, retries: int = 2) -> dict:
    """LLM judge 单次判定。judge_llm=固定同一模型（make_llm("judge")）。"""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER.format(
            question=question, reference=reference_answer, reply=reply or "")},
    ]
    last_text = ""
    for _ in range(retries):
        last_text = await judge_llm.acomplete(messages)
        parsed = _parse_judge(last_text)
        if parsed:
            return parsed
    # 解析失败兜底
    return {"verdict": "wrong", "reason": f"judge 输出不可解析: {last_text[:80]}", "attr": "漏要点"}


# ------------------------------------------------------------------
# 融合（实际入口为 async_answer_correct）
# ------------------------------------------------------------------
async def async_answer_correct(question: str, task: Any, reply: str,
                               tool_results: list[tuple[str, str]] | None,
                               judge_llm: Any, *, allow_rule_veto: bool = False,
                               rule: float | None = None) -> dict:
    """计算单任务 answer_correct 全套分项（async，实际入口）。

    返回 dict（供 per_task.jsonl / summary 落账）：
      rule_score / judge_verdict / judge_score / answer_correct(融合) /
      judge_reason / judge_attr / judge_rule_disagree(bool) / grounded(bool)。
    融合：judge 分 0/0.5/1；与 rule 一致取该值，不一致以 judge 仲裁（rule 可否决见 allow_rule_veto）。
    """
    if rule is None:
        rule = rule_grounding(task, reply, tool_results)
    j = await llm_judge(question, task.reference_answer, reply, judge_llm)
    judge_score = _VERDICT_SCORE[j["verdict"]]

    # 融合：默认以 judge 仲裁；rule 否决仅在 allow_rule_veto 且 rule 明确覆盖时覆盖
    verdict = j["verdict"]
    fused = judge_score
    disagree = False
    if rule is not None:
        rule_bin = 1.0 if rule >= 0.5 else 0.0   # 粗筛二值
        judge_bin = 1.0 if judge_score >= 0.5 else 0.0
        disagree = (rule_bin != judge_bin)
        if allow_rule_veto and disagree and rule_bin == 1.0 and judge_score == 0.0:
            # 规则明确覆盖全部要点但 judge 误判 wrong → 规则否决回 1.0
            fused = 1.0
            verdict = "correct"
    return {
        "rule_score": rule,
        "judge_verdict": verdict,
        "judge_score": judge_score,
        "answer_correct": fused,
        "judge_reason": j.get("reason", ""),
        "judge_attr": j.get("attr", ""),
        "judge_rule_disagree": disagree,
        "grounded": bool(tool_results and any(
            n in _sc.RAG_TOOL_NAMES and (res or "").strip()
            and not any(m in res for m in _sc._RAG_NO_HIT_MARKERS) for n, res in (tool_results or []))),
    }
