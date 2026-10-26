"""评分模块：功能正确性 / 子目标召回 / 轨迹匹配 / 事实一致性 / rubric 加权分。

双 harness（custom / tau）共享。输入为 Task + state 的既有数据结构，输出数值分。
所有分数都在 [0, 1]；无法判定的场景返回 1.0（视为"未发现错误"，不惩罚）。
"""
from __future__ import annotations

import re
from typing import Any

DEFAULT_RUBRIC = {"subgoal_recall": 0.4, "tool_flow_match": 0.3, "fact_consistency": 0.3}
# answer_grounding 无默认权重：仅在任务 reward_rubric 显式指定时才参与加权（向后兼容）。
_RUBRIC_KEYS = ("subgoal_recall", "tool_flow_match", "fact_consistency", "answer_grounding")

# 检索（RAG）工具名集合：其"非空命中"结果可视为答案 ground 来源（改动二）。
# 与 tools/cs_tools.py、guide_tools.py 的 domain="rag" 工具对应；空命中/RAG 降级文本
# 视为无 ground。RAG OFF 时这些工具不被调用 → 无检索记录 → 天然 None。
RAG_TOOL_NAMES = {"query_policy", "query_cs_knowledge", "recommend_products", "verify_policy"}
# RAG 客户端降级/空命中标记（mcp_client / RAG-MCP query_knowledge_hub 返回）
_RAG_NO_HIT_MARKERS = ("[RAG]", "没有找到匹配 chunk", "知识库服务不可用", "调用")

# 失败原因常量（两 harness 共用，避免字面量漂移）
FAIL_REASONS = {
    "MAX_STEPS": "max_steps_exceeded",   # 步数超限 = 效率作为正确性的一部分
    "PROBE": "probe_fail",
    "SCRIPT": "script_exhausted",
    "ERROR": "error",
}

# 数值语境关键词（只有这些词附近的数字才作为"事实断言"）
VALUE_KEYWORDS = ["价格", "售价", "库存", "金额", "数量", "¥", "元", "件", "笔", "单",
                  "下单", "发货", "运费", "共", "库存有", "总价"]
# 常见序数/量词数字不作为事实证据（避免 "2笔" 因 "订单#2" 虚假接地）
ORDINAL_EXCLUDE = {"0", "1", "2", "3"}


def normalize_serialized(value: Any) -> str:
    """把工具参数值归一成字符串用于比对（int/str/float 统一，去空白）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # 5.0 → "5"，避免 "5" 与 "5.0" 误判
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, dict):
        return " ".join(f"{k}:{normalize_serialized(v)}" for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return " ".join(normalize_serialized(v) for v in value)
    return str(value).strip()


def subgoal_recall(task: Any, state: dict) -> float:
    """子目标召回：已执行工具是否覆盖黄金工具流（按工具名）。"""
    flow = task.golden_tool_flow
    if not flow:
        return 1.0
    executed = {r.name for r in state.get("tool_records", [])}
    expected = {s.tool for s in flow}
    if not expected:
        return 1.0
    return round(len(executed & expected) / len(expected), 3)


def tool_flow_match(task: Any, state: dict) -> float:
    """轨迹匹配：黄金工具流是否以相对顺序出现在已执行序列中。

    采用最长公共子序列（LCS）比值：允许中间穿插其它工具，但顺序必须保持。
    """
    flow = [s.tool for s in task.golden_tool_flow]
    if not flow:
        return 1.0
    executed = [r.name for r in state.get("tool_records", [])]

    # LCS（工具名序列）
    dp = [[0] * (len(flow) + 1) for _ in range(len(executed) + 1)]
    for i in range(1, len(executed) + 1):
        for j in range(1, len(flow) + 1):
            if executed[i - 1] == flow[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(executed)][len(flow)]
    return round(lcs / len(flow), 3)


async def functional_correct(probe: Any, task: Any) -> float:
    """功能正确性：expected_state 各字段达成比例（probe 逐字段判定）。"""
    if probe is None:
        return 1.0
    expected = task.expected_state
    if expected is None:
        return 1.0
    field_results = await probe.partial_check(expected)
    if not field_results:
        return 1.0
    return round(sum(1 for v in field_results.values() if v) / len(field_results), 3)


def _rag_hit_results(tool_results: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """筛出检索工具中"真命中"的结果（result 非空且非降级/空命中标记）。

    判定 ground 的依据是**检索命中>0 且 result 非空**，而非"调用过检索工具名"
    （R1：RAG OFF / 空命中 / RAG 降级均不算有 ground）。命中文本需含实际内容——
    用最低长度 + 排除降级/空命中标记来识别，避免把"[RAG] 服务不可用"或
    "没有找到匹配 chunk"当成可作依据的检索返回。
    """
    hits: list[tuple[str, str]] = []
    for name, res in tool_results:
        if name not in RAG_TOOL_NAMES:
            continue
        res = (res or "").strip()
        if not res:
            continue
        if any(mark in res for mark in _RAG_NO_HIT_MARKERS):
            continue
        # 真命中文本通常比空命中标记长；低于阈值的视为无实质内容
        if len(res) < 8:
            continue
        hits.append((name, res))
    return hits


def answer_grounding(task: Any, last_reply: str, tool_results: list[tuple[str, str]]) -> float | None:
    """答案级 grounding：回复对 key_facts 的覆盖比（有检索命中为 ground 前提）。

    - 无检索命中（RAG OFF / 空命中 / 降级）→ None（不参与加权，避免惩罚无 RAG 场景）。
    - 有命中但 task 无 key_facts → None（无标准可判）。
    - 有命中 + 有 key_facts → reply 中出现该要点的比例 ∈ [0,1]。
      要点匹配做宽松判定：key_fact 整体或去除空白后的子串出现在 reply 中即算覆盖；
      数字口径类事实（含数字）按归一后数字比对（复用 _extract_value_claims 思路）。
    """
    hits = _rag_hit_results(tool_results)
    if not hits:
        return None
    key_facts = list(getattr(task, "key_facts", []) or [])
    if not key_facts:
        return None
    # grounding 测的是**回复**是否覆盖要点；检索命中只用来确证 ground 存在。
    # 不把"命中文本里有该要点"算作回复覆盖（否则回复漏要点也会得高分）。
    covered = 0
    for kf in key_facts:
        if _fact_in_reply(kf, last_reply):
            covered += 1
    return round(covered / len(key_facts), 3)


# 数字方向词（P0-2 收紧数字兜底）：fact 表达"X 内/前"窗口，text 同数却在"X 后/外/超"
# 属关系错误（"48小时后"≠"48小时内"），判不覆盖。仅查数字**后**缀窗口，避免把
# "下单后48小时"里数字前的"后"（=after 下单，非窗口方向）误当方向词。
_NUM_BOUND_INSIDE = ("内", "前")
_NUM_BOUND_OUTSIDE = ("后", "外", "超")


def _direction_conflict(fact: str, text: str, num: str) -> bool:
    """num 为 fact 中匹配到的数字原文。fact 在数字后带"内/前"、text 同数字后带"后/外/超"
    且无"内/前" → 方向冲突（不覆盖）。后缀窗口取 4 字以覆盖"天内/小时后/个工作日外"。
    保守设计：fact 无数值窗口词 → 不冲突（如"支持支付宝"）。"""
    def post(s: str) -> str:
        s = "".join(s.split())
        i = s.find(num)
        return s[i + len(num): i + len(num) + 4] if i >= 0 else ""

    f_in = any(w in post(fact) for w in _NUM_BOUND_INSIDE)
    t = post(text)
    if not f_in or not t:
        return False
    return (any(w in t for w in _NUM_BOUND_OUTSIDE)
            and not any(w in t for w in _NUM_BOUND_INSIDE))


def _fact_in_reply(fact: str, text: str) -> bool:
    """宽松判定要点 fact 是否出现在 text：去空白后子串匹配；数字类按归一数字比对。

    数字兜底（P0-2 收紧）：fact 含数字时取出 ≥4 的数值断言，text 含同归一值即候选覆盖，
    但**方向冲突**（"48小时内" vs 回复"48小时后"）判不覆盖——数字对但关系错仍是口径错误。
    """
    if not fact or not text:
        return False
    compact_fact = "".join(fact.split())
    compact_text = "".join(text.split())
    if compact_fact in compact_text:
        return True
    # 兜底：fact 含数字时，取出其中的多位数断言，看 text 是否含同归一值
    nums = [n for n in re.findall(r"\d+(?:\.\d+)?", fact) if _norm_num(n) not in {"0", "1", "2", "3"}]
    if nums:
        text_nums = {_norm_num(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}
        if all(_norm_num(n) in text_nums for n in nums):
            # 数字都在 ≠ 关系对：同数错向（内→后/外）仍判不覆盖
            if not any(_direction_conflict(fact, text, n) for n in nums):
                return True
    return False


def fact_consistency(reply: str, tool_results: list[tuple[str, str]]) -> float | None:
    """事实一致性分：回复中的"数值断言"是否能在工具返回值中找到依据。

    只对"价格/库存/金额/数量/¥/元"等**数值语境**附近的数字做比对，
    排除 0/1/2/3 等常见序数量词（避免 "2笔订单" 因工具含 "订单#2" 虚假接地）。
    回复里可核验的数值断言都能在工具结果里找到 → 1.0；有找不到的 → 按比例扣分。
    无数值语境或无数值断言 → None（不参与 rubric 加权，避免惩罚无法核验的场景）。
    """
    if not tool_results:
        return None
    result_text = " ".join(res for _, res in tool_results)

    # 找回复中"数值语境"附近的数字断言（如 "价格是 100 元" → 100）
    claims = _extract_value_claims(reply)
    if not claims:
        return None

    # 工具结果里的完整数字（含小数），注意 \d{1,3}(?:,\d{3})* 会截断 3788 → 378，故用 \d+
    result_numbers = set(re.findall(r"(?<!\d)(\d+(?:\.\d+)?)", result_text))
    # 归一比对：3888 与 3888.0 视作同一数字（避免整数 claim 因工具带 .0 而未接地）
    result_normalized = {_norm_num(n) for n in result_numbers}
    grounded = [c for c in claims if _norm_num(c) in result_normalized]
    return round(len(grounded) / len(claims), 3)


def _norm_num(num: str) -> str:
    """数字归一：去千分位逗号与尾随 .0，3888.0 → 3888。"""
    n = num.replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
        if not n:
            n = "0"
    return n


def _extract_value_claims(reply: str) -> set[str]:
    """提取回复中的"数值断言"：数值语境关键词后面跟的数字。"""
    claims: set[str] = set()
    for kw in VALUE_KEYWORDS:
        for m in re.finditer(re.escape(kw) + r"\s*[:：是约为]?\s*([0-9][\d,.]*)", reply):
            num = m.group(1).replace(",", "")
            # 排除常见序数/量词
            if num.rstrip(".").rstrip("0").lstrip("0") in ORDINAL_EXCLUDE or num in ORDINAL_EXCLUDE:
                continue
            claims.add(num)
    # 兜底：整段回复里的独立数字（无语境时也收集，但仅当有语境关键词时返回）
    if not claims:
        for m in re.finditer(r"(?<![¥\d.])([1-9]\d{2,})(?:\.\d+)?", reply):
            claims.add(m.group(1))
    return claims


def param_accuracy(task: Any, state: dict) -> float:
    """错参数率（参数正确性）：黄金流里带参数期望的调用，参数是否匹配。

    只对 `ToolStep.params` 非空的步骤做判定：agent 的 tool_record 里该工具调用
    的 args 必须包含期望参数（宽松包含匹配，值转 str 归一后比对）。
    无可判定的参数期望 → 返回 1.0（不惩罚）。
    """
    flow = task.golden_tool_flow or []
    expected = [s for s in flow if s.params]
    if not expected:
        return 1.0

    executed: list[Any] = state.get("tool_records", [])
    matched = 0
    checkable = 0
    for step in expected:
        recs = [r for r in executed if r.name == step.tool]
        if not recs:
            continue
        checkable += 1
        # 该工具任意一次调用命中期望参数即算通过
        if any(_params_match(r.args, step.params) for r in recs):
            matched += 1
    if checkable == 0:
        return 1.0
    return round(matched / checkable, 3)


def _params_match(got: dict[str, Any], want: dict[str, Any]) -> bool:
    """参数匹配：期望的每个字段都出现在 agent 实参中（值宽松归一）。"""
    for k, want_v in want.items():
        if k not in got:
            return False
        got_v = got[k]
        # 列表：期望为子集
        if isinstance(want_v, list):
            if not isinstance(got_v, list):
                return False
            want_norm = {normalize_serialized(v) for v in want_v}
            got_norm = {normalize_serialized(v) for v in got_v}
            if not want_norm.issubset(got_norm):
                return False
        else:
            if normalize_serialized(got_v) != normalize_serialized(want_v):
                return False
    return True


def rubric_score(task: Any, subgoal: float | None = None,
                 flow: float | None = None, fact: float | None = None,
                 grounding: float | None = None) -> float:
    """rubric 加权分：消费 task.reward_rubric（缺省用 DEFAULT_RUBRIC）。

    只对**实际有值**的分项加权：None 的分项不参与（fact=None 时权重重分配到
    其余分项），避免"无法核验的事实"用不可靠权重拉低/拉高总分。
    answer_grounding：任务未显式给权重时（向后兼容）不参与加权；PQ 等带
    reward_rubric.answer_grounding 的任务会把答案 grounding 纳入总 rubric。
    """
    rubric = dict(task.reward_rubric) if getattr(task, "reward_rubric", None) else {}
    weights = dict(DEFAULT_RUBRIC)
    for k in _RUBRIC_KEYS:
        if k in rubric:
            weights[k] = float(rubric[k])
    # answer_grounding 未在任务 rubric 出现 → 不参与（即使用户传值也忽略权重为 0）
    if "answer_grounding" not in rubric:
        weights.pop("answer_grounding", None)
    values = {
        "subgoal_recall": subgoal,
        "tool_flow_match": flow,
        "fact_consistency": fact,
        "answer_grounding": grounding,
    }
    # 只保留有值分项
    active = {k: v for k, v in values.items() if v is not None and k in weights}
    if not active:
        return 1.0
    active_w = sum(weights[k] for k in active)
    if active_w <= 0:
        return 1.0
    score = sum(active[k] * weights[k] for k in active) / active_w
    return round(score, 3)


async def compute_success_signal(probe: Any, task: Any, state: dict,
                                 last_reply: str, outcome: str) -> dict[str, float]:
    """统一构造 success_signal（两 harness 共用）。

    分项：pass_at_1 / functional_correct / subgoal_recall / tool_flow_match /
    fact_consistency / answer_grounding / rubric_score / param_accuracy。
    """
    tool_results = [(r.name, r.result) for r in state.get("tool_records", [])]
    fact = fact_consistency(last_reply, tool_results)
    grounding = answer_grounding(task, last_reply, tool_results)
    fc = await functional_correct(probe, task)
    sg = subgoal_recall(task, state)
    tf = tool_flow_match(task, state)
    pa = param_accuracy(task, state)
    return {
        "pass_at_1": 1.0 if outcome == "success" else 0.0,
        "functional_correct": fc,
        "subgoal_recall": sg,
        "tool_flow_match": tf,
        "param_accuracy": pa,
        "fact_consistency": fact if fact is not None else 0.0,
        "answer_grounding": grounding if grounding is not None else None,
        "grounded": grounding is not None,
        "rubric_score": rubric_score(task, sg, tf, fact, grounding),
    }
