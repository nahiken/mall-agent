"""幻觉防护模块：在工具返回空结果时注入约束，并检测编造信号。

两层设计：
1. **主动防护**（`empty_result_guard`）：检索类工具返回空/未找到时，
   给 LLM 注入系统级约束 —— 必须如实告知"未找到"，不得编造价格/库存/物流。
2. **被动检测**（`HallucinationDetector`）：对 agent 最终回复扫描编造信号，
   标记可疑输出（供评测/日志使用）。

接入点：ToolExecutor.execute_tool_call 在工具返回后调用 empty_result_guard。
词表统一从 metrics.detection 导入。
"""
from __future__ import annotations

import re

from ..metrics.detection import EMPTY_MARKERS, HALLUCINATION_MARKERS, HALLUCINATION_REGEX, NEGATION_WORDS

# 检索类工具（返回空即应如实告知）
SEARCH_LIKE_TOOLS = {
    "search_product", "search_order", "get_order_detail", "get_product_detail",
    "list_cart", "list_coupons", "list_coupons_by_product", "list_address",
    "query_policy", "query_cs_knowledge", "recommend_products", "check_stock",
}


def empty_result_guard(tool_name: str, result: str) -> str:
    """工具返回空结果时，注入系统级约束。

    返回：(注入后的结果, 是否命中空结果)
    """
    if tool_name not in SEARCH_LIKE_TOOLS:
        return result, False

    is_empty = any(m in result for m in EMPTY_MARKERS)
    if not is_empty:
        # 可能结果本身非空但很短，或工具返回了"[]"
        if result.strip() in ("", "[]", "{}", "null", "None"):
            is_empty = True

    if is_empty:
        # 给 LLM 的约束：附在工具结果后，强调不得编造
        guard = (
            "\n\n[系统约束] 注意：以上是工具检索的真实结果，未找到相关数据。"
            "你必须如实告知用户'未找到/没有相关信息'，不得编造任何价格、库存、"
            "订单号、发货时间或物流信息。"
        )
        return result + guard, True

    return result, False


class HallucinationDetector:
    """检测最终回复中的编造信号（在"未找到"语境下）。"""

    @staticmethod
    def check(reply: str, tool_results: list[tuple[str, str]]) -> dict:
        """扫描回复是否在空结果后仍编造了信息。"""
        # 找出哪些工具返回了空结果
        empty_tools = [name for name, res in tool_results
                       if any(m in res for m in EMPTY_MARKERS)]
        if not empty_tools:
            return {"hallucination": False, "empty_tools": [], "reason": ""}

        # 空结果语境下，回复却提到价格/库存/订单/发货 = 编造。
        # 但"无法提供价格/库存"这类否定表述不算编造——就近窗口检查否定词。
        hallucinated = []
        for marker in HALLUCINATION_MARKERS:
            for m in re.finditer(marker, reply):
                window = reply[max(0, m.start() - 40): m.end() + 40]
                if any(n in window for n in NEGATION_WORDS):
                    continue
                hallucinated.append(marker)
                break
        # 正则版信号（订单号 W\d{7}）——同样做否定窗口检查
        for m in HALLUCINATION_REGEX.finditer(reply):
            window = reply[max(0, m.start() - 40): m.end() + 40]
            if any(n in window for n in NEGATION_WORDS):
                continue
            hallucinated.append("订单号W码")
            break
        return {
            "hallucination": bool(hallucinated),
            "empty_tools": empty_tools,
            "reason": f"空结果工具 {empty_tools} 后回复含编造信号 {hallucinated}" if hallucinated else "",
        }
