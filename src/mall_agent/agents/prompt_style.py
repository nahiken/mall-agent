"""提示词风格层（改动七 / T11）：输出压缩的**可开关风格轴**，并入实验 → 8 格。

`STYLES = {"standard", "concise"}`；当前格经由 env `PROMPT_STYLE` 或 state["prompt_style"]
解析（`resolve_prompt_style`，优先级 env > 显式 > 默认 standard）。style 只作用
**agent 主调用**的 system prompt（agent_node / worker_node 装配处读），不作用于
judge/user_sim/supervisor/router 的简短调用——它们各自固定 prompt。

`concise` 语义（改动七.1）：在保质量（answer_correct / 要点完整度）前提下压 out-token。
约束刻意写成"仍须完整给出关键数字口径、不得因简洁而漏要点或臆造"，避免把
压缩劣化误读为 RAG 失效（R9：concise 下"少检索/少步数"是预期副作用，单独记录）。
"""
from __future__ import annotations

import os

# 输出风格追加段（追加到 system prompt 末尾；standard = 空 = 零改动基线）
STYLE_SUFFIX: dict[str, str] = {
    "standard": "",
    "concise": (
        "\n回答风格要求【简洁】：只回答用户所问的要点，客服场景不解释内部过程、"
        "不堆砌客套与免责话术；能用一句话就不用两段。"
        "但关键数字口径（天数/时长/金额/条件）必须完整给出，不得因简洁而漏要点或臆造。"
    ),
}

# 合法风格（非法值回落 standard）
STYLES = tuple(STYLE_SUFFIX.keys())


def resolve_prompt_style(explicit: str | None = None) -> str:
    """解析生效风格：env PROMPT_STYLE > explicit 参数 > 默认 standard。"""
    for cand in (os.getenv("PROMPT_STYLE"), explicit):
        if cand and cand in STYLE_SUFFIX:
            return cand
    return "standard"


def style_system_suffix(style: str | None = None) -> str:
    """返回 style 的 system prompt 追加段（供 prompt 装配处拼到 base 后）。"""
    style = style if style in STYLE_SUFFIX else resolve_prompt_style(style)
    return STYLE_SUFFIX[style]
