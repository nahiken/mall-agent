"""黄金数据集 schema：GoldenTask。

评测双 harness（custom / tau）共享此 schema，golden_task_id 是跨 harness 的 join 键。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolStep(BaseModel):
    """期望工具调用序列（轨迹比对的基准）。"""
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    order: int = 0


class ExpectedState(BaseModel):
    """业务状态断言（extract/probe 用）。"""
    order_created: bool | None = None
    order_status: int | None = None      # 0..5
    cart_items: dict[str, int] | None = None   # {product_id: qty}
    address_id: int | None = None
    coupon_received: bool | None = None
    return_apply_id: int | None = None
    order_cancelled: bool | None = None
    # 精确断言（避免「任一订单被取消/任领一张券」这种弱判定）：
    order_cancelled_id: int | None = None   # 指定订单的 status 必须变成 4（已关闭）
    coupon_id: int | None = None            # 必须领到指定 id 的券


class GoldenTask(BaseModel):
    task_id: str                          # "CS-001"
    domain: Literal["customer_service", "shopping_guide", "mixed"]
    task_type: Literal["question", "transaction", "multi_turn", "escalation"]
    opener: str                           # 首条用户消息
    user_profile: dict[str, Any] = Field(default_factory=dict)
    initial_state: dict[str, Any] = Field(default_factory=dict)  # user_id 等
    required_subgoals: list[str] = Field(default_factory=list)
    golden_tool_flow: list[ToolStep] = Field(default_factory=list)
    expected_state: ExpectedState = Field(default_factory=ExpectedState)
    constraints: list[str] = Field(default_factory=list)   # "≤6 步"、"不得调用 member.list_address"
    safety_tags: list[Literal["injection", "pii", "escalation", "jailbreak", "cross_user"]] = Field(default_factory=list)
    scripted_user_replies: list[str] = Field(default_factory=list)
    persona: str | None = None
    reward_rubric: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
