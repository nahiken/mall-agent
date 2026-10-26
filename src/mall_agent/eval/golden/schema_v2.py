"""黄金数据集 v2 schema 扩展（在 v1 基础上增加"难分辨"特征）。

参考 τ-Bench 的设计精髓：
1. 复杂条件偏好（"如果没X就选Y"）→ 需要 Agent 从多个候选推理
2. DB 状态哈希判定（确定性、可复现）
3. 身份确认规则（先确认用户再操作）
4. confusable 任务对（表面相似但答案/路径不同）
5. 多子目标交织（一个请求里多个动作）
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .schema import ExpectedState, ToolStep


class ConditionalPreference(BaseModel):
    """条件偏好：当主选项不可用时如何降级。"""
    primary: str                 # 首选（如"机械键盘 clicky 轴"）
    fallback: str                # 降级条件
    fallback_choice: str         # 降级后的选择
    condition: str = "not_available"  # 触发条件


class ConfusableTag(BaseModel):
    """confusable 任务对：标识两个表面相似但答案不同的任务。"""
    pair_id: str                 # 同一对的 ID
    variant: Literal["a", "b"]   # 变体
    key_difference: str          # 两者的关键差异（用于评测时人工核对）


class GoldenTaskV2(BaseModel):
    """v2 复杂黄金任务（扩展 v1）。"""
    # ---- v1 基础字段 ----
    task_id: str
    domain: Literal["customer_service", "shopping_guide", "mixed"]
    task_type: Literal["question", "transaction", "multi_turn", "escalation", "conditional"]
    opener: str                  # 首条用户消息（v2 为复杂指令）
    user_profile: dict[str, Any] = Field(default_factory=dict)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    required_subgoals: list[str] = Field(default_factory=list)
    golden_tool_flow: list[ToolStep] = Field(default_factory=list)
    expected_state: ExpectedState = Field(default_factory=ExpectedState)
    constraints: list[str] = Field(default_factory=list)
    safety_tags: list[Literal["injection", "pii", "escalation", "jailbreak", "cross_user"]] = Field(default_factory=list)
    scripted_user_replies: list[str] = Field(default_factory=list)
    persona: str | None = None
    reward_rubric: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)

    # ---- v2 难分辨特征 ----
    # 1. 条件偏好（需要多候选推理）
    conditional_preferences: list[ConditionalPreference] = Field(default_factory=list)
    # 2. 期望最终 DB 状态（确定性判定，可算哈希）
    expected_db_state: dict[str, Any] = Field(default_factory=dict)
    # 3. confusable 任务对
    confusable: ConfusableTag | None = None
    # 4. 身份确认（模仿 τ-Bench rules：先确认身份再操作）
    identity_required: bool = False
    identity_info: dict[str, Any] = Field(default_factory=dict)  # {"name":..., "phone":...}
    # 5. 显式授权（敏感操作需用户明确说"是"）
    explicit_consent_required: bool = False
    # 6. 干扰项（用户提到的无关信息，Agent 不应被带偏）
    distractors: list[str] = Field(default_factory=list)
    # 7. 正确性陷阱（看似该用A工具，实际该用B）
    trap: str | None = None      # 描述陷阱

    # ---- v3 答案级评测字段（改动一；缺省向后兼容，旧数据可无）----
    reference_answer: str = ""                # 参考答案（用于答案级判答，rule+judge 对目标）
    key_facts: list[str] = Field(default_factory=list)   # 必须覆盖的要点（判答按覆盖比）
    rag_hub: str | None = None                # 知识域：cs / policy / guide（None=无需检索）
    category: str = ""                        # 数据集类别标签：PQ/APP/MH/NUM/SECURITY 等
    depends_on_private_kb: bool = False       # 答案是否依赖 KB 私有事实（分层分析用）


def v2_to_v1(task: GoldenTaskV2) -> "GoldenTask":
    """v2 转 v1（供现有评测管线复用）。"""
    from .schema import GoldenTask
    return GoldenTask(
        task_id=task.task_id, domain=task.domain, task_type=task.task_type,
        opener=task.opener, user_profile=task.user_profile,
        initial_state=task.initial_state, required_subgoals=task.required_subgoals,
        golden_tool_flow=task.golden_tool_flow, expected_state=task.expected_state,
        constraints=task.constraints, safety_tags=task.safety_tags,
        scripted_user_replies=task.scripted_user_replies, persona=task.persona,
        reward_rubric=task.reward_rubric, params=task.params,
    )
