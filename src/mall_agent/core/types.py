"""共享类型定义：消息、工具调用记录、Agent 状态、评测原始结果。

这些类型被 core / tools / permission / agents / monitoring / eval 共同使用，
必须保持单一事实源（不要在别处重复定义）。
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ============================================================
# 消息
# ============================================================

class ChatMessage(BaseModel):
    """一条对话消息（与 LangChain message 互转）。"""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None          # tool 消息的 tool name
    tool_call_id: str | None = None  # assistant 的 tool_call 绑定
    tool_calls: list[dict] | None = None  # assistant 消息携带的 tool_calls（OpenAI 格式）


# ============================================================
# 工具调用与权限
# ============================================================

class ToolCall(BaseModel):
    """LLM 发起的一次工具调用请求（guard 校验前）。"""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""


class PermissionEvent(BaseModel):
    """权限拦截事件（供 exp_permission 量化）。"""
    tool: str
    role: str
    decision: Literal["allow", "block_tool", "block_data", "block_sensitive"]
    reason: str = ""


class ToolRecord(BaseModel):
    """一次工具调用的完整记录（含权限结果与耗时）。"""
    step: int = 0
    name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    ok: bool = True
    permission: PermissionEvent | None = None
    latency_ms: float = 0.0


# ============================================================
# Agent 状态（LangGraph State schema）
# ============================================================

class AgentState(TypedDict, total=False):
    session_id: str
    role: str                        # "cs" | "guide" | "supervisor" | "cs_worker" | "guide_worker"
    user_id: str | None              # 会话绑定会员
    scheme: str                      # "1" | "2" | "3"
    messages: list[ChatMessage]      # 原始消息完整累积
    effective_messages: list[ChatMessage]  # 经 context policy 裁剪后真正喂 LLM 的输入（评测关键）
    context_policy: str              # full|sliding|summarize|hier
    tool_records: list[ToolRecord]   # 每次工具调用（含权限结果）
    steps: int
    max_steps: int
    pending_tool_calls: list[dict]   # LLM 发起的待执行工具调用
    active_skill: str | None         # 方案3：当前激活的 skill
    last_route: str | None           # 方案2：路由目标
    permission_events: list[PermissionEvent]
    flags: dict[str, Any]            # 发散/注入标记
    facts: dict[str, Any]            # 分层记忆的事实槽
    summary: dict[str, Any]          # AutoContext 增量摘要 {text,msg_count,token_estimate}（持久化续聊用）
    done: bool


# ============================================================
# 评测：统一原始结果（harness 只产这个，聚合在 metrics 层做）
# ============================================================

class CostBreakdown(BaseModel):
    agent_in: int = 0
    agent_out: int = 0
    user_sim_in: int = 0
    user_sim_out: int = 0
    judge_in: int = 0
    judge_out: int = 0
    tool_calls: int = 0
    external_cost_rmb: float = 0.0


class LatencyMs(BaseModel):
    ttft: float | None = None        # 首 token 时延
    tpot: float | None = None        # 每 token 时延
    total: float | None = None       # 总响应时间
    p95: float | None = None
    p99: float | None = None


class TrajectoryStep(BaseModel):
    step: int
    type: Literal["llm", "tool", "route", "permission", "transfer"]
    name: str
    ok: bool = True


class TrialResult(BaseModel):
    """单个任务单次运行的原始结果。双 harness（custom/tau）共享此 schema。"""
    trial_id: str
    harness: Literal["custom", "tau"]
    agent_id: str                    # "1" | "2" | "3"
    golden_task_id: str
    context_policy: str
    outcome: Literal["success", "fail", "blocked", "error"]
    success_signal: dict[str, float] = Field(default_factory=dict)  # pass_at_1/rubric_score/functional_correct
    steps: int = 0
    repeated_actions: int = 0
    detour_actions: int = 0
    router_calls: int = 0
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    latency_ms: LatencyMs = Field(default_factory=LatencyMs)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    permission: dict[str, int] = Field(default_factory=dict)  # blocked_attempts/total_attempts
    safety: dict[str, Any] = Field(default_factory=dict)      # injection_triggered/pii_leak/guard_ok + judge 附加字段
    meta: dict[str, Any] = Field(default_factory=dict)        # model/date/error
