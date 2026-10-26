"""通用节点：tool_executor 把「工具调用 + 权限守卫 + 会话」绑定在一起。

这是所有方案的公共组件：
- guard 做三层校验（工具级/数据级/敏感级）
- 放行则执行工具，拦截则返回 block 结果（也作为 ToolRecord 记录）
- 每个事件上报 tracing（ToolEvent / PermissionEventEvt）
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..core.tracing import PermissionEventEvt, ToolEvent, tracing
from ..core.types import AgentState, ChatMessage, ToolCall, ToolRecord
from ..permission.model import PermissionGuard, Session, ToolScope


class ToolExecutor:
    """执行器：根据 registry + guard + session 执行工具调用。"""

    def __init__(self, registry: Any, guard: PermissionGuard, session: Session):
        self.registry = registry
        self.guard = guard
        self.session = session

    def _to_openai_schema(self, spec) -> dict:
        return spec.to_openai_schema()

    async def execute_tool_call(self, call: ToolCall, state: AgentState) -> str:
        """执行一次工具调用，返回工具结果字符串（含权限拦截结果）。

        会话（role/user_id）优先取自当轮 state——worker 子代理与真实 user_id
        会随 state 变化；仅当 state 未携带时回退到构图时注入的固定 session。
        """
        role = state.get("role", self.session.role)
        # 方案2/3 的 worker 会把内部角色名（cs_worker/guide_worker）写进 state，
        # 权限层只认基础角色（cs/guide/supervisor/admin）。不做归一的话
        # guard.tool_permission_ok 会因 "guide_worker" 不在 spec.roles 里而拦掉
        # 该 worker 的全部工具调用——表现为"多 Agent 一用工具就被拦截"。
        role = {"cs_worker": "cs", "guide_worker": "guide"}.get(role, role)
        user_id = state.get("user_id", self.session.user_id)
        # 同意必须从**当轮用户消息**推导：executor 的 self.session 是构图时的固定值
        # （consent 恒为 implicit），照搬它会让 sensitive 工具永远被 block_sensitive。
        consent = "implicit"
        for m in reversed(state.get("messages") or []):
            role_of = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            if role_of != "user":
                continue
            text = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
            text = str(text or "")
            if any(c in text for c in ("确认", "确定", "是的", "对的", "同意", "帮我取消",
                                       "帮我退", "就这个", "没问题", "好")):
                consent = "explicit"
            break
        session = Session(role=role, user_id=user_id, consent=consent)

        spec = self.registry.get(call.name)
        if spec is None:
            msg = f"未知工具: {call.name}"
            tracing.emit(ToolEvent(name=call.name, args=call.arguments, ok=False))
            return msg

        t0 = time.perf_counter()
        allowed, pevt = self.guard.check(spec.scope, call.arguments, session, tool_name=call.name)

        record = ToolRecord(
            name=call.name, args=call.arguments,
            permission=pevt, step=state.get("steps", 0),
        )

        if not allowed:
            msg = f"[权限拦截] 无权调用 {call.name}：{pevt.reason if pevt else ''}"
            record.ok = False
            record.result = msg
            tracing.emit(PermissionEventEvt(tool=call.name, role=session.role,
                                            decision=pevt.decision if pevt else "block_tool",
                                            reason=pevt.reason if pevt else ""))
            record.latency_ms = (time.perf_counter() - t0) * 1000
            state.setdefault("permission_events", []).append(pevt)
            state.setdefault("tool_records", []).append(record)
            tracing.emit(ToolEvent(name=call.name, args=call.arguments, ok=False,
                                   latency_ms=record.latency_ms))
            return msg

        try:
            result = await spec.invoke(**call.arguments)
        except Exception as e:
            result = f"工具 {call.name} 执行失败: {e}"
            record.ok = False

        # 幻觉防护：检索类工具返回空结果时注入"不得编造"约束
        from ..eval.harness.hallucination_guard import empty_result_guard
        result, hit_empty = empty_result_guard(call.name, result)
        if hit_empty:
            state.setdefault("flags", {})["empty_result_guard"] = True

        record.result = result
        record.latency_ms = (time.perf_counter() - t0) * 1000
        state.setdefault("tool_records", []).append(record)
        tracing.emit(ToolEvent(name=call.name, args=call.arguments, ok=record.ok,
                               latency_ms=record.latency_ms))
        return result

    async def execute_tool_calls(self, tool_calls: list[dict], state: AgentState) -> list[ChatMessage]:
        """执行一组 tool_calls，返回 tool 消息列表（LangGraph/OpenAI 格式）。"""
        msgs: list[ChatMessage] = []
        for tc in tool_calls:
            call = ToolCall(name=tc.get("name", ""), arguments=tc.get("arguments", {}), call_id=tc.get("id", ""))
            result = await self.execute_tool_call(call, state)
            msgs.append(ChatMessage(
                role="tool",
                content=result,
                name=call.name,
                tool_call_id=call.call_id,
            ))
        return msgs
