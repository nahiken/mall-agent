"""LangGraph 三方案图工厂。

方案1 单Agent全工具：ReAct，绑全部工具。
方案2 中心化多Agent：supervisor → cs_worker / guide_worker。
方案3 Skill懒加载：router(意图) → cs_loop / guide_loop。

共享 AgentState（TypedDict）与通用节点（ctx_node / agent_node / tool_node / router_node / supervisor_node）。
消息用 OpenAI 兼容 dict 格式（{"role","content","name","tool_call_id"}），与核心 LLMClient 直接对接。
"""
from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..core.llm import LLMClient, make_llm
from ..core.settings import settings
from ..core.types import AgentState
from ..context.manager import ContextManager
from ..permission.model import PermissionGuard, Session, build_default_guard
from ..permission.probe import MockDataProbe
from ..tools.registry import ToolRegistry, registry
from .nodes import ToolExecutor
from .skills.skill_base import build_skills


# ------------------------------------------------------------------
# 会话与工具装配
# ------------------------------------------------------------------
_CONSENT_CUES = ("确认", "确定", "是的", "对的", "可以", "同意", "帮我取消", "帮我退",
                 "就这个", "没问题", "嗯", "好")


def _user_consent(state: AgentState) -> str:
    """从最后一条用户消息判断是否"显式同意"。

    原实现把所有会话的 consent 固定成 implicit，于是 sensitive 工具（取消订单/确认收货/
    退货申请）在**任何**会话里都被 block_sensitive 拦死——"多轮确认"这条链路实际上
    从未跑通过。这里按用户最新一句话判定：出现确认类措辞才升级为 explicit。
    只影响当轮，不做跨轮记忆（跨轮记忆应放在会话状态里，属后续改造）。
    """
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if getattr(m, "role", "") == "user" or (isinstance(m, dict) and m.get("role") == "user"):
            text = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
            text = str(text or "")
            return "explicit" if any(c in text for c in _CONSENT_CUES) else "implicit"
    return "implicit"


def _make_session(state: AgentState) -> Session:
    return Session(role=state.get("role", "cs"), user_id=state.get("user_id"),
                   consent=_user_consent(state))


def _make_registry(state: AgentState, scheme: str, rag_enabled: bool = True) -> ToolRegistry:
    """按方案过滤工具集。

    rag_enabled=False 时剔除 domain="rag" 的检索工具（query_policy/query_cs_knowledge/
    verify_policy/recommend_products）→ RAG OFF 无外部知识注入（R1 定义）。
    """
    if scheme == "1":
        # 全工具 = 客服 + 导购 + admin? 不含 admin（那是越权测试专用）
        reg = registry.filter_by_roles({"cs", "guide", "supervisor"})
        if not rag_enabled:
            reg = reg.filter_by_excluding_domain({"rag"})
        return reg
    if scheme == "2":
        role = state.get("role", "cs")
        roles = {"cs"} if role in ("cs", "cs_worker", "supervisor") else {"guide"}
        if role == "supervisor":
            # supervisor 只需只读 + transfer
            return registry.filter_by_roles({"supervisor"})
        reg = registry.filter_by_roles(roles)
        if not rag_enabled:
            reg = reg.filter_by_excluding_domain({"rag"})
        return reg
    if scheme == "3":
        skill_name = state.get("active_skill")
        if not skill_name:
            return registry.filter_by_roles({"supervisor"})
        skills = build_skills()
        skill = skills.get(skill_name)
        if not skill:
            return registry.filter_by_roles({"supervisor"})
        reg = registry.filter_by_skill(skill_name, skill.tool_names)
        if not rag_enabled:
            reg = reg.filter_by_excluding_domain({"rag"})
        return reg
    return registry


def _build_system_prompt(state: AgentState, scheme: str) -> str:
    role = state.get("role", "cs")
    role_desc = {
        "cs": "你是商城智能客服。可以查询订单、处理退换、查看地址/优惠券、查询政策知识库。",
        "cs_worker": "你是商城智能客服子代理，只处理客服类任务（订单/退换/优惠券/售后政策）。",
        "guide": "你是商城导购。可以搜索商品、查看详情/库存、加入购物车、查询优惠券。",
        "guide_worker": "你是商城导购子代理，只处理导购类任务（商品/购物车/推荐）。",
        "supervisor": "你是任务调度中心。判断用户请求属于客服还是导购，用 transfer 工具分派。",
    }
    base = role_desc.get(role, "你是商城AI助手。")
    scheme_desc = {
        "1": "你可以使用全部工具（客服+导购）。",
        "2": "你会被 supervisor 调度，只使用自己领域的工具。",
        "3": "你根据当前激活的技能使用对应工具。",
    }
    rule = (
        "重要规则：当你需要查询数据（订单/商品/库存/优惠券/政策等）时，必须先调用对应的工具获取真实数据，"
        "再基于工具返回结果回答用户。不要凭空编造数据。工具返回后请继续完成回答。"
    )
    # 输出风格轴（改动七/T11）：state["prompt_style"] > env PROMPT_STYLE > standard。
    # 只作用 agent 主调用（本函数仅 agent_node/worker_node 装配 system 时调用）。
    from .prompt_style import style_system_suffix
    return f"{base}\n{scheme_desc.get(scheme, '')}\n{rule}{style_system_suffix(state.get('prompt_style'))}"


# ------------------------------------------------------------------
# 通用节点
# ------------------------------------------------------------------
def make_ctx_node(context_policy: str):
    """ctx 节点：按运行期策略裁剪/压缩消息 → effective_messages。

    context_policy 传 "auto"（或 None 哨兵）时，每步从 state["context_policy"]
    读取（HTTP 可按会话/请求选策略）；summarize/auto 需要 LLM，注入 make_llm。
    """
    async def ctx_node(state: AgentState) -> dict:
        policy = context_policy
        if policy is None or policy == "auto":
            policy = state.get("context_policy") or "auto"
        from ..core.llm import make_llm
        cm = ContextManager(policy, llm=make_llm("agent"))
        eff = await cm.build([m for m in state.get("messages", [])], state)
        out: dict[str, Any] = {"effective_messages": eff, "context_policy": policy}
        # AutoContext 会把增量摘要写进 state["summary"]；LangGraph 只合并返回值，
        # 原地 mutation 不保证持久，这里显式回传让摘要随 state 落盘/续聊。
        if "summary" in state:
            out["summary"] = state.get("summary")
        return out
    return ctx_node


def make_agent_node(llm: LLMClient):
    """LLM 节点：读 effective_messages，产出 assistant 消息（含可能的 tool_calls）。

    LLM 输出需要兼容 OpenAI tools 格式。用 acomplete_with_tools 拿到 (content, tool_calls)。
    """
    async def agent_node(state: AgentState) -> dict:
        eff = state.get("effective_messages") or state.get("messages", [])
        # 注入 system prompt
        sys_prompt = _build_system_prompt(state, state.get("scheme", "1"))
        msgs = [{"role": "system", "content": sys_prompt}]
        for m in eff:
            if hasattr(m, "role"):
                msgs.append(_chat_to_dict(m))
            else:
                msgs.append(m)
        tools = _tools_of(state)
        content, tool_calls = await llm.acomplete_with_tools(msgs, tools)
        assistant_data: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_data["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
                for tc in tool_calls
            ]
        assistant = _dict_to_chat(assistant_data)
        new_msgs = list(state.get("messages", [])) + [assistant]
        steps = state.get("steps", 0) + 1
        return {"messages": new_msgs, "steps": steps, "pending_tool_calls": tool_calls}
    return agent_node


def _tools_of(state: AgentState, scheme: str | None = None) -> list[dict]:
    rag_enabled = bool(state.get("rag_enabled", True)) if isinstance(state, dict) else True
    reg = _make_registry(state, scheme or state.get("scheme", "1"), rag_enabled=rag_enabled)
    return reg.openai_schemas()


def _chat_to_dict(m: Any) -> dict:
    d = {"role": m.role, "content": m.content}
    if m.name:
        d["name"] = m.name
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if getattr(m, "tool_calls", None):
        d["tool_calls"] = m.tool_calls
    return d


def _dict_to_chat(d: dict) -> Any:
    from ..core.types import ChatMessage
    return ChatMessage(
        role=d.get("role", "user"),
        content=d.get("content", ""),
        name=d.get("name"),
        tool_call_id=d.get("tool_call_id"),
        tool_calls=d.get("tool_calls"),
    )


def make_tool_node(executor: ToolExecutor, scheme: str):
    """工具节点：执行 pending_tool_calls。"""
    async def tool_node(state: AgentState) -> dict:
        pending = state.get("pending_tool_calls") or []
        if not pending:
            return {}
        from ..core.types import ToolCall
        new_msgs = list(state.get("messages", []))
        for tc in pending:
            call = ToolCall(name=tc["name"], arguments=tc["arguments"], call_id=tc["id"])
            result = await executor.execute_tool_call(call, state)
            new_msgs.append(_dict_to_chat({
                "role": "tool", "content": result, "name": tc["name"], "tool_call_id": tc["id"],
            }))
        return {"messages": new_msgs, "pending_tool_calls": []}
    return tool_node


def _has_tool_calls(state: AgentState) -> str:
    """条件路由：有待执行工具 → tools；否则 → end（含 max_steps 终止）。"""
    if state.get("steps", 0) >= state.get("max_steps", 15):
        return "end"
    if state.get("pending_tool_calls"):
        return "tools"
    return "end"


# ------------------------------------------------------------------
# 方案2 节点
# ------------------------------------------------------------------
TRANSFER_ARGS = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "enum": ["cs_worker", "guide_worker", "done"],
                   "description": "客服子代理 / 导购子代理 / 完成"}
    },
    "required": ["target"],
}
TRANSFER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "transfer",
        "description": "把任务分派给客服子代理或导购子代理，或标记完成(done)。",
        "parameters": TRANSFER_ARGS,
    },
}


def make_supervisor_node(llm: LLMClient):
    async def supervisor_node(state: AgentState) -> dict:
        # 收集历史消息（ChatMessage 对象 → dict）
        msgs: list[dict] = []
        for m in state.get("messages", []):
            if hasattr(m, "role"):
                msgs.append(_chat_to_dict(m))
            elif isinstance(m, dict):
                msgs.append(m)
        if not msgs:
            msgs = [{"role": "user", "content": "（开始对话）"}]
        # supervisor 只绑 transfer 工具
        content, tool_calls = await llm.acomplete_with_tools(msgs, [TRANSFER_SCHEMA])
        target = "done"
        if tool_calls:
            args = tool_calls[0]["arguments"]
            target = args.get("target", "done")
        # 记录 assistant 回复
        assistant = _dict_to_chat({"role": "assistant", "content": content or f"[调度] 分派给 {target}"})
        new_msgs = state.get("messages", []) + [assistant]
        return {"messages": new_msgs, "last_route": target, "steps": state.get("steps", 0) + 1}
    return supervisor_node


def make_worker_loop(role: str, executor: ToolExecutor, llm: LLMClient, scheme: str, context_policy: str, max_steps: int):
    """worker 内部循环：agent → tools（最多 max_steps 步），或转移回 supervisor。"""
    async def worker_node(state: AgentState) -> dict:
        # 更新 role 上下文
        sub_state = dict(state)
        sub_state["role"] = role
        sub_state["scheme"] = scheme
        from ..core.llm import make_llm
        policy = context_policy if context_policy and context_policy != "auto" else state.get("context_policy") or "auto"
        eff = await ContextManager(policy, llm=make_llm("agent")).build(
            [m for m in state.get("messages", [])], sub_state)
        sys_prompt = _build_system_prompt(sub_state, scheme)
        msgs = [{"role": "system", "content": sys_prompt}]
        msgs += [_chat_to_dict(m) if hasattr(m, "role") else m for m in eff]
        tools = _tools_of(sub_state, scheme)
        content, tool_calls = await llm.acomplete_with_tools(msgs, tools)
        new_msgs = list(state.get("messages", []))
        if tool_calls:
            assistant = _dict_to_chat({
                "role": "assistant", "content": content,
                "tool_calls": [{"id": tc["id"], "type": "function",
                                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
                               for tc in tool_calls],
            })
            new_msgs.append(assistant)
            from ..core.types import ToolCall
            for tc in tool_calls:
                result = await executor.execute_tool_call(
                    ToolCall(name=tc["name"], arguments=tc["arguments"], call_id=tc["id"]), sub_state)
                new_msgs.append(_dict_to_chat({
                    "role": "tool", "content": result, "name": tc["name"], "tool_call_id": tc["id"],
                }))
            # worker 完成后回 supervisor
            return {"messages": new_msgs, "steps": state.get("steps", 0) + 1, "last_route": None}
        # 无工具调用 → 给用户回复，回 supervisor 标记完成
        new_msgs.append(_dict_to_chat({"role": "assistant", "content": content}))
        return {"messages": new_msgs, "steps": state.get("steps", 0) + 1, "last_route": "done"}
    return worker_node


# ------------------------------------------------------------------
# 方案3 节点
# ------------------------------------------------------------------
ROUTER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "select_skill",
        "description": "判断用户请求属于哪个技能：cs(客服) / guide(导购) / none(其他)。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "enum": ["cs", "guide", "none"],
                          "description": "cs=客服类 / guide=导购类 / none=都不属于"}
            },
            "required": ["skill"],
        },
    },
}


def make_router_node(llm: LLMClient, mode: str = "rule"):
    """意图路由：rule(关键词) 或 llm(1次便宜调用)。

    若 worker 已完成（last_route=="done"）则标记 none 结束，避免死循环。
    """
    skills = build_skills()

    async def router_node(state: AgentState) -> dict:
        # worker 刚完成 → 结束本轮
        if state.get("last_route") == "done":
            return {"active_skill": "none", "last_route": "done"}
        text = ""
        for m in state.get("messages", []):
            if hasattr(m, "role") and m.role == "user":
                text += m.content or ""
        skill = "none"
        if mode == "rule":
            for name, sk in skills.items():
                if sk.matches_rule(text):
                    skill = name
                    break
        else:
            content, tool_calls = await llm.acomplete_with_tools(
                [{"role": "system", "content": "你是意图路由器。"},
                 {"role": "user", "content": text}], [ROUTER_SCHEMA])
            if tool_calls:
                skill = tool_calls[0]["arguments"].get("skill", "none")
        return {"active_skill": skill, "last_route": skill, "steps": state.get("steps", 0) + 1}
    return router_node


def _route_by_skill(state: AgentState) -> str:
    skill = state.get("active_skill")
    if skill in ("cs", "guide"):
        return skill
    return "none"


# ------------------------------------------------------------------
# 方案1 图
# ------------------------------------------------------------------
def build_scheme1(executor: ToolExecutor, llm: LLMClient, context_policy: str, max_steps: int):
    g = StateGraph(AgentState)
    g.add_node("ctx", make_ctx_node(context_policy))
    g.add_node("agent", make_agent_node(llm))
    g.add_node("tools", make_tool_node(executor, "1"))
    g.add_edge(START, "ctx")
    g.add_edge("ctx", "agent")
    g.add_conditional_edges("agent", _has_tool_calls, {"tools": "tools", "end": END})
    g.add_edge("tools", "ctx")
    return g


# ------------------------------------------------------------------
# 方案2 图
# ------------------------------------------------------------------
def build_scheme2(executor: ToolExecutor, llm: LLMClient, context_policy: str, max_steps: int):
    g = StateGraph(AgentState)
    g.add_node("supervisor", make_supervisor_node(llm))
    g.add_node("cs_worker", make_worker_loop("cs_worker", executor, llm, "2", context_policy, max_steps))
    g.add_node("guide_worker", make_worker_loop("guide_worker", executor, llm, "2", context_policy, max_steps))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", _route_by_route, {
        "cs_worker": "cs_worker", "guide_worker": "guide_worker", "done": END})
    g.add_edge("cs_worker", "supervisor")
    g.add_edge("guide_worker", "supervisor")
    return g


def _route_by_route(state: AgentState) -> str:
    return state.get("last_route") or "done"


# ------------------------------------------------------------------
# 方案3 图
# ------------------------------------------------------------------
def build_scheme3(executor: ToolExecutor, llm: LLMClient, context_policy: str, max_steps: int, router_mode: str = "rule"):
    router_llm = make_llm("agent") if router_mode == "llm" else None
    g = StateGraph(AgentState)
    g.add_node("router", make_router_node(router_llm, mode=router_mode))
    g.add_node("cs_loop", make_worker_loop("cs_worker", executor, llm, "3", context_policy, max_steps))
    g.add_node("guide_loop", make_worker_loop("guide_worker", executor, llm, "3", context_policy, max_steps))
    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route_by_skill, {"cs": "cs_loop", "guide": "guide_loop", "none": END})
    g.add_edge("cs_loop", "router")
    g.add_edge("guide_loop", "router")
    return g


# ------------------------------------------------------------------
# 组装
# ------------------------------------------------------------------
class MallAgent:
    """对编译后图的封装，提供 acomplete_episode 高层接口。"""

    def __init__(self, graph, scheme: str, rag_enabled: bool = True,
                 prompt_style: str | None = None):
        self.graph = graph
        self.scheme = scheme
        self.rag_enabled = rag_enabled
        # 输出风格轴（改动七/T11）：agent 主调用 system 追加段；default standard（零改动）
        from .prompt_style import resolve_prompt_style
        self.prompt_style = resolve_prompt_style(prompt_style)
        self.app = graph.compile()

    async def acomplete_episode(self, user_text: str, user_id: str | None = None,
                                role: str = "cs", max_steps: int | None = None,
                                history: list[ChatMessage] | None = None) -> dict:
        """执行一轮完整对话（单条用户输入 → 最终回复）。

        history：可选的多轮历史消息（累积的 ChatMessage 列表）。传入后
        本轮 init state 的 messages = history + [新 user 消息]，实现真正的
        多轮上下文（断线恢复：把上轮失败前完整 messages 传进来重试即可）。
        """
        max_steps = max_steps or settings.max_steps
        from ..core.types import ChatMessage
        messages = list(history or []) + [ChatMessage(role="user", content=user_text)]
        init: AgentState = {
            "session_id": "ep", "role": role, "user_id": user_id or "1",
            "messages": messages,
            "effective_messages": [], "context_policy": "full",
            "tool_records": [], "steps": 0, "max_steps": max_steps,
            "permission_events": [], "flags": {}, "done": False,
            "rag_enabled": self.rag_enabled,
            "prompt_style": self.prompt_style,
        }
        return await self.acomplete_state(init, max_steps=max_steps)

    async def acomplete_state(self, state: AgentState, max_steps: int | None = None) -> dict:
        """低层接口：直接跑编译图，返回 {reply, steps, state}。

        供 HTTP 服务层做多轮会话：把持久化的完整 state 传进来、追加新消息
        后调用，即可带历史上下文续聊（断线恢复的载体）。
        """
        max_steps = max_steps or settings.max_steps
        result = await self.app.ainvoke(
            state,
            config={"recursion_limit": max_steps * 2 + 10, "max_steps": max_steps},
        )
        # 提取最终 assistant 回复
        reply = self._extract_reply(result)
        return {"reply": reply, "steps": result.get("steps", 0), "state": result}

    @staticmethod
    def _extract_reply(state: AgentState) -> str:
        msgs = state.get("messages", [])
        for m in reversed(msgs):
            if isinstance(m, dict):
                if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
                    return m["content"]
                continue
            if getattr(m, "role", None) != "assistant" or not m.content:
                continue
            # 含 tool_calls 的中间助手消息不作为最终回复
            if getattr(m, "tool_calls", None):
                continue
            return m.content
        return ""


def _build_executor(state: AgentState, scheme: str, rag_enabled: bool = True) -> ToolExecutor:
    guard = build_default_guard()
    # 确保 mall mock 就绪（同步），并注入 probe
    try:
        from ..mall_api.client import get_mall_client
        client = get_mall_client()
        mock = client.ensure_mock_sync()
        if mock is not None:
            guard.data_probe = MockDataProbe(mock)
    except Exception:
        pass
    session = _make_session(state)
    reg = _make_registry(state, scheme, rag_enabled=rag_enabled)
    return ToolExecutor(reg, guard, session)


def build_agent(scheme: str | int, context_policy: str | None = None,
                llm: LLMClient | None = None, router_mode: str = "rule",
                rag_enabled: bool = True, prompt_style: str | None = None) -> MallAgent:
    """构建指定方案的 MallAgent。

    rag_enabled=False：屏蔽全部 RAG 检索工具（R1），agent 仅凭参数先验+客服常识作答。
    prompt_style（改动七/T11）："standard"|"concise"，输出风格轴；None→env PROMPT_STYLE→standard。
    """
    scheme = str(scheme)
    policy = context_policy or settings.default_context_policy
    llm = llm or make_llm("agent")
    executor = _build_executor({"role": "cs", "user_id": "1", "rag_enabled": rag_enabled}, scheme, rag_enabled)
    if scheme == "1":
        graph = build_scheme1(executor, llm, policy, settings.max_steps)
    elif scheme == "2":
        graph = build_scheme2(executor, llm, policy, settings.max_steps)
    elif scheme == "3":
        graph = build_scheme3(executor, llm, policy, settings.max_steps, router_mode)
    else:
        raise ValueError(f"未知方案: {scheme}")
    return MallAgent(graph, scheme, rag_enabled=rag_enabled, prompt_style=prompt_style)
