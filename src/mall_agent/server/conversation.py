"""会话服务：把 MallAgent + sessions/store 编排成多轮 HTTP 会话。

职责：
- ensure_conversation：续用已存在会话（校验归属）或新建。
- run_turn：追加用户消息 → 跑 agent（含历史上下文）→ 产出 SSE 事件
  （meta/text/products/done + 周期心跳）→ 持久化完整 state 与可见消息。
  - 心跳：agent 回合跑在后台 task，事件经队列，15s 无产出时发 keepalive，
    防代理 idle-timeout 掐断长回合。
  - 断线部分落盘：客户端中途断开（CancelledError）→ 取消 agent task，
    但把已构造的 state（含本轮 user 消息）落盘，重连可续聊。

多轮实现：不走 acomplete_episode（它会重置 messages），而是把持久化的
state 直接传给 agent.app.ainvoke（经 MallAgent.acomplete_state），系统提示
由节点现场拼，历史 messages 只增不减——天然支持断线恢复。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from ..agents.graph_factory import MallAgent
from ..core.types import ChatMessage
from ..sessions import store as session_store
from .sse import sse_comment, sse_event


def _to_chat(msg: dict[str, Any]) -> ChatMessage:
    """存储的 dict 消息还原为 ChatMessage（兼容缺省字段）。"""
    return ChatMessage(**{k: v for k, v in msg.items() if k in (
        "role", "content", "name", "tool_call_id", "tool_calls")})


class ConversationService:
    def __init__(self, agent: MallAgent) -> None:
        self.agent = agent
        self.role = "cs"
        self.max_steps: int | None = None

    # ------------------------------------------------------------------
    # 会话定位
    # ------------------------------------------------------------------
    def ensure(self, user_id: str, conversation_id: str | None, first_text: str = "") -> dict:
        return session_store.ensure_conversation(user_id, conversation_id, first_text)

    # ------------------------------------------------------------------
    # 一次对话轮次（SSE 事件流 + 心跳 + 断线部分落盘）
    # ------------------------------------------------------------------
    async def run_turn(self, user_id: str, conversation_id: str | None,
                       user_text: str, context_policy: str | None = None) -> AsyncIterator[str]:
        """处理一条用户消息，yield SSE 事件文本（meta/text/products/done）。

        context_policy：按请求选策略（full/sliding/summarize/hier/auto）；
        缺省用会话已存策略，都没有则 auto（自动压缩）。
        """
        rec = self.ensure(user_id, conversation_id, user_text)
        conv_id = rec["session_id"]

        # 1. 构造 state：持久化 state + 新用户消息
        state = dict(rec.get("state") or {})
        if not state.get("messages"):
            policy = context_policy or "auto"
            state = self._initial_state(user_id, conv_id, policy)
        else:
            # 沿用会话策略；请求显式指定则覆盖
            if context_policy:
                state["context_policy"] = context_policy
            state.setdefault("context_policy", rec.get("context_policy") or "auto")
        # RAG 开关随 agent 构建态注入（旧持久化 state 缺字段时补齐，防屏蔽失效）
        state.setdefault("rag_enabled", getattr(self.agent, "rag_enabled", True))
        # 持久化的 messages 是 dict 列表（_to_dict 序列化）；agent 节点用 m.role 访问，
        # 必须还原为 ChatMessage（含 tool 消息的 name/tool_call_id/tool_calls）
        msgs = [_to_chat(m) if isinstance(m, dict) else m for m in state.get("messages", [])]
        state["messages"] = msgs + [_to_chat({"role": "user", "content": user_text})]

        # 2. 先发 meta（前端拿 conversation_id）
        yield sse_event("meta", {"conversation_id": conv_id})

        # 3. 跑 agent（含历史上下文）——放后台 task，事件进队列，支持心跳与断线
        events: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        state_snapshot: dict[str, Any] = dict(state)

        async def _run():
            products: list[dict] = []
            try:
                out = await self.agent.acomplete_state(state_snapshot, max_steps=self.max_steps)
                final_state = out.get("state") or {}
                reply = out.get("reply", "")
                if reply:
                    events.put_nowait(sse_event("text", {"content": reply}))
                try:
                    products = await self._collect_products(final_state.get("tool_records", []))
                    if products:
                        events.put_nowait(sse_event("products", {"products": products}))
                except Exception:
                    pass
                self._persist(conv_id, user_id, user_text, reply, products, final_state, rec)
                events.put_nowait(sse_event("done", None))
            except Exception as e:
                # agent 失败：错误事件 + 部分落盘（保留本轮 user 消息，可续聊）
                events.put_nowait(sse_event("error", {"message": f"agent 执行失败: {type(e).__name__}: {e}"}))
                events.put_nowait(sse_event("done", None))
                try:
                    self._persist_partial(conv_id, user_id, user_text, state_snapshot, rec)
                except Exception:
                    pass

        task = asyncio.create_task(_run())
        last_emit = time.time()
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(events.get(), timeout=15.0)
                    last_emit = time.time()
                    yield evt
                    if evt.startswith("event: done"):
                        break
                except asyncio.TimeoutError:
                    # 15s 无产出 → 心跳（防代理 idle-timeout）
                    yield sse_comment("keepalive")
        except asyncio.CancelledError:
            # 客户端中途断开：取消 agent task，部分落盘（重连可续聊）
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                self._persist_partial(conv_id, user_id, user_text, state_snapshot, rec)
            except Exception:
                pass
            raise
        finally:
            if not task.done():
                task.cancel()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _initial_state(self, user_id: str, conv_id: str, policy: str = "auto") -> dict:
        return {
            "session_id": conv_id, "role": self.role, "user_id": user_id,
            "messages": [], "effective_messages": [], "context_policy": policy,
            "tool_records": [], "steps": 0, "max_steps": self.max_steps or 15,
            "permission_events": [], "flags": {}, "done": False,
            # RAG 开关（阶段 C 吞吐对照）：随 agent 构建态，运行期 registry 过滤据此生效
            "rag_enabled": getattr(self.agent, "rag_enabled", True),
        }

    async def _collect_products(self, tool_records: list[Any]) -> list[dict]:
        """收集本轮展示的商品卡片（兼容旧前端 AgentProduct 结构）。"""
        from ..mall_api import product as product_api

        cards: list[dict] = []
        seen_ids: set[int] = set()
        for r in tool_records:
            if getattr(r, "name", None) != "search_product":
                continue
            keyword = (r.args or {}).get("keyword", "")
            try:
                rows = await product_api.search_product(keyword=keyword, page_size=8)
            except Exception:
                continue
            for p in rows:
                pid = p.get("id")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                cards.append({
                    "id": pid, "name": p.get("name", ""),
                    "price": float(p.get("price", 0) or 0),
                    "pic": p.get("pic") or p.get("pic_url") or "",
                    "subTitle": p.get("sub_title") or p.get("subTitle") or "",
                    "brandName": p.get("brandName") or p.get("brand_name") or "",
                    "url": f"/pages/product/product?id={pid}",
                })
        return cards

    def _persist(self, conv_id: str, user_id: str, user_text: str,
                 reply: str, products: list[dict], final_state: dict, rec: dict) -> None:
        """落盘：可见消息（user/assistant 文本 + products 分开存）+ 完整 state。"""
        visible = self._append_visible(rec.get("messages") or [], user_text)
        if reply:
            visible.append({"role": "assistant", "content": reply, "id": f"{len(visible)}",
                            "ts": time.time(), "products": products})
        self._save_visible(conv_id, user_id, user_text, visible, final_state, rec)

    def _persist_partial(self, conv_id: str, user_id: str, user_text: str,
                         state: dict, rec: dict) -> None:
        """断线部分落盘：agent 未完成时，把「含本轮 user 消息的 state」存下。

        让重连后能带着上下文续聊（用户消息不丢，assistant 回复缺失可重试）。
        """
        visible = self._append_visible(rec.get("messages") or [], user_text)
        self._save_visible(conv_id, user_id, user_text, visible, state, rec)

    @staticmethod
    def _append_visible(visible: list[dict], user_text: str) -> list[dict]:
        """追加本轮 user 消息（去重 meta 阶段已追加的）。"""
        now = time.time()
        visible = list(visible)
        if visible and isinstance(visible[-1], dict) and visible[-1].get("role") == "user" \
                and visible[-1].get("content") == user_text:
            visible = visible[:-1]
        visible.append({"role": "user", "content": user_text, "id": f"{len(visible)}", "ts": now})
        return visible

    @staticmethod
    def _save_visible(conv_id: str, user_id: str, user_text: str,
                      visible: list[dict], state: dict, rec: dict) -> None:
        session_store.save_session(conv_id, {
            "user_id": user_id,
            "title": rec.get("title") or user_text[:30],
            "context_policy": state.get("context_policy") or rec.get("context_policy") or "auto",
            "messages": visible,
            "state": session_store._to_dict(state),
        })


def build_conversation_service(agent: MallAgent) -> ConversationService:
    return ConversationService(agent)
