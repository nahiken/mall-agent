"""Agent 会话 API（前缀 /api/mall，兼容旧移动端 frontend 协议）。

端点：
- POST /chat/stream          SSE 流式对话（meta/text/products/done）
- GET  /conversations        会话列表（page=1 最新）
- GET  /conversations/{id}/messages  消息列表（page=1 最新、页内最旧在前）
- POST /conversations/hide   隐藏该用户全部会话
- GET  /health               健康检查
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..sessions import store as session_store
from .conversation import ConversationService
from .sse import sse_event, sse_comment


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]] = []
    user_id: str = "1"
    conversation_id: str | None = None
    context_policy: str | None = None  # full/sliding/summarize/hier/auto（缺省 auto）


def _last_user_text(req: ChatRequest) -> str:
    """取最后一条 user 消息内容作为本轮输入。"""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""


def build_agent_router(conv: ConversationService) -> APIRouter:
    router = APIRouter(prefix="/api/mall", tags=["agent"])

    @router.post("/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        user_text = _last_user_text(req)
        if not user_text:
            user_text = "你好"

        async def gen():
            # 心跳：避免 SSE 连接空闲被代理断开
            yield sse_comment("connected")
            async for evt in conv.run_turn(req.user_id, req.conversation_id, user_text,
                                           req.context_policy):
                yield evt

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.get("/conversations")
    async def conversations(user_id: str, page: int = 1, page_size: int = 10) -> dict:
        """会话列表。兼容旧前端协议：返回裸结构，items[0].id 为最新会话。"""
        data = session_store.list_sessions(user_id, page, page_size)
        for item in data.get("items", []):
            item["id"] = item["session_id"]
            item["message_count"] = item.get("message_count", 0)
        return data

    @router.get("/conversations/{conv_id}/messages")
    async def messages(conv_id: str, user_id: str, page: int = 1, page_size: int = 40) -> dict:
        """消息列表。兼容旧前端：items 数组，每条含 id/role/content/msg_type/created_at。"""
        return session_store.list_messages(conv_id, user_id, page, page_size)

    @router.post("/conversations/hide")
    async def hide(user_id: str) -> dict:
        session_store.hide_sessions(user_id)
        return {"code": 200, "message": "success", "data": None}

    @router.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "mall-agent"}

    return router
