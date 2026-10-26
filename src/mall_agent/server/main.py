"""启动入口：python -m mall_agent.server

用法：
    python -m mall_agent.server                       # 默认 :8002
    python -m mall_agent.server --port 8002 --scheme 1 --policy full
    python -m mall_agent.server --mock-agent           # 压测用的瞬时 FakeAgent（不调 LLM）
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time


def _instant_agent_factory(scheme: str, context_policy: str, rag_enabled: bool = True):
    """压测 mock 模式：瞬时返回的 agent（不调 LLM，测服务层自身吞吐）。"""
    from ..core.types import ChatMessage

    class _Instant:
        scheme = "mock"
        rag_enabled = True  # 占位；实际值在构造后由实例覆盖
        async def acomplete_state(self, state, max_steps=None):
            msgs = state.get("messages", [])
            reply = "ok"
            state["messages"] = msgs + [ChatMessage(role="assistant", content=reply)]
            return {"reply": reply, "steps": 1, "state": state}

    agent = _Instant()
    agent.rag_enabled = rag_enabled  # 实例级覆盖（避开 class body 同名作用域陷阱）
    from ..server.conversation import ConversationService
    conv = ConversationService(agent)
    # 商品提取跳过（无 tool_records）
    conv._collect_products = _collect_none
    return agent, conv


async def _collect_none(tool_records):
    return []


def _build(scheme: str, policy: str, mock: bool, rag_enabled: bool = True):
    from .app import create_app
    if mock:
        return create_app(scheme=scheme, context_policy=policy, agent_factory=_instant_agent_factory,
                          rag_enabled=rag_enabled)
    return create_app(scheme=scheme, context_policy=policy, rag_enabled=rag_enabled)


def main() -> None:
    parser = argparse.ArgumentParser(description="mall-agent HTTP server")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--scheme", default="1")
    parser.add_argument("--policy", default="auto", help="full/sliding/summarize/hier/auto（默认 auto=自动压缩）")
    parser.add_argument("--mock-agent", action="store_true", help="用瞬时 FakeAgent（压测服务层吞吐）")
    parser.add_argument("--rag-on", dest="rag_enabled", action="store_true", default=True,
                        help="启用 RAG 检索工具（默认，RAG ON）")
    parser.add_argument("--rag-off", dest="rag_enabled", action="store_false",
                        help="屏蔽 RAG 检索工具（R1：agent 零外部知识，测 ON/OFF 吞吐对照）")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    import uvicorn
    from . import main as _self  # noqa 确保包初始化

    app = _build(args.scheme, args.policy, args.mock_agent, args.rag_enabled)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
