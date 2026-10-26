"""CLI：跑单个方案的单次对话，或交互式体验。

用法：
    python -m mall_agent.cli.run_agent --scheme 1 "帮我查一下我的订单"
    python -m mall_agent.cli.run_agent --scheme 1            # 交互模式
    python -m mall_agent.cli.run_agent --scheme 1 --resume <id>   # 续聊历史会话
    python -m mall_agent.cli.run_agent --scheme 1 --user 42        # 绑定用户
    python -m mall_agent.cli.run_agent --mock-check          # 验证 mall mock 是否可跑
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _mock_check() -> None:
    from mall_agent.mall_api.client import get_mall_client

    client = get_mall_client()
    await client.ensure_session()
    from mall_agent.mall_api import product as product_api

    results = await product_api.search_product("手机")
    print(f"[mock-check] search '手机' -> {len(results)} 条")
    for p in results[:3]:
        print(f"  - id={p.get('id')} {p.get('name')} ¥{p.get('price')}")
    print("✅ mall mock 可用")


async def _run_once(scheme: str, text: str, policy: str, user_id: str = "1",
                    resume: str | None = None) -> None:
    # 延迟导入，避免未装 langgraph 时 CLI 崩溃
    from mall_agent.agents.graph_factory import build_agent
    from mall_agent.core.types import ChatMessage
    from mall_agent.monitoring.monitor import monitor
    from mall_agent.sessions import store as session_store

    # 断线恢复：--resume 加载历史会话
    history: list[ChatMessage] | None = None
    if resume:
        rec = session_store.load_session(resume)
        if rec:
            history = [ChatMessage(**m) for m in rec.get("messages", [])
                       if m.get("role") in ("user", "assistant")]
    agent = build_agent(scheme=scheme, context_policy=policy)
    monitor.begin_trial(harness="custom", agent_id=scheme, golden_task_id="manual", context_policy=policy)
    out = await agent.acomplete_episode(text, user_id=user_id, history=history)
    result = monitor.end_trial(steps=out.get("steps", 0))
    print(out.get("reply", ""))
    print("\n--- 评测摘要 ---")
    print(json.dumps({
        "steps": result["steps"],
        "tool_calls": result["cost"]["tool_calls"],
        "agent_in": result["cost"]["agent_in"],
        "agent_out": result["cost"]["agent_out"],
        "latency_total_ms": result["latency_ms"]["total"],
    }, ensure_ascii=False, indent=2))


async def _interactive(scheme: str, policy: str, user_id: str = "1",
                       resume: str | None = None) -> None:
    from mall_agent.agents.graph_factory import build_agent
    from mall_agent.core.types import ChatMessage
    from mall_agent.sessions import store as session_store

    agent = build_agent(scheme=scheme, context_policy=policy)

    # 会话恢复/新建
    if resume:
        rec = session_store.load_session(resume)
        if rec:
            session_id = resume
            print(f"[resume] 会话 {resume}（{rec.get('title', '')}）")
        else:
            session_id = session_store.create_session(user_id)["session_id"]
            print(f"[新建] 会话 {session_id}")
    else:
        session_id = session_store.create_session(user_id)["session_id"]
        print(f"[会话] {session_id}")

    # 重建历史（消息还原为 ChatMessage）
    def _history() -> list[ChatMessage]:
        rec = session_store.load_session(session_id)
        msgs = rec.get("messages", []) if rec else []
        return [ChatMessage(**m) for m in msgs if m.get("role") in ("user", "assistant")]

    print(f"🧪 方案 {scheme} 交互模式（policy={policy}），输入 exit 退出")
    while True:
        try:
            text = input("你> ").strip()
        except EOFError:
            break
        if not text or text in ("exit", "quit"):
            break
        out = await agent.acomplete_episode(text, user_id=user_id, history=_history())
        reply = out.get("reply", "")
        print(f"AI> {reply}")

        # 每轮后保存会话（messages 追加 user + assistant）
        rec = session_store.load_session(session_id) or {}
        msgs = rec.get("messages", [])
        msgs.append({"role": "user", "content": text, "id": f"{len(msgs)}", "ts": __import__("time").time()})
        if reply:
            msgs.append({"role": "assistant", "content": reply, "id": f"{len(msgs)}", "ts": __import__("time").time()})
        session_store.save_session(session_id, {
            "user_id": user_id, "title": rec.get("title") or text[:30], "messages": msgs,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="mall-agent CLI")
    parser.add_argument("text", nargs="?", help="一次性输入（缺省则交互模式）")
    parser.add_argument("--scheme", default=None, help="1=单Agent全工具 2=中心化多Agent 3=Skill懒加载")
    parser.add_argument("--policy", default=None, help="full/sliding/summarize/hier")
    parser.add_argument("--user", default="1", help="绑定用户 id")
    parser.add_argument("--resume", default=None, help="续聊历史会话 id（data/sessions/）")
    parser.add_argument("--mock-check", action="store_true", help="验证 mock 后端")
    args = parser.parse_args()

    if args.mock_check:
        asyncio.run(_mock_check())
        return

    from mall_agent.core.settings import settings

    scheme = args.scheme or settings.default_scheme
    policy = args.policy or settings.default_context_policy

    if args.text:
        asyncio.run(_run_once(scheme, args.text, policy, args.user, args.resume))
    else:
        asyncio.run(_interactive(scheme, policy, args.user, args.resume))


if __name__ == "__main__":
    sys.exit(main())
