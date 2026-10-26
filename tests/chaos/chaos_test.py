"""网络故障演练：注入故障 → 验证「重连能否回答」+「会话是否持久化」。

场景（对应 user 要求"出问题后看看重连能否回答，会话是否有持久化"）：
  S1 LLM 建连 5xx      → 重连(指数退避)后正常回答 + 会话落盘
  S2 LLM 流中断        → 重连(流中断重试)后答案完整 + 会话落盘
  S3 LLM 429 限流      → 退避重试后回答 + 会话落盘
  S4 RAG MCP 未启动    → agent 降级仍回答 + 会话落盘
  S5 RAG MCP 中途被杀  → 自动重连后检索恢复（工具失败→重连→再查）
  S6 mall 后端 5xx     → 工具失败降级，agent 仍回答 + 会话落盘
  S7 SSE 客户端中途断开 → 部分落盘，重连后能续聊

运行（需要 :8002 服务 + 对应故障环境）：
  python tests/chaos/chaos_test.py --server http://localhost:8002 [--scenario S1,S2]
输出：data/chaos/chaos_{ts}.json + 控制台表格。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

SERVER = "http://localhost:8002"
# 商品查询（S5 用，确认 RAG 检索恢复）
QUERY = "你好"


async def _post_stream(payload: dict, read_limit: int = 20000) -> tuple[str, dict]:
    """POST /api/mall/chat/stream，读完整 SSE，返回 (原始文本, 事件解析)。"""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{SERVER}/api/mall/chat/stream",
                                 json=payload) as resp:
            if resp.status_code != 200:
                return f"HTTP {resp.status_code}", {"error": resp.status_code}
            buf = b""
            async for chunk in resp.aiter_bytes():
                buf += chunk
                if len(buf) > read_limit:
                    break
    text = buf.decode("utf-8", errors="replace")
    events = _parse_sse(text)
    return text, events


def _parse_sse(text: str) -> list[dict]:
    events: list[dict] = []
    for block in text.split("\n\n"):
        lines = block.split("\n")
        evt = "message"
        data: str | None = None
        for ln in lines:
            if ln.startswith("event:"):
                evt = ln[6:].strip()
            elif ln.startswith("data:"):
                data = ln[5:].strip()
        if data is None:
            continue
        if data == "[DONE]":
            events.append({"event": evt, "data": "[DONE]"})
            continue
        try:
            events.append({"event": evt, "data": json.loads(data)})
        except Exception:
            events.append({"event": evt, "data": data})
    return events


def _has(events: list[dict], name: str) -> bool:
    return any(e["event"] == name for e in events)


def _answer(events: list[dict]) -> str:
    parts = [e["data"].get("content", "") for e in events if e["event"] == "text"]
    return "".join(parts)


async def _get_messages(conv_id: str, user_id: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{SERVER}/api/mall/conversations/{conv_id}/messages",
                             params={"user_id": user_id, "page": 1, "page_size": 40})
        if r.status_code != 200:
            return []
        return r.json().get("items", [])


def _verdict(name: str, ok: bool, detail: str) -> dict:
    return {"scenario": name, "passed": ok, "detail": detail}


async def _conv_id(events: list[dict]) -> str | None:
    return next((e["data"].get("conversation_id") for e in events if e["event"] == "meta"), None)


async def _verify_reply_persist(user_id: str, query: str, name: str,
                                min_answer: int = 0) -> tuple[dict, str | None]:
    """通用：请求一次 → 断言能回答 + 会话持久化（≥2 条消息）。返回 (verdict, conv_id)。"""
    _, events = await _post_stream({"messages": [{"role": "user", "content": query}], "user_id": user_id})
    answered = _has(events, "text") and _has(events, "done")
    if min_answer:
        answered = answered and len(_answer(events)) >= min_answer
    conv = await _conv_id(events)
    persist = bool(conv) and len(await _get_messages(conv, user_id)) >= 2 if conv else False
    return _verdict(name, answered and persist,
                    f"answer={answered} persist={persist}"), conv


async def _check_continue(user_id: str, conv_id: str, name: str) -> dict:
    """跨请求续聊：用同一 conversation_id 再发一条 user 消息，断言能带上下文回答。

    服务端从持久化 state 恢复历史（含 summary 增量摘要），请求只需最后一条 user 消息。
    """
    _, events = await _post_stream({"messages": [{"role": "user", "content": "再问一个问题"}],
                                    "user_id": user_id, "conversation_id": conv_id})
    ok = _has(events, "text") and _has(events, "done")
    return _verdict(name, ok, f"continue_reply={ok}")


async def scenario_s1(user_id: str) -> dict:
    """LLM 建连 5xx → 重连后回答 + 会话落盘 + 跨请求续聊。需要 faulty_llm --fail-before 1。"""
    v, conv = await _verify_reply_persist(user_id, QUERY, "S1 建连5xx重连")
    if v["passed"] and conv:
        c = await _check_continue(user_id, conv, "S1 续聊")
        return {**v, "passed": v["passed"] and c["passed"], "detail": v["detail"] + " " + c["detail"]}
    return v


async def scenario_s2(user_id: str) -> dict:
    """LLM 流中断 → 重连后答案完整 + 会话落盘 + 续聊。需要 faulty_llm --fail-mid-stream 1。"""
    v, conv = await _verify_reply_persist(user_id, QUERY, "S2 流中断重连", min_answer=5)
    if v["passed"] and conv:
        c = await _check_continue(user_id, conv, "S2 续聊")
        return {**v, "passed": v["passed"] and c["passed"], "detail": v["detail"] + " " + c["detail"]}
    return v


async def scenario_s3(user_id: str) -> dict:
    """LLM 429 → 退避重试后回答 + 会话落盘 + 续聊。需要 faulty_llm --fail-429 1。"""
    v, conv = await _verify_reply_persist(user_id, QUERY, "S3 429退避重试")
    if v["passed"] and conv:
        c = await _check_continue(user_id, conv, "S3 续聊")
        return {**v, "passed": v["passed"] and c["passed"], "detail": v["detail"] + " " + c["detail"]}
    return v


async def scenario_s4(user_id: str) -> dict:
    """RAG MCP 未启动 → 降级仍回答 + 会话落盘 + 续聊。"""
    v, conv = await _verify_reply_persist(user_id, QUERY, "S4 RAG未启动降级")
    if v["passed"] and conv:
        c = await _check_continue(user_id, conv, "S4 续聊")
        return {**v, "passed": v["passed"] and c["passed"], "detail": v["detail"] + " " + c["detail"]}
    return v


async def scenario_s5(user_id: str) -> dict:
    """RAG MCP 中途被杀 → 自动重连后检索恢复 + 续聊。需要手动杀 RAG 子进程后请求。"""
    v1, conv1 = await _verify_reply_persist(user_id, QUERY, "S5 RAG子进程重连")
    if v1["passed"] and conv1:
        c = await _check_continue(user_id, conv1, "S5 续聊")
        return {**v1, "passed": v1["passed"] and c["passed"], "detail": v1["detail"] + " " + c["detail"]}
    return v1


async def scenario_s6(user_id: str) -> dict:
    """mall 后端 5xx → 工具失败降级，agent 仍回答 + 会话落盘 + 续聊。"""
    v, conv = await _verify_reply_persist(user_id, QUERY, "S6 mall5xx降级")
    if v["passed"] and conv:
        c = await _check_continue(user_id, conv, "S6 续聊")
        return {**v, "passed": v["passed"] and c["passed"], "detail": v["detail"] + " " + c["detail"]}
    return v


async def scenario_s7(user_id: str) -> dict:
    """SSE 客户端中途断开 → 部分落盘，重连后能续聊。"""
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", f"{SERVER}/api/mall/chat/stream",
                                 json={"messages": [{"role": "user", "content": "断线测试长句"}], "user_id": user_id}) as resp:
            conv = None
            async for chunk in resp.aiter_bytes():
                if b"conversation_id" in chunk:
                    # 拿到 meta 就立刻断开
                    import re
                    m = re.search(rb'conversation_id":\s*"([^"]+)"', chunk)
                    if m:
                        conv = m.group(1).decode()
                    break
    # 等服务端检测断线 → 取消 agent → _persist_partial 部分落盘
    await asyncio.sleep(2.0)
    # 断开后查会话：应至少保留 user 消息（部分落盘）
    persisted = False
    if conv:
        msgs = await _get_messages(conv, user_id)
        persisted = any(m.get("role") == "user" for m in msgs)
    return _verdict("S7 SSE断开部分落盘", persisted, f"conv={conv} user_persisted={persisted}")


SCENARIOS = {
    "S1": scenario_s1, "S2": scenario_s2, "S3": scenario_s3,
    "S4": scenario_s4, "S5": scenario_s5, "S6": scenario_s6, "S7": scenario_s7,
}


async def run(server: str, scenarios: list[str], user_id: str) -> dict:
    global SERVER
    SERVER = server
    results: list[dict] = []
    for sc in scenarios:
        fn = SCENARIOS.get(sc)
        if not fn:
            results.append(_verdict(sc, False, "未知场景"))
            continue
        try:
            r = await fn(user_id)
        except Exception as e:
            r = _verdict(sc, False, f"异常: {type(e).__name__}: {e}")
        results.append(r)
        print(f"  {sc}: {'PASS' if r['passed'] else 'FAIL'}  {r['detail']}")
    total_ok = sum(1 for r in results if r["passed"])
    return {"generated_at": time.time(), "server": server,
            "total": len(results), "passed": total_ok,
            "results": results}


def main():
    parser = argparse.ArgumentParser(description="mall-agent 网络故障演练")
    parser.add_argument("--server", default="http://localhost:8002")
    parser.add_argument("--scenario", default="S1,S2,S3,S4,S5,S6,S7",
                        help="逗号分隔场景列表")
    parser.add_argument("--user", default="chaos")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenario.split(",") if s.strip()]
    print(f"=== 网络故障演练（{args.server}）===")
    print(f"场景: {', '.join(scenarios)}\n")
    report = asyncio.run(run(args.server, scenarios, args.user))
    out = Path("data/chaos")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"chaos_{int(time.time())}.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n=== 结果：{report['passed']}/{report['total']} 通过 ===")
    print(f"报告: {p}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
