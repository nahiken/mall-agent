"""故障注入 mock LLM server（OpenAI 兼容 /v1/chat/completions）。

用法（在另一终端起，然后让 mall-agent 的 LLM_BASE_URL 指向它）：
    python tests/chaos/faulty_llm.py --port 8010 --fail-before 1 --fail-mid-stream 2

故障参数（可组合）：
  --fail-before N       前 N 个请求直接抛 5xx（测试建连期重试）
  --fail-429 N          前 N 个请求返回 429（测试退避重试）
  --fail-mid-stream N   流式输出 N 个 chunk 后强制断开连接（测试流中断重试）
  --delay S             每个请求首 chunk 前延迟 S 秒（测试长静默/心跳）
  --always-fail         全部请求失败（5xx）
  --port PORT           监听端口（默认 8010）

故障计数按 `(fail-*)` 消耗，之后正常返回一段固定文本（含 tool_calls 可选）。
"""
from __future__ import annotations

import argparse
import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

# 全局故障配置（由 CLI 注入）
CFG = {
    "fail_before": 0, "fail_429": 0, "fail_mid": 0,
    "delay": 0.0, "always_fail": False,
}
# 已消耗计数
consumed = {"before": 0, "429": 0, "mid": 0}


def _text_response(content: str = "你好，这是故障注入 mock 的回复。", usage=True):
    """构造一个 OpenAI 兼容的流式 text chunk。"""
    chunk = {
        "id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0,
        "model": "faulty-mock", "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    if usage:
        chunk["usage"] = {"prompt_tokens": 10, "completion_tokens": len(content)}
    return chunk


def _done_chunk():
    return {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0,
            "model": "faulty-mock", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


@app.post("/v1/chat/completions")
async def chat_completions():
    # ---- 建连期故障 ----
    if CFG["always_fail"]:
        return _json_error(500, "injected 500 always")
    if consumed["before"] < CFG["fail_before"]:
        consumed["before"] += 1
        return _json_error(500, "injected 500 before stream")
    if consumed["429"] < CFG["fail_429"]:
        consumed["429"] += 1
        return _json_error(429, "injected rate limit")

    # 延迟（模拟慢响应/长静默）
    if CFG["delay"] > 0:
        await asyncio.sleep(CFG["delay"])

    # ---- 流式（可中途断开） ----
    parts = ["故障注入", "的", "流式", "回复", "文本", "，用于", "验证", "断流重试。"]

    async def gen():
        # 中途断流：发前 2 个 chunk 后抛异常 → uvicorn 中断连接（客户端收到
        # APIConnectionError/RemoteProtocolError，触发 llm.py 流中断重试）
        if CFG["fail_mid"] and consumed["mid"] < CFG["fail_mid"]:
            consumed["mid"] += 1
            for p in parts[:2]:
                yield f"data: {json.dumps(_text_response(p, usage=False), ensure_ascii=False)}\n\n"
            raise RuntimeError("injected mid-stream disconnect")
        # 正常流
        for p in parts:
            yield f"data: {json.dumps(_text_response(p, usage=False), ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(_done_chunk(), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _json_error(status: int, msg: str):
    from fastapi.responses import JSONResponse
    body = {"error": {"message": msg, "type": "injected", "code": status}}
    if status == 429:
        return JSONResponse(status_code=429, content=body)
    return JSONResponse(status_code=status, content=body)


def main():
    parser = argparse.ArgumentParser(description="faulty mock LLM")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--fail-before", type=int, default=0)
    parser.add_argument("--fail-429", type=int, default=0)
    parser.add_argument("--fail-mid-stream", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--always-fail", action="store_true")
    args = parser.parse_args()
    CFG.update({
        "fail_before": args.fail_before, "fail_429": args.fail_429,
        "fail_mid": args.fail_mid_stream, "delay": args.delay,
        "always_fail": args.always_fail,
    })
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
