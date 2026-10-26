"""SSE 事件格式化：兼容旧移动端 frontend（stores/agent.ts）的解析协议。

事件格式（每事件两行 + 空行分隔）：
    event: <name>
    data: <json>            （done 事件 data 为裸字符串 [DONE]）

事件名：meta / text / products / error / done。前端只消费这 4-5 种。
"""
from __future__ import annotations

import json
from typing import Any

# 结束哨兵（前端解析层先跳过 data: [DONE]）
DONE = "[DONE]"


def sse_event(name: str, data: Any) -> str:
    """构造一条 SSE 事件文本。data 为 JSON 序列化；done 用裸 [DONE]。"""
    if name == "done":
        payload: Any = DONE
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"event: {name}\ndata: {payload}\n\n"


def sse_comment(text: str) -> str:
    """SSE 心跳注释行（防代理断开）。"""
    return f": {text}\n\n"
