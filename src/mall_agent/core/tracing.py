"""事件总线：统一收集 LLM 调用 / 工具调用 / 路由 / 权限 四类事件。

设计：monitor 是全局单例，各层通过 `tracing.emit(...)` 上报事件；
monitor 负责把事件落盘（JSONL）并在需要时聚合为指标。
这样 core/agents/tools 不直接依赖 monitor 实现，避免循环依赖。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class LLMEvent:
    caller: str                # "agent" | "user_sim" | "judge" | "rag"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    total_ms: float | None = None
    cost_rmb: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class ToolEvent:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    latency_ms: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class RouteEvent:
    kind: Literal["route", "transfer", "skill_load"]
    source: str
    target: str
    ts: float = field(default_factory=time.time)


@dataclass
class PermissionEventEvt:
    tool: str
    role: str
    decision: str
    reason: str = ""
    ts: float = field(default_factory=time.time)


class _Tracing:
    """轻量事件总线：emit 后交给已注册的 sink（monitor）。"""

    def __init__(self) -> None:
        self._sinks: list[Any] = []

    def register(self, sink: Any) -> None:
        if sink not in self._sinks:
            self._sinks.append(sink)

    def emit(self, event: Any) -> None:
        for sink in self._sinks:
            sink(event)


tracing = _Tracing()
