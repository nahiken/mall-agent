"""实时监控：tracing 总线第二个订阅者，事件级聚合 + 滚动窗口统计。

设计：
- `RealtimeStats` 注册到 tracing 总线（与 monitor 并存，零侵入）。
- 维护最近事件环形缓冲（供 SSE /api/monitor/stream 推流）。
- 维护 60s 滚动窗口统计（QPS/延迟/token/工具/权限/成本/活跃 trial），
  /api/monitor/snapshot 直接读；监控页轮询兜底。
- 事件按 ts 落窗；窗口滑过的事件丢弃（用单调时钟近似）。

contextvar 隔离问题：trial 状态在 monitor 的 contextvars 里，本模块读不到
进行中 trial——所以这里做**事件级**聚合（事件带 ts，天然跨 task 可聚合），
活跃 trial 数由"最近 30s 有事件的会话数"近似。
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

from ..core.tracing import tracing
from .monitor import monitor

# 滚动窗口（秒）与事件缓冲上限
WINDOW_SECS = 60.0
BUFFER_MAX = 5000
# 活跃 trial 判定窗口
ACTIVE_WINDOW_SECS = 30.0


class RealtimeStats:
    def __init__(self, window_secs: float = WINDOW_SECS) -> None:
        self.window_secs = window_secs
        self._buffer: deque[dict[str, Any]] = deque(maxlen=BUFFER_MAX)
        self._lock = threading.Lock()
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._subscribers = 0
        self._started = False
        self._start = time.time()
        tracing.register(self._on_event)

    # ------------------------------------------------------------------
    # 事件订阅（tracing sink）
    # ------------------------------------------------------------------
    def _on_event(self, event: Any) -> None:
        d = monitor._event_to_dict(event)
        with self._lock:
            self._buffer.append(d)
            if self._queue is not None:
                try:
                    self._queue.put_nowait(d)
                except asyncio.QueueFull:
                    pass

    # ------------------------------------------------------------------
    # SSE 流：每订阅者一个队列
    # ------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        if self._queue is None or self._subscribers == 0:
            self._queue = asyncio.Queue(maxsize=1000)
        self._subscribers += 1
        return self._queue

    def unsubscribe(self) -> None:
        self._subscribers = max(0, self._subscribers - 1)
        if self._subscribers == 0:
            self._queue = None

    # ------------------------------------------------------------------
    # 快照统计
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            events = [d for d in self._buffer if d.get("ts", 0) >= now - self.window_secs]
            active_events = [d for d in self._buffer if d.get("ts", 0) >= now - ACTIVE_WINDOW_SECS]

        n_llm = 0
        tokens_in = 0
        tokens_out = 0
        totals: list[float] = []
        ttfts: list[float] = []
        tpots: list[float] = []
        caller_tokens: dict[str, dict[str, int]] = {}
        tool_calls = 0
        tool_errors = 0
        permission_attempts = 0
        blocked = 0
        cost = 0.0

        for e in events:
            etype = e.get("type")
            if etype == "LLMEvent":
                n_llm += 1
                ti = int(e.get("input_tokens", 0) or 0)
                to = int(e.get("output_tokens", 0) or 0)
                tokens_in += ti
                tokens_out += to
                cost += float(e.get("cost_rmb", 0) or 0)
                caller = e.get("caller", "?")
                c = caller_tokens.setdefault(caller, {"in": 0, "out": 0})
                c["in"] += ti
                c["out"] += to
                for key, arr in (("total_ms", totals), ("ttft_ms", ttfts), ("tpot_ms", tpots)):
                    v = e.get(key)
                    if v is not None:
                        arr.append(float(v))
            elif etype == "ToolEvent":
                tool_calls += 1
                if not e.get("ok"):
                    tool_errors += 1
            elif etype == "PermissionEventEvt":
                permission_attempts += 1
                if e.get("decision") != "allow":
                    blocked += 1

        def pct(arr: list[float], p: float) -> float | None:
            if not arr:
                return None
            s = sorted(arr)
            return round(s[min(len(s) - 1, int(len(s) * p))], 1)

        # 活跃 trial：最近 ACTIVE 窗口内有事件的不同 trial（用 golden_task_id/agent 近似）
        active_trials = len({(e.get("caller"), id(e)) for e in active_events})

        return {
            "window_secs": self.window_secs,
            "ts": now,
            "uptime_secs": round(now - self._start, 1),
            "qps": round(len(events) / self.window_secs, 3),
            "llm_calls": n_llm,
            "tokens": {"in": tokens_in, "out": tokens_out},
            "tokens_by_caller": caller_tokens,
            "latency": {
                "p50_total_ms": pct(totals, 0.5), "p95_total_ms": pct(totals, 0.95),
                "p95_ttft_ms": pct(ttfts, 0.95), "avg_tpot_ms": pct(tpots, 0.5),
                "sample_llm": len(totals),
            },
            "tools": {"calls": tool_calls, "errors": tool_errors},
            "permission": {"attempts": permission_attempts, "blocked": blocked},
            "cost_rmb": round(cost, 6),
            "active_trials": active_trials,
            "event_buffer": len(events),
        }


# 全局单例
realtime = RealtimeStats()
