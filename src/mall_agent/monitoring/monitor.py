"""LLMInteractionMonitor：订阅 tracing 事件，落盘 JSONL 并暴露累计统计。

这是整个评测的数据采集核心（对应实战书中"LLMInteractionMonitor 可跟踪 token 用量"）。
设计：
- 单例，构造时注册到 tracing 总线。
- 每个 trial 调用 `begin_trial(...)` 开启一个会话，`end_trial()` 返回 TrialResult。
- 事件按 caller 分桶累计（agent / user_sim / judge / rag），供 cost 层算钱。
- **并发安全**：trial 状态放 contextvars（asyncio task 各自持有独立拷贝），
  因此 `eval --concurrency N` 并发压测时各 trial 互不覆盖。
"""
from __future__ import annotations

import contextvars
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.settings import settings
from ..core.tracing import LLMEvent, PermissionEventEvt, RouteEvent, ToolEvent, tracing

# trial 状态按 context 隔离（asyncio.create_task 会拷贝父 context）
_current_var: contextvars.ContextVar["TrialStats | None"] = (
    contextvars.ContextVar("monitor_current", default=None)
)
_event_var: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("monitor_event_log", default=None)
)


@dataclass
class TrialStats:
    trial_id: str = ""
    harness: str = "custom"
    agent_id: str = "1"
    golden_task_id: str = ""
    context_policy: str = ""
    # LLM 累计
    agent_in: int = 0
    agent_out: int = 0
    user_sim_in: int = 0
    user_sim_out: int = 0
    judge_in: int = 0
    judge_out: int = 0
    rag_in: int = 0
    rag_out: int = 0
    llm_calls: int = 0
    # 时延
    ttfts: list[float] = field(default_factory=list)
    tpots: list[float] = field(default_factory=list)
    totals: list[float] = field(default_factory=list)
    # 工具
    tool_calls: int = 0
    tool_errors: int = 0
    tool_latencies: list[float] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    # 权限
    blocked_attempts: int = 0
    permission_attempts: int = 0
    # 路由
    router_calls: int = 0
    transfers: int = 0
    skill_loads: int = 0
    # 轨迹
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    # 发散
    repeated_actions: int = 0
    seen_tool_calls: set = field(default_factory=set)
    # 外部成本
    external_cost_rmb: float = 0.0


class LLMInteractionMonitor:
    """评测数据采集器。"""

    def __init__(self, traces_dir: str | Path | None = None) -> None:
        self.traces_dir = Path(traces_dir or settings.get("data_dir", "data")) / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        tracing.register(self._on_event)

    # ------------------------------------------------------------------
    # trial 生命周期
    # ------------------------------------------------------------------
    def begin_trial(
        self,
        harness: str = "custom",
        agent_id: str = "1",
        golden_task_id: str = "",
        context_policy: str = "",
    ) -> str:
        trial_id = uuid.uuid4().hex[:8]
        _current_var.set(TrialStats(
            trial_id=trial_id,
            harness=harness,
            agent_id=agent_id,
            golden_task_id=golden_task_id,
            context_policy=context_policy,
        ))
        _event_var.set([])
        return trial_id

    @property
    def current(self) -> TrialStats | None:
        return _current_var.get()

    def end_trial(
        self,
        outcome: str = "success",
        success_signal: dict[str, float] | None = None,
        steps: int = 0,
        meta: dict[str, Any] | None = None,
        safety: dict[str, Any] | None = None,
        expected_steps: int = 0,
    ) -> dict[str, Any]:
        """结束 trial，返回统一的 TrialResult dict（供 eval 聚合）。

        safety 由调用方（harness/judge）提供真实判定；未提供时仅保证 guard_ok
        反映真实的权限拦截统计，injection/pii 明确标注为未判定（None → 不落入布尔）。
        """
        s = _current_var.get()
        assert s is not None, "end_trial 前必须先 begin_trial"
        # 成本汇总
        cost_in, cost_out = settings.cost_rates
        result = {
            "trial_id": s.trial_id,
            "harness": s.harness,
            "agent_id": s.agent_id,
            "golden_task_id": s.golden_task_id,
            "context_policy": s.context_policy,
            "outcome": outcome,
            "success_signal": success_signal or {},
            "steps": steps,
            "repeated_actions": s.repeated_actions,
            # 有效绕路：超过"期望步数"的部分才算绕路（期望步数默认 0 = 无黄金流任务不惩罚）
            "detour_actions": max(0, steps - expected_steps) if steps > expected_steps else 0,
            "router_calls": s.router_calls,
            "cost": {
                "agent_in": s.agent_in,
                "agent_out": s.agent_out,
                "user_sim_in": s.user_sim_in,
                "user_sim_out": s.user_sim_out,
                "judge_in": s.judge_in,
                "judge_out": s.judge_out,
                "tool_calls": s.tool_calls,
                "external_cost_rmb": round(s.external_cost_rmb, 6),
            },
            "latency_ms": self._aggregate_latency(),
            "tool_latency_ms": self._raw_tool_latency(),
            "trajectory": s.trajectory,
            "permission": {
                "blocked_attempts": s.blocked_attempts,
                "total_attempts": s.permission_attempts,
            },
            "safety": {
                "injection_triggered": (safety or {}).get("injection_triggered"),
                "pii_leak": (safety or {}).get("pii_leak"),
                "guard_ok": s.blocked_attempts == 0,
                **({k: v for k, v in (safety or {}).items() if k not in ("injection_triggered", "pii_leak")}),
            },
            "meta": meta or {},
        }
        # 落盘事件日志
        self._flush_event_log()
        _current_var.set(None)
        return result

    def _raw_tool_latency(self) -> dict[str, Any]:
        """原始按工具耗时：{tool: [latency_ms, ...]}，供聚合层算真实 p95/max。"""
        s = _current_var.get()
        by_tool: dict[str, list[float]] = {}
        for name, lat in zip(s.tool_names, s.tool_latencies):
            by_tool.setdefault(name, []).append(lat)
        return by_tool

    def _aggregate_latency(self) -> dict[str, float | None]:
        s = _current_var.get()
        totals = sorted(t for t in s.totals if t is not None)
        ttfts = sorted(t for t in s.ttfts if t is not None)
        tpots = sorted(t for t in s.tpots if t is not None)

        def pct(arr: list[float], p: float) -> float | None:
            if not arr:
                return None
            idx = min(len(arr) - 1, int(len(arr) * p))
            return round(arr[idx], 1)

        return {
            "ttft": round(ttfts[0], 1) if ttfts else None,
            "tpot": round(sum(tpots) / len(tpots), 1) if tpots else None,
            "total": round(totals[-1], 1) if totals else None,
            "p95": pct(totals, 0.95),
            "p99": pct(totals, 0.99),
        }

    # ------------------------------------------------------------------
    # 事件接收
    # ------------------------------------------------------------------
    def _on_event(self, event: Any) -> None:
        s = _current_var.get()
        if s is None:
            return
        if isinstance(event, LLMEvent):
            if event.caller == "agent":
                s.agent_in += event.input_tokens
                s.agent_out += event.output_tokens
            elif event.caller == "user_sim":
                s.user_sim_in += event.input_tokens
                s.user_sim_out += event.output_tokens
            elif event.caller == "judge":
                s.judge_in += event.input_tokens
                s.judge_out += event.output_tokens
            else:  # rag
                s.rag_in += event.input_tokens
                s.rag_out += event.output_tokens
            s.llm_calls += 1
            s.external_cost_rmb += event.cost_rmb
            if event.ttft_ms is not None:
                s.ttfts.append(event.ttft_ms)
            if event.tpot_ms is not None:
                s.tpots.append(event.tpot_ms)
            if event.total_ms is not None:
                s.totals.append(event.total_ms)
            s.trajectory.append(
                {"step": s.tool_calls + 1, "type": "llm", "name": event.caller, "ok": True}
            )
        elif isinstance(event, ToolEvent):
            s.tool_calls += 1
            s.tool_latencies.append(event.latency_ms)
            s.tool_names.append(event.name)
            if not event.ok:
                s.tool_errors += 1
            s.trajectory.append(
                {"step": s.tool_calls, "type": "tool", "name": event.name, "ok": event.ok}
            )
        elif isinstance(event, RouteEvent):
            s.trajectory.append(
                {"step": s.tool_calls + 1, "type": event.kind, "name": f"{event.source}->{event.target}", "ok": True}
            )
            if event.kind == "route":
                s.router_calls += 1
            elif event.kind == "transfer":
                s.transfers += 1
            else:
                s.skill_loads += 1
        elif isinstance(event, PermissionEventEvt):
            s.permission_attempts += 1
            if event.decision != "allow":
                s.blocked_attempts += 1
            s.trajectory.append(
                {"step": s.tool_calls + 1, "type": "permission", "name": event.decision, "ok": event.decision == "allow"}
            )
        # 原始事件落盘（JSONL）：修复 _flush_event_log 恒早退、trial 文件全空的问题
        (_event_var.get() or []).append(self._event_to_dict(event))

    @staticmethod
    def _event_to_dict(event: Any) -> dict[str, Any]:
        """把事件 dataclass 序列化为 JSON 可写 dict。"""
        d: dict[str, Any] = {"type": type(event).__name__}
        for k, v in vars(event).items():
            if isinstance(v, dict):
                d[k] = v
            elif isinstance(v, (str, int, float, bool)) or v is None:
                d[k] = v
            else:
                d[k] = str(v)
        return d

    def _flush_event_log(self) -> None:
        event_log = _event_var.get() or []
        if not event_log:
            return
        s = _current_var.get()
        path = self.traces_dir / f"trial_{s.trial_id}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for line in event_log:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        _event_var.set([])


# 全局单例（所有模块共享）
monitor = LLMInteractionMonitor()
