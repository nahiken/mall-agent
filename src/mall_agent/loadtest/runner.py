"""压测 runner：阶梯并发打 /api/mall/chat/stream，采集高并发容量指标。

指标口径（对应需求）：
- 最大并发数   错误率 ≤ 1% 且无超时的最大并发档位
- 峰值 QPS     压测全程 1s 窗口完成请求数最大值
- 稳定 QPS     稳态段（错误率<1% 且 P95 稳定）的平均 QPS
- 最大吞吐量   单位时间传输数据量（响应字节/s）峰值
- 错误率       失败请求占比（稳定线 < 1%）
- 资源利用率   psutil 进程级+系统级 CPU/内存/网络，峰值

方法：并发从 1 翻倍到 max_concurrency，每档跑 duration/档数 秒，
每档采样请求（持续发出，记录完成），记录每档 qps/延迟/错误/吞吐，
并用 psutil 采集 CPU/内存/网络峰值。
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


DEFAULT_PAYLOAD = {"messages": [{"role": "user", "content": "帮我查一下我的订单"}], "user_id": "loadtest"}


@dataclass
class RampResult:
    concurrency: int
    total: int = 0
    ok: int = 0
    fail: int = 0
    timeouts: int = 0
    durations_ms: list[float] = field(default_factory=list)
    total_bytes: int = 0
    qps: float = 0.0
    p50_ms: float | None = None
    p95_ms: float | None = None
    error_rate: float = 0.0
    bytes_per_sec: float = 0.0
    max_qps_window: float = 0.0
    cpu_peak: float = 0.0
    mem_mb_peak: float = 0.0
    net_sent_mb_peak: float = 0.0
    net_recv_mb_peak: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency, "total": self.total, "ok": self.ok,
            "fail": self.fail, "timeouts": self.timeouts, "qps": round(self.qps, 2),
            "p50_ms": self.p50_ms, "p95_ms": self.p95_ms,
            "error_rate": round(self.error_rate, 4),
            "throughput_bytes_per_sec": round(self.bytes_per_sec, 1),
            "max_qps_window": round(self.max_qps_window, 2),
            "cpu_peak": round(self.cpu_peak, 1), "mem_mb_peak": round(self.mem_mb_peak, 1),
            "net_sent_mb_peak": round(self.net_sent_mb_peak, 3),
            "net_recv_mb_peak": round(self.net_recv_mb_peak, 3),
        }


class PsutilSampler:
    """CPU/内存/网络采样（进程级 + 系统级）。"""

    def __init__(self) -> None:
        self.proc = psutil.Process() if psutil else None
        self._prev_net: tuple[int, int] | None = None
        self._prev_t = time.time()
        self.net_sent = 0.0
        self.net_recv = 0.0

    def sample(self, cpu: list[float], mem: list[float]) -> None:
        if self.proc is None:
            return
        cpu.append(self.proc.cpu_percent(interval=None))
        mem.append(self.proc.memory_info().rss / 1024 / 1024)
        # 网络（系统级累计 → 增量）
        if psutil:
            io = psutil.net_io_counters()
            now = time.time()
            if self._prev_net:
                dt = max(0.001, now - self._prev_t)
                self.net_sent += (io.bytes_sent - self._prev_net[0]) / dt / 1024 / 1024
                self.net_recv += (io.bytes_recv - self._prev_net[1]) / dt / 1024 / 1024
            self._prev_net = (io.bytes_sent, io.bytes_recv)
            self._prev_t = now


class LoadTestRunner:
    def __init__(self, url: str, max_concurrency: int, duration: float,
                 payload: dict | None = None, timeout: float = 60.0,
                 max_qps_limit: float | None = None) -> None:
        self.url = url
        self.max_concurrency = max_concurrency
        self.duration = duration
        self.payload = payload or DEFAULT_PAYLOAD
        self.timeout = timeout
        self.max_qps_limit = max_qps_limit
        self.sampler = PsutilSampler()

    # ------------------------------------------------------------------
    # 单请求（SSE：POST /chat/stream，读完整流）
    # ------------------------------------------------------------------
    async def _one(self, client, req_id: int) -> tuple[float, bool, int]:
        """返回 (耗时ms, 是否成功, 响应字节数)。"""
        import httpx

        t0 = time.perf_counter()
        bytes_n = 0
        try:
            async with client.stream("POST", self.url, json=self.payload,
                                     timeout=httpx.Timeout(self.timeout)) as resp:
                if resp.status_code != 200:
                    return (time.perf_counter() - t0) * 1000, False, bytes_n
                # 读完整 SSE 流（模拟真实前端消费）
                async for chunk in resp.aiter_bytes():
                    bytes_n += len(chunk)
                return (time.perf_counter() - t0) * 1000, True, bytes_n
        except Exception:
            return (time.perf_counter() - t0) * 1000, False, bytes_n

    # ------------------------------------------------------------------
    # 单档压测
    # ------------------------------------------------------------------
    async def _ramp(self, concurrency: int, seconds: float) -> RampResult:
        import httpx

        res = RampResult(concurrency=concurrency)
        # 采样间隔任务
        stop = asyncio.Event()
        cpu_vals: list[float] = []
        mem_vals: list[float] = []

        async def sampler_loop():
            while not stop.is_set():
                self.sampler.sample(cpu_vals, mem_vals)
                await asyncio.sleep(0.1)

        sampler_task = asyncio.create_task(sampler_loop())

        # 固定并发 worker：每个 worker 循环发请求直到时间到
        t_end = time.perf_counter() + seconds
        # 1s QPS 窗口：统计每 1s 完成数 → 峰值窗口 QPS
        qps_windows: dict[int, int] = {}
        qps_lock = asyncio.Lock()

        async def worker():
            async with httpx.AsyncClient(timeout=self.timeout, limits=httpx.Limits(
                    max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)) as client:
                req_id = 0
                while time.perf_counter() < t_end:
                    req_id += 1
                    dt_ms, ok, bn = await self._one(client, req_id)
                    dur_sec = (time.perf_counter() - t_end + seconds)  # 已经过的秒数近似
                    # 记录窗口（按完成时间整秒）
                    now_int = int(time.perf_counter())
                    async with qps_lock:
                        qps_windows[now_int] = qps_windows.get(now_int, 0) + 1
                    res.total += 1
                    res.durations_ms.append(dt_ms)
                    res.total_bytes += bn
                    if ok:
                        res.ok += 1
                    else:
                        res.fail += 1
                        if dt_ms >= self.timeout * 1000:
                            res.timeouts += 1

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)
        stop.set()
        await sampler_task

        # 计算指标
        if res.total:
            res.qps = res.total / seconds
            res.bytes_per_sec = res.total_bytes / seconds  # 吞吐（字节/s）
            res.error_rate = res.fail / res.total
            d = sorted(res.durations_ms)
            res.p50_ms = round(statistics.median(d), 1)
            res.p95_ms = round(d[min(len(d) - 1, int(len(d) * 0.95))], 1)
        if qps_windows:
            res.max_qps_window = max(qps_windows.values())
        if cpu_vals:
            res.cpu_peak = max(cpu_vals)
        if mem_vals:
            res.mem_mb_peak = max(mem_vals)
        res.net_sent_mb_peak = self.sampler.net_sent
        res.net_recv_mb_peak = self.sampler.net_recv
        return res

    # ------------------------------------------------------------------
    # 跑全部档位
    # ------------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        # 档位：1,2,4,...,max（保证最后一档=max）
        levels: list[int] = []
        v = 1
        while v < self.max_concurrency:
            levels.append(v)
            v *= 2
        levels.append(self.max_concurrency)

        n_levels = len(levels)
        per_level = self.duration / n_levels
        ramps: list[RampResult] = []
        for lv in levels:
            print(f"  ramp concurrency={lv} ({per_level:.1f}s)...", flush=True)
            r = await self._ramp(lv, per_level)
            ramps.append(r)
            print(f"    -> qps={r.qps:.1f} ok={r.ok} fail={r.fail} err={r.error_rate:.3f} "
                  f"p95={r.p95_ms}ms cpu_peak={r.cpu_peak}%", flush=True)

        return self._summarize(ramps)

    # ------------------------------------------------------------------
    # 汇总 6 项指标
    # ------------------------------------------------------------------
    def _summarize(self, ramps: list[RampResult]) -> dict[str, Any]:
        # 最大并发：错误率≤1% 且无超时的最大档位
        max_conc = 0
        for r in ramps:
            if r.error_rate <= 0.01 and r.timeouts == 0 and r.total > 0:
                max_conc = r.concurrency
        # 峰值 QPS：全程 1s 窗口最大
        peak_qps = max((r.max_qps_window for r in ramps), default=0)
        # 稳定 QPS：错误率<1% 档位的平均 qps（最接近最大并发且稳定的档）
        stable_ramps = [r for r in ramps if r.error_rate <= 0.01 and r.qps > 0]
        stable_qps = round(statistics.mean([r.qps for r in stable_ramps]), 2) if stable_ramps else 0.0
        # 吞吐峰值（字节/s → MB/s）
        peak_bps = max((r.bytes_per_sec for r in ramps), default=0)
        # 错误率：最大并发档的错误率
        final = ramps[-1] if ramps else None
        error_rate = final.error_rate if final else 0.0
        # 资源峰值
        cpu = max((r.cpu_peak for r in ramps), default=0)
        mem = max((r.mem_mb_peak for r in ramps), default=0)
        net_s = max((r.net_sent_mb_peak for r in ramps), default=0)
        net_r = max((r.net_recv_mb_peak for r in ramps), default=0)

        return {
            "generated_at": time.time(),
            "url": self.url,
            "max_concurrency_target": self.max_concurrency,
            "metrics": {
                "max_concurrency": max_conc,
                "peak_qps": round(peak_qps, 2),
                "stable_qps": stable_qps,
                "peak_throughput_mbps": round(peak_bps / 1024 / 1024, 3),
                "error_rate": round(error_rate, 4),
                "error_rate_ok": error_rate <= 0.01,
                "resource": {
                    "cpu_percent_peak": round(cpu, 1),
                    "mem_mb_peak": round(mem, 1),
                    "net_sent_mbps_peak": round(net_s, 3),
                    "net_recv_mbps_peak": round(net_r, 3),
                },
            },
            "ramps": [r.to_dict() for r in ramps],
        }


def save_report(result: dict[str, Any], path: Path | None = None) -> Path:
    from ..core.settings import settings
    out = path or (Path(settings.get("data_dir", "data")) / "loadtest" /
                   f"loadtest_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
