"""CLI 压测：python -m mall_agent.cli.load_test

用法：
    # 对已启动的 mock-agent 服务压测（测服务层吞吐）
    python -m mall_agent.server --mock-agent --port 8002
    python -m mall_agent.cli.load_test --url http://localhost:8002/api/mall/chat/stream --max-concurrency 64 --duration 30

    # 对真实 agent 服务压测（受 LLM 限速，注意成本）
    python -m mall_agent.server --port 8002
    python -m mall_agent.cli.load_test --url http://localhost:8002/api/mall/chat/stream --max-concurrency 16 --duration 20

输出：data/loadtest/loadtest_{ts}.json（前端压测指标页读取）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path


def _aggregate(results: list[dict], clients: int) -> dict:
    """合并 N 个压测进程的结果：QPS 相加，延迟取各档最大值，错误率按请求数加权。"""
    ramps: dict[int, dict] = {}
    for r in results:
        for rp in r.get("ramps", []):
            c = rp["concurrency"]
            a = ramps.setdefault(c, {"concurrency": c, "total": 0, "ok": 0, "fail": 0,
                                     "timeouts": 0, "qps": 0.0, "p50_ms": 0.0,
                                     "p95_ms": 0.0, "cpu_peak": 0.0})
            for k in ("total", "ok", "fail", "timeouts"):
                a[k] += rp.get(k, 0) or 0
            a["qps"] += rp.get("qps", 0) or 0
            for k in ("p50_ms", "p95_ms", "cpu_peak"):
                a[k] = max(a[k], rp.get(k, 0) or 0)

    rows = [ramps[c] for c in sorted(ramps)]
    for a in rows:
        a["qps"] = round(a["qps"], 2)
        a["error_rate"] = round(a["fail"] / a["total"], 4) if a["total"] else 0.0
        a["p50_ms"] = round(a["p50_ms"], 1)
        a["p95_ms"] = round(a["p95_ms"], 1)
        a["cpu_peak"] = round(a["cpu_peak"], 1)

    peak = max((r.get("metrics", {}).get("peak_qps", 0) or 0) for r in results) if results else 0
    peak_sum = sum((r.get("metrics", {}).get("peak_qps", 0) or 0) for r in results)
    stable_sum = sum((r.get("metrics", {}).get("stable_qps", 0) or 0) for r in results)
    total = sum(a["total"] for a in rows)
    fail = sum(a["fail"] for a in rows) + sum(a["timeouts"] for a in rows)
    return {
        "url": results[0].get("url") if results else "",
        "clients": clients,
        "max_concurrency_target": max((r.get("max_concurrency_target", 0) for r in results), default=0),
        "metrics": {
            "max_concurrency": max((r.get("metrics", {}).get("max_concurrency", 0) for r in results), default=0),
            "peak_qps": round(peak_sum if clients > 1 else peak, 2),
            "stable_qps": round(stable_sum, 2),
            "error_rate": round(fail / total, 4) if total else 0.0,
            "error_rate_ok": (fail / total if total else 0) <= 0.01,
            "total_requests": total,
        },
        "ramps": rows,
    }


def _run_children(args) -> dict:
    """多进程压测：N 个子进程各压一路，合并结果。

    动机：单进程压测客户端在 ~15-20 QPS 时自身 CPU 打满，会把「客户端瓶颈」
    误读成「服务端上限」。多进程后才测得到服务端真实容量。
    """
    import os
    import subprocess
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="mall_lt_"))
    procs, outs = [], []
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "src")}
    for i in range(args.clients):
        out = tmpdir / f"c{i}.json"
        cmd = [sys.executable, "-X", "utf8", "-m", "mall_agent.cli.load_test",
               "--url", args.url,
               "--max-concurrency", str(args.max_concurrency),
               "--duration", str(args.duration),
               "--timeout", str(args.timeout),
               "--clients", "1", "--json-out", str(out)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
        outs.append(out)
    for p in procs:
        p.wait()
    results = [json.loads(o.read_text(encoding="utf-8")) for o in outs if o.exists()]
    return _aggregate(results, args.clients)


def main() -> None:
    parser = argparse.ArgumentParser(description="mall-agent 高并发压测")
    parser.add_argument("--url", default="http://localhost:8002/api/mall/chat/stream")
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--duration", type=float, default=20.0, help="总时长（秒，按档位均分）")
    parser.add_argument("--timeout", type=float, default=60.0, help="单请求超时（秒）")
    parser.add_argument("--clients", type=int, default=1,
                        help="压测进程数（>1 时消除客户端单进程 CPU 瓶颈）")
    parser.add_argument("--json-out", default=None, help="结果写入指定路径（子进程用）")
    args = parser.parse_args()

    from ..loadtest.runner import LoadTestRunner, save_report

    if args.clients > 1:
        merged = _run_children(args)
        merged["generated_at"] = time.time()
        path = save_report(merged)
        print(json.dumps(merged["metrics"], ensure_ascii=False, indent=1))
        print(f"\n报告已保存: {path}")
        sys.exit(0 if merged["metrics"]["error_rate_ok"] else 1)

    async def _run():
        print(f"压测 {args.url}")
        print(f"档位：1 → {args.max_concurrency}（时长 {args.duration}s）")
        runner = LoadTestRunner(args.url, args.max_concurrency, args.duration, timeout=args.timeout)
        result = await runner.run()
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            path = args.json_out
        else:
            path = save_report(result)
        print("\n=== 压测指标 ===")
        m = result["metrics"]
        print(json.dumps(m, ensure_ascii=False, indent=1))
        print(f"\n报告已保存: {path}")
        # 退出码：错误率<1% 视为稳定通过
        return 0 if m["error_rate_ok"] else 1

    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
