"""实时监控 API（前缀 /api/monitor）。

- GET /stream    SSE 事件流（实时推送，供监控页图表）
- GET /snapshot  当前 60s 滚动窗口统计（轮询兜底）
- GET /loadtests 历史压测结果列表（data/loadtest/*.json，供压测指标页）
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core.settings import settings
from ..monitoring.realtime import realtime
from .sse import sse_event, sse_comment


def build_monitor_router() -> APIRouter:
    router = APIRouter(prefix="/api/monitor", tags=["monitor"])

    @router.get("/stream")
    async def stream() -> StreamingResponse:
        async def gen():
            yield sse_comment("monitor-stream")
            q = realtime.subscribe()
            try:
                while True:
                    evt = await q.get()
                    yield sse_event("event", evt)
            except asyncio.CancelledError:  # noqa: F821
                raise
            finally:
                realtime.unsubscribe()

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.get("/snapshot")
    async def snapshot() -> dict:
        return realtime.snapshot()

    @router.get("/loadtests")
    async def loadtests() -> dict:
        """列出 data/loadtest/ 下的压测结果（前端压测指标页用）。"""
        d = Path(settings.get("data_dir", "data")) / "loadtest"
        files: list[dict[str, Any]] = []
        if d.exists():
            for p in sorted(d.glob("loadtest_*.json"), reverse=True)[:20]:
                try:
                    files.append(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    continue
        return {"items": files, "total": len(files)}

    return router
