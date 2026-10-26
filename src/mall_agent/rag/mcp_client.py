"""RAG-MCP-SERVER 的 stdio 客户端。

通过 MCP 协议（stdio 子进程）调用 RAG-MCP-SERVER 的 9 个工具：
    query_knowledge_hub / semantic_search / list_collections / get_document_summary
    list_directory / list_documents / preview_document / verify_fact / self_check

生命周期参照官方脚本 scripts/mcp_client.py：
    StdioServerParameters(command=python, args=[-m, module], cwd=ROOT,
                          env={PYTHONPATH: SRC, PYTHONIOENCODING: utf-8})
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..core.settings import settings
from ..core.tracing import LLMEvent, tracing

logger = logging.getLogger(__name__)


class RagMcpClient:
    """RAG MCP stdio 客户端（进程内单例，懒连接，失败降级）。"""

    def __init__(self) -> None:
        self._session = None
        self._proc = None
        self._lock = asyncio.Lock()       # 保护 connect
        self._call_lock = asyncio.Lock()  # 串行化 tool_call（stdio 单写流）
        self._connected = False
        self._tools: list[dict] = []
        self._connect_timeout = 15.0
        self._call_timeout = 30.0

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    async def connect(self) -> bool:
        if self._connected:
            return True
        async with self._lock:
            if self._connected:
                return True
            try:
                from mcp import ClientSession, StdioServerParameters, stdio_client

                rag = settings.rag
                python = settings.env("RAG_MCP_PYTHON") or rag.get("python")
                if not python:
                    logger.warning("RAG_MCP_PYTHON 未配置，RAG 能力不可用（将降级）")
                    return False
                params = StdioServerParameters(
                    command=python,
                    args=["-m", rag.get("module", "rag_mcp_server.mcp_server.server")],
                    cwd=rag.get("cwd"),
                    env={
                        "PYTHONPATH": rag.get("src"),
                        "PYTHONIOENCODING": "utf-8",
                        "PYTHONUTF8": "1",
                    },
                )
                self._proc = stdio_client(params)
                read, write = await asyncio.wait_for(self._proc.__aenter__(), timeout=self._connect_timeout)
                self._session = await asyncio.wait_for(
                    ClientSession(read, write).__aenter__(), timeout=self._connect_timeout)
                await asyncio.wait_for(self._session.initialize(), timeout=self._connect_timeout)
                self._connected = True
                self._tools = await self._list_tools()
                logger.info(f"✅ RAG-MCP 已连接，{len(self._tools)} 个工具")
                return True
            except Exception as e:
                logger.error(f"RAG-MCP 连接失败: {e}")
                self._connected = False
                # 连接中途失败时清理已开启的子进程/会话，避免泄漏
                await self.close()
                return False

    async def _list_tools(self) -> list[dict]:
        try:
            tools = await self._session.list_tools()
            return [t.name for t in tools]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 工具调用
    # ------------------------------------------------------------------
    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict:
        """调用 MCP 工具，返回 {text, references, has_image}。

        并发安全：同 session 串行化 tool_call（stdio 单写流）。
        自动重连：调用失败且仍标记 connected → 清理会话置为未连接 →
        重连一次 → 重发；仍失败才降级。实现子进程死亡自愈。
        """
        if not self._connected:
            ok = await self.connect()  # connect 内部持 _lock 双重检查
            if not ok:
                return {"text": "[RAG] 知识库服务不可用", "references": [], "has_image": False}
        async with self._call_lock:
            try:
                return await self._call_once(tool, arguments)
            except Exception as e:
                # 会话可能已死（子进程崩溃/流断开）→ 重置连接状态并重连一次
                logger.warning("RAG tool %s 调用失败: %s；尝试重连重发", tool, e)
                await self.close()  # 重置 _connected=False
                ok = await self.connect()
                if not ok:
                    return {"text": f"[RAG] 调用 {tool} 失败", "references": [], "has_image": False}
                try:
                    return await self._call_once(tool, arguments)
                except Exception as e2:
                    logger.error(f"RAG tool {tool} 重连后仍失败: {e2}")
                    return {"text": f"[RAG] 调用 {tool} 失败", "references": [], "has_image": False}

    async def _call_once(self, tool: str, arguments: dict[str, Any] | None) -> dict:
        """单次 MCP 工具调用（不重连）。"""
        result = await asyncio.wait_for(
            self._session.call_tool(tool, arguments or {}), timeout=self._call_timeout)
        return self._parse_result(result)

    @staticmethod
    def _parse_result(result: Any) -> dict:
        text_parts: list[str] = []
        refs: list[dict] = []
        has_image = False
        for block in result.content or []:
            if block.type == "text":
                text_parts.append(block.text)
                try:
                    import json
                    data = json.loads(block.text)
                    if isinstance(data, dict) and "citations" in data:
                        refs = data.get("citations", [])
                except Exception:
                    pass
            elif block.type == "image":
                has_image = True
        return {"text": "\n".join(text_parts), "references": refs, "has_image": has_image}

    # ------------------------------------------------------------------
    # 高层语义：供 Agent 工具直接调用
    # ------------------------------------------------------------------
    async def query_knowledge(self, query: str, hub: str = "cs", top_k: int = 3) -> str:
        """按 hub 路由查询知识库（query_knowledge_hub + 集合选择）。"""
        hub_cfg = settings.rag.get("hubs", {}).get(hub, {})
        result = await self.call("query_knowledge_hub", {
            "query": query,
            "index_dir": hub_cfg.get("index_dir", "data/index"),
            "collection": hub_cfg.get("collection"),
            "top_k": top_k,
        })
        return result["text"]

    async def verify_fact(self, claim: str, hub: str = "policy") -> str:
        """事实核查（RAG 侧 verify_fact）。"""
        hub_cfg = settings.rag.get("hubs", {}).get(hub, {})
        result = await self.call("verify_fact", {
            "claim": claim,
            "index_dir": hub_cfg.get("index_dir", "data/index"),
            "collection": hub_cfg.get("collection"),
        })
        return result["text"]

    async def list_collections(self) -> str:
        result = await self.call("list_collections")
        return result["text"]

    async def close(self) -> None:
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
            if self._proc:
                await self._proc.__aexit__(None, None, None)
        except Exception:
            pass
        self._session = None
        self._proc = None
        self._connected = False


# 全局单例
_rag_client: RagMcpClient | None = None


def get_rag_client() -> RagMcpClient:
    global _rag_client
    if _rag_client is None:
        _rag_client = RagMcpClient()
    return _rag_client
