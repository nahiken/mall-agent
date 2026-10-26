"""FastAPI 应用工厂：注册工具 → build_agent → 装配路由。

启动前必须注册工具（build_agent 不注册），否则 registry 空、agent 无工具可调。
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..agents.graph_factory import build_agent
from ..tools.admin_tools import register_admin_tools
from ..tools.cs_tools import register_cs_tools
from ..tools.guide_tools import register_guide_tools


def register_tools() -> None:
    """幂等注册全部工具（cs/guide/admin）。"""
    register_cs_tools()
    register_guide_tools()
    register_admin_tools()


def create_app(scheme: str = "1", context_policy: str = "full",
               agent: Any = None, agent_factory: Any = None,
               rag_enabled: bool = True) -> FastAPI:
    """创建应用。

    agent_factory：可注入的自定义 agent 工厂（压测 mock 模式用），
    返回 (agent, conv_service) 或直接 MallAgent。缺省用 build_agent。
    rag_enabled：进程级 RAG 开关（R1）。False 时 build_agent 屏蔽 domain="rag"
    检索工具 → agent 无外部知识（用于 ON/OFF 吞吐对照压测，阶段 C）。
    """
    register_tools()

    # 构建 agent（可注入）
    if agent_factory is not None:
        built = agent_factory(scheme, context_policy, rag_enabled)
    elif agent is not None:
        built = agent
    else:
        built = build_agent(scheme=scheme, context_policy=context_policy,
                            rag_enabled=rag_enabled)

    from .conversation import ConversationService
    if isinstance(built, tuple):
        mall_agent, conv = built
    elif isinstance(built, ConversationService):
        mall_agent, conv = None, built
    else:
        mall_agent, conv = built, ConversationService(built)

    from .agent_api import build_agent_router
    from .monitor_api import build_monitor_router

    app = FastAPI(title="mall-agent API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(build_agent_router(conv))
    app.include_router(build_monitor_router())

    # 全局引用（压测/监控读取）
    app.state.agent = mall_agent
    app.state.conv = conv
    app.state.scheme = scheme

    @app.get("/")
    async def root() -> dict:
        return {"service": "mall-agent", "scheme": scheme, "health": "/api/mall/health"}

    return app
