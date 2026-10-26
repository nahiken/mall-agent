"""P1 验证：工具注册 / 权限守卫 / 数据探针 / RAG-MCP 连接。

用法：python -m mall_agent.cli.p1_check
"""
from __future__ import annotations

import asyncio
import json


def _setup() -> None:
    """注册全部工具 + 构建 guard + 绑定 mock 探针。"""
    from mall_agent.tools.cs_tools import register_cs_tools
    from mall_agent.tools.guide_tools import register_guide_tools
    from mall_agent.tools.admin_tools import register_admin_tools
    from mall_agent.tools.registry import registry

    register_cs_tools()
    register_guide_tools()
    register_admin_tools()
    return registry


async def main() -> None:
    print("=" * 50)
    print("P1 工具/权限/RAG 自检")
    print("=" * 50)

    registry = _setup()
    print(f"\n[1/5] 工具注册: {len(registry.names())} 个")
    print(f"  客服工具: {[n for n in registry.names() if n in ('search_order','cancel_order','query_policy','list_address')]}")
    print(f"  导购工具: {[n for n in registry.names() if n in ('search_product','add_to_cart','check_stock')]}")
    print(f"  admin工具(越权测试): {[n for n in registry.names() if n in ('update_product_stock','force_refund')]}")

    # 2. 权限守卫：构建 guard + mock 探针
    print("\n[2/5] 权限守卫三层检查")
    from mall_agent.mall_api.client import get_mall_client
    from mall_agent.permission.model import Session, build_default_guard
    from mall_agent.permission.probe import MockDataProbe

    client = get_mall_client()
    await client.ensure_session()
    mock = client._mock
    probe = MockDataProbe(mock)
    guard = build_default_guard(data_probe=probe)

    cs_session = Session(role="cs", user_id=str(client._user_id), consent="implicit")
    guide_session = Session(role="guide", user_id=str(client._user_id), consent="implicit")

    # 2a. 客服调 cancel_order（工具级通过，但敏感级需 explicit consent → 拦截）
    spec = registry.get("cancel_order")
    ok, evt = guard.check(spec.scope, {"order_id": 1}, cs_session)
    print(f"  [客服取消订单·未显式确认] allow={ok} decision={evt.decision}")
    assert not ok and evt.decision == "block_sensitive", "敏感操作未确认应被拦截"

    # 2b. 导购调 cancel_order（工具级拦截：guide 无 order.operate）
    ok, evt = guard.check(registry.get("cancel_order").scope, {"order_id": 1}, guide_session, tool_name="cancel_order")
    print(f"  [导购取消订单] allow={ok} decision={evt.decision}")
    assert not ok and evt.decision == "block_tool", "导购无权取消订单"

    # 2c. 导购调 admin 工具（工具级拦截）
    ok, evt = guard.check(registry.get("update_product_stock").scope, {"product_id": 1, "stock": 10}, guide_session, tool_name="update_product_stock")
    print(f"  [导购改库存] allow={ok} decision={evt.decision}")
    assert not ok and evt.decision == "block_tool", "导购无权改库存"

    # 2d. 客服查自己的订单（放行）
    ok, evt = guard.check(registry.get("search_order").scope, {"status": None}, cs_session, tool_name="search_order")
    print(f"  [客服查订单·合法] allow={ok} decision={evt.decision}")
    assert ok, "合法操作应放行"

    # 3. 数据级：own_order（跨用户订单应被拦截）
    print("\n[3/5] 数据级 own_order 校验")
    spec = registry.get("get_order_detail")
    # 给测试用户建一个订单
    uid = client._user_id
    order_data = await client._post("/order/generateOrder", {"user_id": uid, "totalAmount": 100})
    order_id = order_data["data"].get("order_id")
    ok, evt = guard.check(spec.scope, {"order_id": order_id}, cs_session, tool_name="get_order_detail")
    print(f"  [查自己订单#{order_id}] allow={ok}")
    assert ok, "自己订单应放行"
    # 伪造一个不存在的订单 id（探针会判定非归属）
    ok, evt = guard.check(spec.scope, {"order_id": 999999}, cs_session, tool_name="get_order_detail")
    print(f"  [查他人/不存在订单#999999] allow={ok} reason={evt.reason}")
    assert not ok and evt.decision == "block_data", "非本用户订单应被数据级拦截"

    # 4. 工具执行（经 ToolExecutor）
    print("\n[4/5] ToolExecutor 执行")
    from mall_agent.agents.nodes import ToolExecutor
    from mall_agent.core.types import AgentState, ToolCall

    executor = ToolExecutor(registry, guard, cs_session)
    state: AgentState = {"session_id": "p1", "role": "cs", "user_id": str(uid), "steps": 0,
                         "messages": [], "effective_messages": [], "tool_records": [],
                         "permission_events": [], "flags": {}}
    msg = await executor.execute_tool_call(ToolCall(name="search_order", arguments={}), state)
    print(f"  search_order 结果: {msg[:120]}")

    # 5. RAG-MCP 连接（若配置了 RAG_MCP_PYTHON 则连，否则提示跳过）
    print("\n[5/5] RAG-MCP 连接")
    from mall_agent.core.settings import settings
    rag_py = settings.env("RAG_MCP_PYTHON") or settings.rag.get("python")
    if not rag_py:
        print("  ⚠️ RAG_MCP_PYTHON 未配置，跳过 RAG-MCP 连接（P1 后期配置后再验）")
    else:
        from mall_agent.rag.mcp_client import get_rag_client
        rag = get_rag_client()
        ok = await rag.connect()
        print(f"  连接: {ok}")
        if ok:
            cols = await rag.list_collections()
            print(f"  list_collections: {cols[:100]}")
        else:
            print("  ⚠️ RAG-MCP 连接失败（可能是 RAG-MCP-SERVER 未建 venv，P1 后期处理）")

    print("\n✅ P1 核心检查通过")


if __name__ == "__main__":
    asyncio.run(main())
