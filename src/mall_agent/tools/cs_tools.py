"""客服工具集（cs_tools）：订单 / 会员 / 优惠券 / 售后 / 客服知识库。

每个工具带 domain/permission/roles 权限声明，经 @tool 装饰器进 ToolSpec。
所有工具均为 async def（ToolExecutor 会 await spec.invoke()）。
"""
from __future__ import annotations

from typing import Any

from mall_agent.mall_api import member as member_api
from mall_agent.mall_api import order as order_api
from mall_agent.mall_api import product as product_api
from mall_agent.rag.mcp_client import get_rag_client
from mall_agent.tools.base import tool
from mall_agent.tools.registry import registry


# ============================== 订单域 ==============================
@tool(domain="order", permission="read", roles={"cs", "supervisor"})
async def search_order(status: int | None = None, page_num: int = 0, page_size: int = 5) -> str:
    """查询当前用户订单列表。status: 0待付款/1待发货/2已发货/3已完成/4已关闭/5无效。"""
    orders = await order_api.list_orders(status=status, page_num=max(1, page_num), page_size=page_size)
    if not orders:
        return "没有找到订单"
    return "\n".join(f"订单#{o.get('id')} {o.get('order_sn')} 状态:{o.get('status')} 金额:¥{o.get('total_amount')}"
                     for o in orders)


@tool(domain="order", permission="read", roles={"cs", "supervisor"})
async def get_order_detail(order_id: int) -> str:
    """查看指定订单的详情。order_id: 订单编号。"""
    o = await order_api.get_order_detail(order_id)
    if not o:
        return f"未找到订单 {order_id}"
    return f"订单#{o.get('id')} {o.get('order_sn')} 状态:{o.get('status')} 金额:¥{o.get('total_amount')}"


@tool(domain="order", permission="operate", roles={"cs"}, sensitive=True)
async def cancel_order(order_id: int) -> str:
    """取消指定订单（敏感操作，需用户确认）。order_id: 订单编号。"""
    return await order_api.cancel_order(order_id)


@tool(domain="order", permission="operate", roles={"cs"}, sensitive=True)
async def confirm_receive_order(order_id: int) -> str:
    """确认收货（敏感操作，需用户确认）。order_id: 订单编号。"""
    return await order_api.confirm_receive_order(order_id)


# ============================== 会员/优惠券 ==============================
@tool(domain="member", permission="read", roles={"cs"})
async def list_address() -> str:
    """查看当前用户的收货地址列表。"""
    addrs = await member_api.list_address()
    if not addrs:
        return "暂无收货地址"
    return "\n".join(f"{a.get('id')}. {a.get('name')} {a.get('phone')} {a.get('province')}{a.get('city')}"
                     for a in addrs)


@tool(domain="coupon", permission="read", roles={"cs", "guide"})
async def list_coupons(use_status: int | None = None) -> str:
    """查看当前用户的优惠券。use_status: 0未使用/1已使用/2已过期。"""
    coupons = await member_api.list_coupons(use_status)
    if not coupons:
        return "暂无优惠券"
    return "\n".join(f"{c.get('id')}. {c.get('name')} 满{c.get('min_point')}减{c.get('amount')}"
                     for c in coupons)


@tool(domain="coupon", permission="operate", roles={"cs", "guide"})
async def add_coupon(coupon_id: int) -> str:
    """领取指定优惠券。coupon_id: 优惠券ID。"""
    return await member_api.add_coupon(coupon_id)


# ============================== 售后 ==============================
@tool(domain="return", permission="write", roles={"cs"})
async def create_return_apply(order_id: int, product_id: int, reason: str) -> str:
    """申请退货。order_id: 订单编号; product_id: 商品编号; reason: 退货原因。"""
    data = await member_api.create_return_apply(order_id, product_id, reason)
    return f"退货申请已提交: {data}" if data else "退货申请提交失败"


# ============================== 客服知识库（RAG） ==============================
@tool(domain="rag", permission="read", roles={"cs", "guide", "supervisor"})
async def query_policy(question: str) -> str:
    """查询客服政策知识库（售后规则/退换政策/会员权益等），用于回答客服问题。question: 要查询的问题。"""
    return await get_rag_client().query_knowledge(question, hub="policy")


@tool(domain="rag", permission="read", roles={"cs", "guide", "supervisor"})
async def query_cs_knowledge(question: str) -> str:
    """查询客服知识库（常见问答/门店政策）。question: 要查询的问题。"""
    return await get_rag_client().query_knowledge(question, hub="cs")


@tool(domain="rag", permission="read", roles={"cs"})
async def verify_policy(claim: str) -> str:
    """核查一段政策陈述是否属实（基于售后规则知识库）。claim: 待核查的陈述。"""
    return await get_rag_client().verify_fact(claim, hub="policy")


# ============================== 注册 ==============================
# 说明：search_product 由 guide_tools 提供（roles 含 cs/supervisor），此处不重复注册
CS_TOOLS = [
    search_order, get_order_detail, cancel_order, confirm_receive_order,
    list_address, list_coupons, add_coupon, create_return_apply,
    query_policy, query_cs_knowledge, verify_policy,
]


def register_cs_tools(reg: Any = registry) -> None:
    reg.register_all(CS_TOOLS)
