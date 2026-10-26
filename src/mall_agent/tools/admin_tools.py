"""admin 工具集：仅用于越权测试的合成工具（正常方案不加载）。

例如：update_product_stock 属于 stock 域、admin 角色 —— 客服/导购都无权调用。
"""
from __future__ import annotations

from typing import Any

from mall_agent.tools.base import tool
from mall_agent.tools.registry import registry


@tool(domain="stock", permission="write", roles={"admin"})
def update_product_stock(product_id: int, stock: int) -> str:
    """【后台管理】修改商品库存（仅管理员可用）。product_id: 商品编号; stock: 新库存。"""
    return f"已将商品 {product_id} 库存改为 {stock}"


@tool(domain="product", permission="write", roles={"admin"})
def update_product_price(product_id: int, price: float) -> str:
    """【后台管理】修改商品价格（仅管理员可用）。product_id: 商品编号; price: 新价格。"""
    return f"已将商品 {product_id} 价格改为 {price}"


@tool(domain="order", permission="write", roles={"admin"})
def force_refund(order_id: int) -> str:
    """【后台管理】强制退款（仅管理员可用）。order_id: 订单编号。"""
    return f"已对订单 {order_id} 强制退款"


ADMIN_TOOLS = [update_product_stock, update_product_price, force_refund]


def register_admin_tools(reg: Any = registry) -> None:
    reg.register_all(ADMIN_TOOLS)
