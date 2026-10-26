"""导购工具集（guide_tools）：商品 / 购物车 / 优惠券 / 商品知识库。

所有工具均为 async def（ToolExecutor 会 await spec.invoke()）。
"""
from __future__ import annotations

from typing import Any

from mall_agent.mall_api import cart as cart_api
from mall_agent.mall_api import member as member_api
from mall_agent.mall_api import product as product_api
from mall_agent.rag.mcp_client import get_rag_client
from mall_agent.tools.base import tool
from mall_agent.tools.registry import registry


# ============================== 商品域 ==============================
@tool(domain="product", permission="read", roles={"guide", "cs", "supervisor"})
async def search_product(keyword: str = "", page_num: int = 1, page_size: int = 5) -> str:
    """搜索商品。keyword: 商品关键词（如 手机、冰箱）。"""
    products = await product_api.search_product(keyword, page_num, page_size)
    if not products:
        return "没有找到相关商品"
    return "\n".join(f"{p.get('id')}. {p.get('name')} ¥{p.get('price')} 库存:{p.get('stock')}"
                     for p in products)


@tool(domain="product", permission="read", roles={"guide", "supervisor"})
async def get_product_detail(product_id: int) -> str:
    """查看商品详情。product_id: 商品编号。"""
    p = await product_api.get_product_detail(product_id)
    if not p:
        return f"未找到商品 {product_id}"
    return (f"商品#{p.get('id')} {p.get('name')} ¥{p.get('price')} "
            f"库存:{p.get('stock')} 副标题:{p.get('sub_title', '')}")


@tool(domain="product", permission="read", roles={"guide"})
async def get_category_tree() -> str:
    """获取商品分类树。"""
    cats = await product_api.get_category_tree()
    if not cats:
        return "暂无分类"
    return "\n".join(f"{c.get('id')}. {c.get('name')}" for c in cats)


@tool(domain="product", permission="read", roles={"guide"})
async def check_stock(product_id: int) -> str:
    """检查商品库存。product_id: 商品编号。"""
    stock = await product_api.check_stock(product_id)
    return f"商品 {product_id} 库存: {stock if stock is not None else '未知'}"


# ============================== 购物车域 ==============================
@tool(domain="cart", permission="read", roles={"guide"})
async def list_cart() -> str:
    """查看当前用户的购物车。"""
    items = await cart_api.list_cart()
    if not items:
        return "购物车为空"
    return "\n".join(f"购物车项#{i.get('id')} 商品:{i.get('productId')} 数量:{i.get('quantity')}"
                     for i in items)


@tool(domain="cart", permission="write", roles={"guide"})
async def add_to_cart(product_id: int, quantity: int = 1) -> str:
    """把商品加入购物车。product_id: 商品编号; quantity: 数量。"""
    data = await cart_api.add_to_cart(product_id, quantity)
    return f"已加入购物车: {data}" if data else "加入购物车失败"


@tool(domain="cart", permission="write", roles={"guide"})
async def update_cart_quantity(cart_item_id: int, quantity: int) -> str:
    """修改购物车项数量。cart_item_id: 购物车项编号; quantity: 新数量。"""
    data = await cart_api.update_cart_quantity(cart_item_id, quantity)
    return f"已更新数量: {data}" if data else "更新失败"


# ============================== 优惠券 ==============================
@tool(domain="coupon", permission="read", roles={"guide"})
async def list_coupons_by_product(product_id: int) -> str:
    """查询某商品可用的优惠券。product_id: 商品编号。"""
    coupons = await member_api.list_coupons_by_product(product_id)
    if not coupons:
        return "该商品暂无可用优惠券"
    return "\n".join(f"{c.get('id')}. {c.get('name')} 满{c.get('min_point')}减{c.get('amount')}"
                     for c in coupons)


# ============================== 商品知识库（RAG 导购） ==============================
@tool(domain="rag", permission="read", roles={"guide"})
async def recommend_products(question: str) -> str:
    """基于商品知识库做语义检索，返回相关商品信息（用于推荐）。question: 用户的需求描述。"""
    return await get_rag_client().query_knowledge(question, hub="guide")


# ============================== 注册 ==============================
GUIDE_TOOLS = [
    search_product, get_product_detail, get_category_tree, check_stock,
    list_cart, add_to_cart, update_cart_quantity,
    list_coupons_by_product, recommend_products,
]


def register_guide_tools(reg: Any = registry) -> None:
    reg.register_all(GUIDE_TOOLS)
