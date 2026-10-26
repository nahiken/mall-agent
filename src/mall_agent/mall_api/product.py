"""商品域 API 封装（导购核心）。"""
from __future__ import annotations

from typing import Any

from .client import get_mall_client


async def search_product(keyword: str = "", page_num: int = 1, page_size: int = 5) -> list[dict]:
    """按关键词搜索商品（mock/真实）。"""
    client = get_mall_client()
    r = await client._get("/product/search", {
        "keyword": keyword, "pageNum": page_num, "pageSize": page_size})
    data = r.get("data") or {}
    return data.get("list") or []


async def get_product_detail(product_id: int) -> dict | None:
    client = get_mall_client()
    r = await client._get(f"/product/detail/{product_id}")
    return r.get("data")


async def get_category_tree() -> list[dict]:
    client = get_mall_client()
    r = await client._get("/product/categoryTreeList")
    return r.get("data") or []


async def check_stock(product_id: int) -> int | None:
    """返回库存数；查询失败返回 None。"""
    p = await get_product_detail(product_id)
    if not p:
        return None
    return p.get("stock")
