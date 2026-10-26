"""购物车域 API 封装（导购用）。"""
from __future__ import annotations

from typing import Any

from .client import get_mall_client


async def list_cart() -> list[dict]:
    client = get_mall_client()
    r = await client._get("/cart/list")
    return r.get("data") or []


async def add_to_cart(product_id: int, quantity: int = 1) -> dict:
    client = get_mall_client()
    r = await client._post("/cart/add", {"productId": product_id, "quantity": quantity})
    return r.get("data") or {}


async def update_cart_quantity(cart_item_id: int, quantity: int) -> dict:
    client = get_mall_client()
    r = await client._post("/cart/update/quantity", {"id": cart_item_id, "quantity": quantity})
    return r.get("data") or {}
