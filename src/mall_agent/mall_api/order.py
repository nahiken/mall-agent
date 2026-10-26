"""订单域 API 封装（客服核心）。"""
from __future__ import annotations

from typing import Any

from .client import get_mall_client


async def list_orders(status: int | None = None, page_num: int = 1, page_size: int = 5) -> list[dict]:
    client = get_mall_client()
    params = {"pageNum": page_num, "pageSize": page_size}
    if status is not None:
        params["status"] = status
    r = await client._get("/order/list", params)
    data = r.get("data") or {}
    return data.get("list") or []


async def get_order_detail(order_id: int) -> dict | None:
    client = get_mall_client()
    r = await client._get(f"/order/detail/{order_id}")
    return r.get("data")


async def cancel_order(order_id: int) -> str:
    client = get_mall_client()
    r = await client._post("/order/cancelUserOrder", {"orderId": order_id})
    return str(r.get("message") or "已取消")


async def confirm_receive_order(order_id: int) -> str:
    client = get_mall_client()
    r = await client._post("/order/confirmReceiveOrder", {"orderId": order_id})
    return str(r.get("message") or "已确认收货")


async def generate_order(cart_item_ids: list[int], **order_param) -> dict:
    """下单。order_param 可含 totalAmount/note 等。"""
    client = get_mall_client()
    body = {"cartItemIds": cart_item_ids, **order_param}
    r = await client._post("/order/generateOrder", body)
    return r.get("data") or {}
