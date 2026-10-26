"""购物车 / 会员 / 售后 API 封装。"""
from __future__ import annotations

from typing import Any

from .client import get_mall_client


# ============================== 会员 ==============================
async def list_address() -> list[dict]:
    client = get_mall_client()
    r = await client._get("/member/address/list")
    return r.get("data") or []


async def list_coupons(use_status: int | None = None) -> list[dict]:
    client = get_mall_client()
    params = {}
    if use_status is not None:
        params["useStatus"] = use_status
    r = await client._get("/member/coupon/list", params)
    return r.get("data") or []


async def list_coupons_by_product(product_id: int) -> list[dict]:
    client = get_mall_client()
    r = await client._get(f"/member/coupon/listByProduct/{product_id}")
    return r.get("data") or []


async def add_coupon(coupon_id: int) -> str:
    client = get_mall_client()
    r = await client._post("/member/coupon/add", {"couponId": coupon_id})
    return str(r.get("message") or "领取成功")


# ============================== 售后 ==============================
async def create_return_apply(order_id: int, product_id: int, reason: str, **extra) -> dict:
    client = get_mall_client()
    body = {"orderId": order_id, "productId": product_id, "reason": reason, **extra}
    r = await client._post("/returnApply/create", body)
    return r.get("data") or {}
