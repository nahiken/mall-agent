"""评测夹具：每个 trial 前把 Mock 业务状态重置为确定性初始态。

动机（评测方法论）：
- 真实用户态是空的（uid 由 sso 自增生成，没有订单/券/购物车），导致「查订单」
  「取消订单」类任务在环境层面无解，却被记成 agent 失败 → 归因错误。
- 状态在 trial 之间累积：上一题 add_to_cart 的商品会留在购物车，下一题的
  cart_items 断言可能"白捡"通过；反之 coupon_received 也可能被前一题喂饱。
  τ-bench 等 agent 基准的做法是**每题重置环境**，这里对齐该做法。

投放的订单刻意不含 status=4（已关闭）：probe 的 order_cancelled 断言是
"该用户存在任一 status==4 的订单"，若预置一条已关闭订单，该断言恒真。
"""
from __future__ import annotations

from typing import Any

# 订单 id 取 9001+，与 mall.sql 里的 12..62 段错开，便于在日志里一眼区分夹具数据
SEED_ORDERS: list[dict[str, Any]] = [
    {"id": 9001, "order_sn": "FIXTURE202401010001", "status": 1,
     "total_amount": 3788.0, "create_time": "2024-01-01 10:00:00",
     "receiver_name": "测试用户", "receiver_province": "广东省", "receiver_city": "深圳市"},
    {"id": 9002, "order_sn": "FIXTURE202401020002", "status": 2,
     "total_amount": 5499.0, "create_time": "2024-01-02 11:00:00",
     "receiver_name": "测试用户", "receiver_province": "广东省", "receiver_city": "深圳市"},
    {"id": 9003, "order_sn": "FIXTURE202401030003", "status": 3,
     "total_amount": 2499.0, "create_time": "2024-01-03 12:00:00",
     "receiver_name": "测试用户", "receiver_province": "广东省", "receiver_city": "深圳市"},
    {"id": 9004, "order_sn": "FIXTURE202401040004", "status": 1,
     "total_amount": 199.0, "create_time": "2024-01-04 13:00:00",
     "receiver_name": "测试用户", "receiver_province": "广东省", "receiver_city": "深圳市"},
]


def reset_mock_state(mock: Any, uid: int | None = None) -> int:
    """清空该用户的购物车/优惠券，并把订单重置为固定三条。返回 uid。"""
    uid = int(uid if uid is not None else mock.current_uid())
    mock._carts[uid] = []
    mock._coupons[uid] = []
    mock._orders[uid] = [dict(o, member_id=uid) for o in SEED_ORDERS]
    # 订单自增 id 从夹具之后开始，避免 /order/generateOrder 撞号
    max_id = max(o["id"] for o in SEED_ORDERS)
    mock._next_ids["oms_order"] = max_id
    return uid
