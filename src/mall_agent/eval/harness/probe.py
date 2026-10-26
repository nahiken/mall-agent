"""业务状态探针：断言 expected_state 是否达成（零改宿主，Mock 只读）。

真实模式：直连 MySQL 只读（probe.py 里 SQLDataProbe 的思路扩展）。
Mock 模式：查 MockMallBackend 内存态。
"""
from __future__ import annotations

from typing import Any

from ..golden.schema import ExpectedState


class MockStateProbe:
    """Mock 状态探针：对 MockMallBackend 做只读断言。"""

    def __init__(self, mock_backend: Any, user_id: str | int = "1"):
        self._mock = mock_backend
        # 默认 user_id="1" 与 mock 真实会话用户不一致（ensure_user 后是自增 id）。
        # 统一解析到 mock 的当前会话用户，避免断言错查他人数据。
        if str(user_id) in ("1", "", "None"):
            self._uid = mock_backend.current_uid()
        else:
            self._uid = int(user_id) if str(user_id).isdigit() else mock_backend.current_uid()
        # 期望领取的优惠券 id：任务可在 expected_state 里带 coupon_id；
        # 默认 0 表示"按该用户是否领取过任意券判定"。
        self._expect_coupon_id = 0

    def set_expected_coupon(self, coupon_id: int) -> None:
        """为 coupon_received 断言指定具体优惠券 id（任务级别可覆盖）。"""
        self._expect_coupon_id = coupon_id

    def reset(self) -> int:
        """把该用户态重置为确定性初始态（每题开跑前调用，防跨题状态泄漏）。"""
        from .fixtures import reset_mock_state
        return reset_mock_state(self._mock, self._uid)

    async def check(self, expected: ExpectedState) -> tuple[bool, str]:
        """检查期望业务状态。返回 (是否达成, 原因)。"""
        field_results = await self.fields(expected)
        ok = all(field_results.values()) if field_results else True
        reasons: list[str] = []
        failed = [k for k, v in field_results.items() if not v]
        if failed:
            reasons.append(f"未达成: {', '.join(failed)}")
        return ok, "；".join(reasons) if reasons else ""

    async def partial_check(self, expected: ExpectedState) -> dict[str, bool]:
        """逐字段断言，返回 {字段: 是否达成}。供功能正确性分（functional_correct）使用。"""
        return await self.fields(expected)

    async def fields(self, expected: ExpectedState) -> dict[str, bool]:
        """对每个非 None 的 expected_state 字段做断言，返回逐字段结果。"""
        field_results: dict[str, bool] = {}

        if expected.order_created is not None:
            has_order = len(self._mock._orders.get(self._uid, [])) > 0
            field_results["order_created"] = (has_order == expected.order_created)
        if expected.order_cancelled is not None:
            cancelled = any(o.get("status") == 4 for o in self._mock._orders.get(self._uid, []))
            field_results["order_cancelled"] = (cancelled == expected.order_cancelled)
        if expected.cart_items is not None:
            cart = self._mock._carts.get(self._uid, [])
            cart_map = {}
            for item in cart:
                pid = str(item.get("productId"))
                cart_map[pid] = cart_map.get(pid, 0) + int(item.get("quantity", 0))
            field_results["cart_items"] = all(
                cart_map.get(pid) == qty for pid, qty in expected.cart_items.items())
        if expected.order_cancelled_id is not None:
            oid = int(expected.order_cancelled_id)
            status = next((o.get("status") for o in self._mock._orders.get(self._uid, [])
                           if o.get("id") == oid), None)
            field_results["order_cancelled_id"] = (status == 4)
        if expected.coupon_id is not None:
            field_results["coupon_id"] = self._mock.has_coupon(self._uid, int(expected.coupon_id))
        if expected.coupon_received is not None:
            if self._mock.has_coupon(self._uid, self._expect_coupon_id):
                has_coupon = True
            else:
                has_coupon = len(self._mock._coupons.get(self._uid, [])) > 0
            field_results["coupon_received"] = (has_coupon == expected.coupon_received)

        return field_results


class DummyProbe:
    """空探针：不做状态断言（用于只有轨迹判定的任务）。"""

    async def check(self, expected: ExpectedState) -> tuple[bool, str]:
        return True, ""


def has_assertions(expected: Any) -> bool:
    """该任务的 expected_state 是否含至少一个非 None 字段。

    空断言（全 None）时 probe.check 返回 all({}) == True —— 这是「默认通过」的
    根因。runner 用它区分「真断言」与「空断言」：空断言任务不该在第一轮就 break，
    否则多轮/澄清行为永远不会被评测到。
    """
    if expected is None:
        return False
    try:
        data = expected.model_dump()
    except AttributeError:
        data = dict(vars(expected))
    return any(v is not None for v in data.values())
