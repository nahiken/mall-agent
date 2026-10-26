"""数据级权限探针：own_order / own_session 校验。

真实模式：直连 MySQL `mall` 库只读查询 oms_order 判断归属。
Mock 模式：查 MockMallBackend 的内存态。
（两种模式都保持"只读 + 快速"，是零改宿主策略的一部分。）
"""
from __future__ import annotations

from typing import Any

from ..core.types import PermissionEvent
from .model import Session, ToolScope


class MockDataProbe:
    """Mock 数据探针（配 MockMallBackend 使用）。"""

    def __init__(self, mock_backend: Any) -> None:
        self._mock = mock_backend

    def check(self, rule: Any, args: dict[str, Any], session: Session) -> tuple[bool, str]:
        if rule.check == "own_session":
            # 绑定 session 用户的操作（如 list_address / add_to_cart）无需额外校验
            return True, ""
        if rule.check == "own_order":
            order_id = args.get("order_id") or args.get("orderId")
            uid = session.user_id
            if order_id is None or uid is None:
                return False, "缺少 order_id 或 user_id"
            orders = self._mock._orders.get(int(uid), [])
            if any(o.get("id") == int(order_id) for o in orders):
                return True, ""
            return False, f"订单 {order_id} 不属于当前用户"
        return True, ""


class SQLDataProbe:
    """直连 MySQL 的只读探针（真实模式，P3 后启用）。"""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn  # mysql+pymysql://user:pass@host:3306/mall
        self._conn = None

    def check(self, rule: Any, args: dict[str, Any], session: Session) -> tuple[bool, str]:
        if rule.check == "own_order":
            order_id = args.get("order_id") or args.get("orderId")
            uid = session.user_id
            if order_id is None or uid is None:
                return False, "缺少 order_id 或 user_id"
            return self._check_order_owner(int(order_id), int(uid))
        if rule.check == "own_session":
            return True, ""
        return True, ""

    def _check_order_owner(self, order_id: int, uid: int) -> tuple[bool, str]:
        # 懒连接 + 只读查询
        if self._conn is None:
            self._conn = self._connect()
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT member_id FROM oms_order WHERE id=%s", (order_id,))
            row = cur.fetchone()
            if row and row[0] == uid:
                return True, ""
            return False, f"订单 {order_id} 不属于当前用户"
        except Exception as e:
            return False, f"订单归属查询失败: {e}"

    def _connect(self):
        import pymysql
        # 从 settings 读取 MySQL 连接
        from ..core.settings import settings
        m = settings.mall
        return pymysql.connect(
            host=m.get("mysql_host", "127.0.0.1"),
            port=int(m.get("mysql_port", 3306)),
            user=m.get("mysql_user", "root"),
            password=m.get("mysql_password", "root"),
            database=m.get("mysql_db", "mall"),
            charset="utf8mb4",
        )


def build_probe(mock_backend: Any = None, use_mock: bool = True):
    if use_mock and mock_backend is not None:
        return MockDataProbe(mock_backend)
    return SQLDataProbe()
