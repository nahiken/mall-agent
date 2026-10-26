"""权限模型：Domain / Permission / ToolScope / AgentRole / PermissionGuard。

三层检查：
1. 工具级：tool.scope.roles ∩ session.role 非空
2. 数据级：own_order / own_session 校验（防跨用户）
3. 敏感级：sensitive 操作需显式 consent

设计成纯逻辑（不依赖 mall_api），便于单测与越权实验。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..core.settings import ROOT, settings
from ..core.types import PermissionEvent

PERM_LEVELS = ["read", "operate", "write"]


class ToolScope:
    """工具的作用域声明。"""

    def __init__(self, domain: str, permission: str, roles: set[str], sensitive: bool = False):
        self.domain = domain
        self.permission = permission
        self.roles = set(roles)
        self.sensitive = sensitive

    def allows(self, role: str, required_permission: str | None = None) -> bool:
        """工具级检查：角色是否在允许集合内。"""
        if role not in self.roles:
            return False
        if required_permission and required_permission not in PERM_LEVELS:
            return False
        return True


class Session:
    """一次 Agent 会话的执行上下文（guard 检查用）。"""

    def __init__(self, role: str, user_id: str | None = None, consent: str = "implicit"):
        self.role = role
        self.user_id = user_id
        self.consent = consent  # "implicit" | "explicit"


class DataRule:
    """数据级校验规则。"""

    def __init__(self, check: str, sensitive: bool = False, field: str | None = None):
        self.check = check       # "own_order" | "own_session" | ...
        self.sensitive = sensitive
        self.field = field


class PermissionGuard:
    """权限守卫：校验一次工具调用是否被允许。

    构造时传入一个 `data_probe` 回调，用于数据级校验：
        data_probe(check, args, session) -> (allowed: bool, reason: str)
    在真实环境里，data_probe 会查 MySQL/mock 判断订单归属等。
    """

    def __init__(self, role_policies: dict[str, dict[str, list[str]]] | None = None,
                 data_rules: dict[str, list[dict]] | None = None,
                 data_probe: Any = None):
        self.role_policies = role_policies or {}
        self.data_rules = data_rules or {}
        self.data_probe = data_probe

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "PermissionGuard":
        p = Path(path) if path else ROOT / "config" / "permissions.yaml"
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        role_policies = {r: {d: perm for d, perm in domains.items()} for r, domains in cfg.get("roles", {}).items()}
        data_rules = cfg.get("data_rules", {})
        return cls(role_policies=role_policies, data_rules=data_rules)

    # ------------------------------------------------------------------
    # 角色权限查询
    # ------------------------------------------------------------------
    def role_allows(self, role: str, domain: str, permission: str) -> bool:
        """角色级检查：role 是否有 domain.permission。"""
        allowed = self.role_policies.get(role, {}).get(domain, [])
        if not allowed:
            return False
        return PERM_LEVELS.index(permission) <= max(PERM_LEVELS.index(a) for a in allowed)

    def tool_permission_ok(self, spec: ToolScope, role: str) -> bool:
        """工具级：spec.roles 是否含 role，且 role 拥有 domain.permission。"""
        if role not in spec.roles:
            return False
        return self.role_allows(role, spec.domain, spec.permission)

    # ------------------------------------------------------------------
    # 核心检查
    # ------------------------------------------------------------------
    def check(self, spec: ToolScope, args: dict[str, Any], session: Session,
              tool_name: str | None = None) -> tuple[bool, PermissionEvent | None]:
        """返回 (是否放行, 权限事件)。tool_name 用于查 data_rules（按工具名组织）。"""
        total = PermissionEvent(tool=spec.domain, role=session.role, decision="allow", reason="")

        # 1. 工具级
        if not self.tool_permission_ok(spec, session.role):
            total = PermissionEvent(tool=spec.domain, role=session.role, decision="block_tool",
                                    reason=f"角色 {session.role} 无权调用 {spec.domain}")
            return False, total

        # 2. 数据级（data_rules 按工具名组织）
        name = tool_name or spec.domain
        for rule in self.data_rules.get(name, []):
            dr = DataRule(**rule)
            if self.data_probe is not None:
                allowed, reason = self.data_probe.check(dr, args, session)
                if not allowed:
                    total = PermissionEvent(tool=name, role=session.role, decision="block_data",
                                            reason=reason or f"数据级校验失败: {dr.check}")
                    return False, total

        # 3. 敏感级
        if spec.sensitive and session.consent != "explicit":
            total = PermissionEvent(tool=spec.domain, role=session.role, decision="block_sensitive",
                                    reason=f"敏感操作 {spec.domain} 需要用户显式确认")
            return False, total

        return True, total


def build_default_guard(data_probe: Any = None) -> PermissionGuard:
    """加载 config/permissions.yaml 的默认守卫，并注入数据探针。"""
    guard = PermissionGuard.from_yaml()
    guard.data_probe = data_probe
    return guard
