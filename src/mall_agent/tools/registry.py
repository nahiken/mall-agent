"""工具注册表：注册 / 按角色与 skill 过滤 / 生成 OpenAI tool schema。

懒加载语义：`registry.filter_for(active_skill)` 只返回当前 skill 的工具，
这是方案3省 token 的机理（LLM 的 tools 参数越小，输入 token 越少）。
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from .base import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"重复注册工具: {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def register_all(self, specs: Iterable[ToolSpec]) -> None:
        for s in specs:
            self.register(s)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def all(self) -> dict[str, ToolSpec]:
        return self._tools

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def filter_by_roles(self, roles: set[str]) -> "ToolRegistry":
        """返回新 registry：只含角色可见的工具（工具级权限的预过滤）。"""
        reg = ToolRegistry()
        for name, spec in self._tools.items():
            if spec.scope.roles & roles:
                reg.register(spec)
        return reg

    def filter_by_skill(self, skill_name: str, skill_tool_names: set[str]) -> "ToolRegistry":
        """返回新 registry：只含指定 skill 命中的工具。"""
        reg = ToolRegistry()
        for name, spec in self._tools.items():
            if name in skill_tool_names:
                reg.register(spec)
        return reg

    def filter_by_excluding_domain(self, domains: set[str]) -> "ToolRegistry":
        """返回新 registry：剔除指定 domain 的工具（RAG OFF 用——屏蔽 query_* 检索工具）。"""
        reg = ToolRegistry()
        for name, spec in self._tools.items():
            if spec.scope.domain not in domains:
                reg.register(spec)
        return reg

    # ------------------------------------------------------------------
    # 转 OpenAI schema
    # ------------------------------------------------------------------
    def openai_schemas(self) -> list[dict[str, Any]]:
        return [s.to_openai_schema() for s in self._tools.values()]

    def schema_token_estimate(self) -> int:
        """估算全部工具 schema 的 token（用于 skill 懒加载收益统计）。"""
        from ..core.llm import estimate_tokens
        return sum(estimate_tokens(json.dumps(s.to_openai_schema(), ensure_ascii=False)) for s in self._tools.values())


import json  # noqa: E402


# 全局注册表
registry = ToolRegistry()
