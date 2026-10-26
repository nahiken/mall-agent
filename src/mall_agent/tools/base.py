"""工具声明基座：ToolSpec + @tool 装饰器 + 从函数签名生成 Pydantic args_schema。

用法：
    @tool(domain="order", permission="read", roles={"cs", "supervisor"})
    def search_order(status: int | None = None, page_num: int = 0, page_size: int = 5) -> str:
        \"\"\"查询当前用户订单列表。status: 0待付款/1待发货/2已发货/3已完成/4已关闭/5无效。\"\"\"
        ...
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from ..permission.model import ToolScope


class ToolSpec:
    """一个 Agent 可见工具的完整声明。"""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        domain: str,
        permission: str,
        roles: set[str],
        sensitive: bool = False,
        args_schema: Any = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.scope = ToolScope(domain=domain, permission=permission, roles=set(roles), sensitive=sensitive)
        self.args_schema = args_schema

    # ---- 转 OpenAI tool schema ----
    def to_openai_schema(self) -> dict[str, Any]:
        schema = self.args_schema.model_json_schema() if self.args_schema else {
            "type": "object",
            "properties": {},
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    # ---- 执行 ----
    async def invoke(self, **kwargs) -> str:
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return str(result)


# ------------------------------------------------------------------
# @tool 装饰器
# ------------------------------------------------------------------
def _build_args_schema(func: Callable):
    """从函数签名构建 Pydantic model。"""
    import pydantic

    sig = inspect.signature(func)
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "ctx", "session"):
            continue
        default = param.default
        if default is inspect.Parameter.empty:
            default = ...  # 必填
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        fields[name] = (ann, default)
    if not fields:
        return None
    # 用 create_model 动态生成
    return pydantic.create_model(f"{func.__name__}Args", **fields)


def tool(*, domain: str, permission: str, roles: set[str] | list[str], sensitive: bool = False,
         name: str | None = None):
    """装饰器：把普通异步/同步函数升级为 ToolSpec。

    name: 工具在 Agent 中显示的名字（默认用函数名）。同逻辑工具服务不同角色时可改名避免冲突。
    """

    def decorator(func: Callable) -> ToolSpec:
        doc = (func.__doc__ or "").strip()
        description = doc.split("\n")[0] if doc else func.__name__
        return ToolSpec(
            name=name or func.__name__,
            description=description,
            func=func,
            domain=domain,
            permission=permission,
            roles=set(roles),
            sensitive=sensitive,
            args_schema=_build_args_schema(func),
        )

    return decorator
