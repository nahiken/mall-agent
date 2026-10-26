"""Skill 基座：Skill 声明 + 懒加载语义。

Skill = 一组工具名 + 意图触发描述。方案3 的主 Agent 通过意图路由选择 skill，
`registry.filter_by_skill(...)` 只加载命中 skill 的工具 —— 这是省 token 的机理。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Skill:
    name: str
    description: str                 # 用于 LLM 意图路由的描述
    tool_names: set[str] = field(default_factory=set)
    triggers: list[str] = field(default_factory=list)  # 规则路由的关键词

    def matches_rule(self, text: str) -> bool:
        """规则路由：关键词命中即选中。"""
        t = text.lower()
        return any(tg.lower() in t for tg in self.triggers)


# 预置两个 skill
def build_skills() -> dict[str, Skill]:
    cs_skill = Skill(
        name="cs",
        description="客服：处理订单查询/取消/退货、地址、优惠券、售后政策等客服类问题",
        tool_names={
            "search_order", "get_order_detail", "cancel_order", "confirm_receive_order",
            "list_address", "list_coupons", "add_coupon", "create_return_apply",
            "query_policy", "query_cs_knowledge", "verify_policy",
        },
        triggers=["订单", "取消", "退货", "退款", "地址", "物流", "客服", "优惠券", "售后", "收到"],
    )
    guide_skill = Skill(
        name="guide",
        description="导购：商品推荐、搜索、比价、加入购物车、库存查询等购物类问题",
        tool_names={
            "search_product", "get_product_detail", "get_category_tree", "check_stock",
            "list_cart", "add_to_cart", "update_cart_quantity",
            "list_coupons_by_product", "recommend_products",
        },
        triggers=["买", "推荐", "想买", "商品", "手机", "价格", "购物车", "下单", "哪个", "比价", "库存", "有什么"],
    )
    return {s.name: s for s in (cs_skill, guide_skill)}
