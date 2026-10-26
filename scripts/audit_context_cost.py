"""工具 schema 的 token / 暴露面复核（对外说明的数据口径）。

要核的点：
1. 「23 个工具按角色只暴露 20 个」——按 roles ∩ 集合算，是否真是 20，谁被排除。
2. 「schema 1665 → 952 / 713 token」——必须说明 token 是怎么数的。参考实现用
   tiktoken（cl100k/o200k）或 chars/4；本脚本两种都给，并给出 JSON 字符数，
   让读者能自己换算。

用法：python scripts/audit_context_cost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mall_agent.cli.eval import _register_tools  # noqa: E402
from mall_agent.tools import registry as R  # noqa: E402


def count_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return 0


def main() -> None:
    _register_tools()
    reg = R.registry
    all_specs = [reg.all[n] for n in sorted(reg.all)]
    full = [s.to_openai_schema() for s in all_specs]
    full_json = json.dumps(full, ensure_ascii=False)

    print(f"工具总数：{len(all_specs)}")
    print(f"全量 schema：{len(full_json)} 字符，tiktoken={count_tokens(full_json)} tokens")

    for role in ("cs", "guide", "supervisor", "admin"):
        specs = [s for s in all_specs if role in s.scope.roles]
        js = json.dumps([s.to_openai_schema() for s in specs], ensure_ascii=False)
        names = sorted(s.name for s in specs)
        print(f"\n角色 {role}: {len(specs)} 个工具，{len(js)} 字符，"
              f"tiktoken={count_tokens(js)} tokens")
        print(f"  工具：{names}")

    try:
        from mall_agent.agents.graph_factory import _build_executor
        for scheme in ("1", "2", "3"):
            ex = _build_executor({"role": "cs", "user_id": "1", "rag_enabled": True}, scheme, True)
            ns = getattr(ex, "registry", None) or getattr(ex, "tool_registry", None)
            if ns is not None and hasattr(ns, "all"):
                print(f"\n方案 {scheme} 装配后可见工具数：{len(ns.all)}")
    except Exception as e:  # noqa: BLE001
        print(f"\n（方案装配探测跳过：{type(e).__name__}: {e}）")


if __name__ == "__main__":
    main()
