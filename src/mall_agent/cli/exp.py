"""CLI：三组实验入口。

用法：
    python -m mall_agent.cli.exp --context --limit 6
    python -m mall_agent.cli.exp --permission
    python -m mall_agent.cli.exp --skill --limit 6
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="mall-agent 实验 CLI")
    parser.add_argument("--context", action="store_true", help="上下文管理实验")
    parser.add_argument("--permission", action="store_true", help="权限管理实验")
    parser.add_argument("--skill", action="store_true", help="Skill懒加载实验")
    parser.add_argument("--policy", default="full,sliding,summarize,hier",
                        help="上下文实验的策略列表（逗号分隔）")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.context:
        from mall_agent.experiments.exp_context import run
        asyncio.run(run(args.policy.split(","), args.limit))
    elif args.permission:
        from mall_agent.experiments.exp_permission import run
        asyncio.run(run())
    elif args.skill:
        from mall_agent.experiments.exp_skill import run
        asyncio.run(run(args.limit))
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main())
