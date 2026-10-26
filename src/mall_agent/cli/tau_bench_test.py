"""真正接入 τ-Bench：用官方 τ-Bench 的 retail 环境测试 mall-agent 的 agent。

依赖说明：
- tau_bench 需 `pip install -e` 到当前 venv（指向本机 tau-bench 仓库路径）
- LLM 密钥由 mall-agent/.env 提供（dotenv 加载，不在本文件硬编码任何密钥）
- litellm 通过 openai 兼容端点访问 DeepSeek（环境变量由 .env 注入）

用法（在 mall-agent venv 里，需 PYTHONPATH=src）：
    python -m mall_agent.cli.tau_bench_test --start 0 --end 2 --num-trials 1
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 加载 mall-agent/.env（含模型密钥），不在此文件写入任何密钥
ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")


def _inject_env() -> None:
    """把 .env 的模型配置映射到 litellm 需要的环境变量（仅读取，不硬编码）。"""
    import mall_agent.core.settings as s_mod
    s = s_mod.settings
    base = s.env("LLM_BASE_URL")
    key = s.env("LLM_API_KEY")
    if base:
        os.environ.setdefault("OPENAI_API_BASE", base)
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)


def run_tau_retail(start: int, end: int, num_trials: int, model: str) -> None:
    """跑 τ-Bench 官方 retail 环境（DeepSeek agent + DeepSeek user sim）。"""
    _inject_env()
    from tau_bench.types import RunConfig
    from tau_bench.run import run as tau_run

    config = RunConfig(
        env="retail",
        agent_strategy="tool-calling",
        model=model,
        model_provider="openai",
        user_model=model,
        user_model_provider="openai",
        user_strategy="llm",
        task_split="dev",
        start_index=start,
        end_index=end if end >= 0 else 20,
        num_trials=num_trials,
    )
    results = tau_run(config)
    print(f"\n=== τ-Bench retail 结果（{len(results)} trials）===")
    for r in results:
        print(f"  task#{getattr(r, 'task_id', '?')}: reward={r.reward} done={r.done} "
              f"actions={len(r.actions)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()
    run_tau_retail(args.start, args.end, args.num_trials, args.model)


if __name__ == "__main__":
    sys.exit(main())
