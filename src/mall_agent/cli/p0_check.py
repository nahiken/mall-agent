"""P0 验证：settings / mock / llm / monitor 核心链路是否可用。

用法：python -m mall_agent.cli.p0_check
"""
from __future__ import annotations

import asyncio
import json


async def main() -> None:
    print("=" * 50)
    print("P0 核心链路自检")
    print("=" * 50)

    # 1. settings 加载
    from mall_agent.core.settings import settings

    print(f"\n[1/4] settings:")
    print(f"  llm: {settings.llm.get('base_url')} / {settings.llm.get('model')}")
    print(f"  mall: api_base={settings.mall.get('api_base')} mock={settings.mall.get('use_mock')}")
    print(f"  max_steps={settings.max_steps} default_scheme={settings.default_scheme}")

    # 2. mock 后端
    from mall_agent.mall_api.client import get_mall_client
    from mall_agent.mall_api import product as product_api

    client = get_mall_client()
    await client.ensure_session()
    print(f"\n[2/4] mall mock: token={client._token[:8]}... user_id={client._user_id}")
    results = await product_api.search_product("手机")
    print(f"  search '手机' -> {len(results)} 条: {[p.get('name') for p in results[:3]]}")

    # 3. LLM 客户端（流式 TTFT/TPOT 采集）
    from mall_agent.core.llm import LLMClient

    print("\n[3/4] LLM 流式调用（DeepSeek，可能需几秒）...")
    llm = LLMClient(caller="agent")
    collected = []
    async for tok in llm.astream([
        {"role": "system", "content": "你是一个测试助手，请用一句话回答。"},
        {"role": "user", "content": "你好"},
    ]):
        collected.append(tok)
    print(f"  回复: {''.join(collected)[:80]}")

    # 4. monitor 采集
    from mall_agent.monitoring.monitor import monitor

    monitor.begin_trial(harness="custom", agent_id="1", golden_task_id="p0-check", context_policy="full")
    collected2 = []
    async for tok in llm.astream([{"role": "user", "content": "再测试一次，回答收到即可"}]):
        collected2.append(tok)
    result = monitor.end_trial(steps=1)
    print(f"\n[4/4] monitor 采集:")
    print(f"  agent_in={result['cost']['agent_in']} agent_out={result['cost']['agent_out']}")
    print(f"  tool_calls={result['cost']['tool_calls']} cost_rmb={result['cost']['external_cost_rmb']}")
    print(f"  latency={result['latency_ms']}")

    print("\n✅ P0 核心链路全部通过")


if __name__ == "__main__":
    asyncio.run(main())
