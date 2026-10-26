"""LLM 客户端：OpenAI 兼容接口 + 流式时延采集（TTFT/TPOT）+ usage 上报。

关键点：
- 用 stream=True 流式调用，逐 chunk 记录首 token 时延(TTFT)与 token 间隔(TPOT)。
- 每个调用带 caller 标签（agent/user_sim/judge/rag），monitor 据此分桶计成本。
- 所有时延/token 事件通过 tracing 上报，本类不直接写文件。
- 调用级重试：对 429/5xx/网络/超时做指数退避重试（可配置），评测与交互自动受益。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from .settings import settings
from .tracing import LLMEvent, tracing

logger = logging.getLogger(__name__)

# 中文 LLM 的 token 估算：近似 1 汉字 ≈ 0.6~1 token，这里取 1 字符 ≈ 0.7 token
_CHARS_PER_TOKEN = 1 / 0.7


def estimate_tokens(text: str) -> int:
    """估算字符串 token 数（无 tiktoken 时的近似值，用于工具 schema/上下文统计）。"""
    if not text:
        return 0
    # 粗略：英文按 4 字符/token，中文按 1.4 字符/token
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    others = len(text) - cjk
    return max(1, round(cjk * 0.9 + others / 4))


class LLMClient:
    """OpenAI 兼容流式客户端。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        caller: str = "agent",
        disable_thinking: bool | None = None,
    ) -> None:
        self.model = model or settings.env("LLM_MODEL", settings.llm.get("model", "deepseek-chat"))
        self.base_url = base_url or settings.env("LLM_BASE_URL", settings.llm.get("base_url"))
        self.api_key = api_key or settings.env("LLM_API_KEY", "sk-none")
        self.caller = caller
        # 是否关 DeepSeek 思考模式：显式参数 > env LLM_DISABLE_THINKING > 默认关（测的是 RAG 事实增益，
        # 非隐藏推理，且开思考会烧大量 reasoning token 抬高成本/时延；模型名含 -thinking/-r1 除外）
        if disable_thinking is None:
            disable_thinking = settings.env("LLM_DISABLE_THINKING", "1") not in ("", "0", "false", "False")
        if "-thinking" in self.model or "-r1" in self.model or "reasoner" in self.model:
            disable_thinking = False  # 推理专用模型不注入禁用（可能不支持/语义相反）
        self.disable_thinking = bool(disable_thinking)
        self.temperature = float(settings.llm.get("temperature", 0.0))
        self.max_tokens = int(settings.llm.get("max_tokens", 2048))
        # 调用级重试配置（默认 3 次尝试、退避基值 0.5s）
        self.retry_attempts = max(1, int(settings.llm.get("retry_attempts", 3)))
        self.retry_backoff = max(0.0, float(settings.llm.get("retry_backoff", 0.5)))
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,
        )

    # ------------------------------------------------------------------
    # 调用级重试：指数退避，仅重试可恢复的异常（429/5xx/网络/超时）
    # ------------------------------------------------------------------
    async def _retryable_create(self, **kwargs: Any) -> Any:
        """包一层 chat.completions.create，指数退避重试。

        返回已建立的流式响应（首次成功后即返回，流式读取阶段不再重试——
        一旦开始消费流，中途断流无法安全重试，直接交给调用方处理）。
        """
        attempts = 0
        if self.disable_thinking and "extra_body" not in kwargs:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        while True:
            attempts += 1
            try:
                return await self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError) as e:
                # 5xx/429 重试；4xx（除 429）为不可恢复的客户端错误，不重试
                if isinstance(e, APIStatusError) and e.status_code not in (429, 500, 502, 503, 504):
                    raise
                if attempts >= self.retry_attempts:
                    raise
                delay = self.retry_backoff * (2 ** (attempts - 1))
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次，%s），%.1fs 后重试", attempts, self.retry_attempts, type(e).__name__, delay
                )
                await asyncio.sleep(delay)
            except Exception:
                # 其它异常（参数错误/未知）不重试，直接抛
                raise

    async def _consume_with_retry(self, kwargs: dict[str, Any],
                                  consume: Any, reset: Any = None) -> bool:
        """流式消费 + 中途断流重试。

        消费回调 `consume(chunk)` 每次消费一个 chunk；`reset()` 在每次重试
        开始前清空累积（否则断流前已消费的 chunk 会在重试时重复累积）。
        返回 True 表示成功消费完整流；重试耗尽时抛异常。
        """
        attempts = 0
        while True:
            attempts += 1
            try:
                stream = await self._retryable_create(**kwargs)
                async for chunk in stream:
                    consume(chunk)
                return True
            except (APIConnectionError, APITimeoutError) as e:
                # 仅中途断流（非建连失败，建连已由 _retryable_create 兜底）重试
                if attempts >= self.retry_attempts:
                    logger.error("LLM 流式读取中断（第 %d/%d 次）仍失败: %s", attempts, self.retry_attempts, e)
                    raise
                if reset is not None:
                    reset()
                delay = self.retry_backoff * (2 ** (attempts - 1))
                logger.warning(
                    "LLM 流式读取中断（第 %d/%d 次，%s），%.1fs 后重建重试", attempts, self.retry_attempts, type(e).__name__, delay
                )
                await asyncio.sleep(delay)
            except Exception as e:
                # httpx 底层流错误（RemoteProtocolError/StreamClosed 等）同样视为断流重试
                import httpx
                if isinstance(e, httpx.TransportError) and attempts < self.retry_attempts:
                    if reset is not None:
                        reset()
                    delay = self.retry_backoff * (2 ** (attempts - 1))
                    logger.warning(
                        "LLM 流式传输错误（第 %d/%d 次，%s），%.1fs 后重建重试", attempts, self.retry_attempts, type(e).__name__, delay
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    # ------------------------------------------------------------------
    # 核心：流式 chat（逐 token 产出，采集 TTFT/TPOT）
    # ------------------------------------------------------------------
    async def astream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncIterator[str]:
        """流式调用。yield 文本片段；调用结束自动上报 LLMEvent。"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        t_start = time.perf_counter()
        t_first: float | None = None
        n_chunks = 0
        input_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        if tools:
            input_tokens += sum(estimate_tokens(str(t)) for t in tools)
        output_tokens = 0

        async for chunk in await self._retryable_create(**kwargs):
            n_chunks += 1
            if t_first is None and chunk.choices:
                t_first = time.perf_counter()
            delta = chunk.choices[0].delta if chunk.choices else None
            content = delta.content if delta else None
            if content:
                output_tokens += estimate_tokens(content)
                yield content
            # 累积 tool_calls 片段（非流式时用）
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn and fn.arguments:
                        output_tokens += len(fn.arguments) // 4

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000
        ttft_ms = (t_first - t_start) * 1000 if t_first else None
        tpot_ms = None
        if n_chunks > 1 and ttft_ms is not None:
            tpot_ms = (total_ms - ttft_ms) / max(1, n_chunks - 1)

        self._emit_usage(input_tokens, output_tokens, ttft_ms, tpot_ms, total_ms)

    # ------------------------------------------------------------------
    # 非流式（简单问答/路由/用户模拟/摘要）
    # ------------------------------------------------------------------
    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str:
        """非流式调用，返回文本内容。工具调用场景走 acomplete_tools。"""
        parts: list[str] = []
        async for part in self.astream(messages, tools, tool_choice):
            parts.append(part)
        return "".join(parts)

    async def acomplete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """流式调用 + 工具调用聚合，返回 (文本, tool_calls)。

        流式模式可同时采集 TTFT/TPOT（评测需要），并聚合 tool_calls（agent 需要）。
        tool_calls 结构：[{"name":..., "arguments":{...}, "id":...}]
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        t_start = time.perf_counter()
        t_first: float | None = None
        n_chunks = 0
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}   # index -> {id, name, args}
        input_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        if tools:
            input_tokens += sum(estimate_tokens(str(t)) for t in tools)
        output_tokens = 0

        # 流式读取 + 中途断流重试（消费回调幂等：content/tool_calls 累加）
        def _reset() -> None:
            nonlocal n_chunks, t_first, output_tokens, content_parts, tool_calls_acc
            content_parts.clear()
            tool_calls_acc.clear()
            n_chunks = 0
            t_first = None
            output_tokens = 0

        def _consume(chunk: Any) -> None:
            nonlocal n_chunks, t_first, output_tokens, input_tokens
            n_chunks += 1
            if t_first is None and chunk.choices:
                t_first = time.perf_counter()
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta:
                delta = choice.delta
                if delta.content:
                    content_parts.append(delta.content)
                    output_tokens += estimate_tokens(delta.content)
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index
                        acc = tool_calls_acc.setdefault(idx, {"id": tc.id or "", "name": "", "arguments": ""})
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] += tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments
                                output_tokens += len(tc.function.arguments) // 4
            if chunk.usage:
                u = chunk.usage
                if u.prompt_tokens:
                    input_tokens = u.prompt_tokens
                if u.completion_tokens:
                    output_tokens = u.completion_tokens

        await self._consume_with_retry(kwargs, _consume, reset=_reset)

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000
        ttft_ms = (t_first - t_start) * 1000 if t_first else None
        tpot_ms = None
        if n_chunks > 1 and ttft_ms is not None:
            tpot_ms = (total_ms - ttft_ms) / max(1, n_chunks - 1)
        self._emit_usage(input_tokens, output_tokens, ttft_ms, tpot_ms, total_ms)

        content = "".join(content_parts)
        tool_calls: list[dict[str, Any]] = []
        for acc in tool_calls_acc.values():
            try:
                args = json.loads(acc["arguments"] or "{}")
            except Exception:
                args = {}
            tool_calls.append({"name": acc["name"], "arguments": args, "id": acc["id"]})
        return content, tool_calls

    def _emit_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        ttft_ms: float | None,
        tpot_ms: float | None,
        total_ms: float,
    ) -> None:
        cost_in, cost_out = settings.cost_rates
        cost_rmb = (input_tokens / 1000 * cost_in) + (output_tokens / 1000 * cost_out)
        tracing.emit(
            LLMEvent(
                caller=self.caller,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                total_ms=total_ms,
                cost_rmb=cost_rmb,
            )
        )


def make_llm(caller: str = "agent", disable_thinking: bool | None = None) -> LLMClient:
    """按 caller 返回 LLM 客户端；user_sim/judge 可用更便宜的模型。"""
    if caller in ("user_sim", "judge") and settings.env("USER_SIM_MODEL"):
        return LLMClient(
            model=settings.env("USER_SIM_MODEL"),
            base_url=settings.env("USER_SIM_BASE_URL") or settings.env("LLM_BASE_URL"),
            api_key=settings.env("USER_SIM_API_KEY") or settings.env("LLM_API_KEY"),
            caller=caller,
            disable_thinking=disable_thinking,
        )
    return LLMClient(caller=caller, disable_thinking=disable_thinking)
