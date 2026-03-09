"""API Adapter — direct SDK calls to AI providers.

Supports: Anthropic, OpenAI, and any OpenAI-compatible API (e.g., packyapi).
"""

import logging
from collections.abc import AsyncIterator

import anthropic
import openai

from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    StreamChunk,
    UsageInfo,
)

logger = logging.getLogger(__name__)


class ApiAdapter(BaseAdapter):
    """Adapter for direct API calls to AI providers."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._provider = config.get("provider", "openai")
        self._model = config.get("model", "")
        self._api_key = config.get("api_key", "")
        self._base_url = config.get("base_url")
        self._system_prompt = config.get("system_prompt", "")
        self._anthropic_client: anthropic.AsyncAnthropic | None = None
        self._openai_client: openai.AsyncOpenAI | None = None

    async def connect(self) -> None:
        if self._provider == "anthropic":
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=self._api_key)
        else:
            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._openai_client = openai.AsyncOpenAI(**kwargs)

    async def disconnect(self) -> None:
        if self._anthropic_client:
            await self._anthropic_client.close()
            self._anthropic_client = None
        if self._openai_client:
            await self._openai_client.close()
            self._openai_client = None

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        if self._provider == "anthropic":
            return await self._send_anthropic(messages)
        return await self._send_openai(messages)

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        if self._provider == "anthropic":
            async for chunk in self._stream_anthropic(messages):
                yield chunk
        else:
            async for chunk in self._stream_openai(messages):
                yield chunk

    async def health_check(self) -> bool:
        try:
            if self._provider == "anthropic":
                return self._anthropic_client is not None
            return self._openai_client is not None
        except Exception:
            return False

    @property
    def adapter_type(self) -> str:
        return "api"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L1_RICH

    # ── Anthropic ──

    def _to_anthropic_messages(
        self, messages: list[AdapterMessage]
    ) -> tuple[list[dict], str | None]:
        """Convert messages to Anthropic format, extracting system messages.

        Returns (messages, system_prompt). System messages from the message list
        take priority over the configured system_prompt.
        """
        system_parts: list[str] = []
        chat_messages: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        # Inline system messages override config-level system_prompt
        system_prompt = "\n\n".join(system_parts) if system_parts else self._system_prompt
        return chat_messages, system_prompt or None

    async def _send_anthropic(self, messages: list[AdapterMessage]) -> AdapterResponse:
        if not self._anthropic_client:
            raise RuntimeError("Anthropic client not initialized. Call connect() first.")
        chat_messages, system_prompt = self._to_anthropic_messages(messages)
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._anthropic_client.messages.create(**kwargs)
        content = response.content[0].text if response.content else ""
        return AdapterResponse(content=content, content_type="markdown")

    async def _stream_anthropic(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        if not self._anthropic_client:
            raise RuntimeError("Anthropic client not initialized. Call connect() first.")
        chat_messages, system_prompt = self._to_anthropic_messages(messages)
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        # Use raw event stream to capture both thinking and text blocks
        async with self._anthropic_client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        yield StreamChunk(type="thinking", content=event.delta.thinking)
                    elif event.delta.type == "text_delta":
                        yield StreamChunk(type="text", content=event.delta.text)

            # Yield usage info from the final message
            try:
                final = await stream.get_final_message()
                if final and final.usage:
                    yield StreamChunk(
                        type="usage",
                        usage=UsageInfo(
                            input_tokens=final.usage.input_tokens,
                            output_tokens=final.usage.output_tokens,
                            model=final.model,
                            provider="anthropic",
                        ),
                    )
            except Exception:
                logger.debug("Could not extract usage info from Anthropic stream")

    # ── OpenAI ──

    def _to_openai_messages(self, messages: list[AdapterMessage]) -> list[dict]:
        msgs: list[dict] = []
        # Check if messages already contain a system message
        has_inline_system = any(m.role == "system" for m in messages)
        if not has_inline_system and self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    async def _send_openai(self, messages: list[AdapterMessage]) -> AdapterResponse:
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized. Call connect() first.")
        response = await self._openai_client.chat.completions.create(
            model=self._model,
            messages=self._to_openai_messages(messages),
        )
        content = response.choices[0].message.content or ""
        return AdapterResponse(content=content, content_type="markdown")

    async def _stream_openai(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized. Call connect() first.")
        openai_msgs = self._to_openai_messages(messages)
        try:
            stream = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=openai_msgs,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception:
            # Retry without stream_options for proxies that don't support it
            stream = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=openai_msgs,
                stream=True,
            )
        async for chunk in stream:
            # Final chunk with usage info (no choices)
            if hasattr(chunk, "usage") and chunk.usage and not chunk.choices:
                yield StreamChunk(
                    type="usage",
                    usage=UsageInfo(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        model=chunk.model or self._model,
                        provider="openai",
                    ),
                )
                continue
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # OpenAI reasoning models (o1/o3) may include reasoning content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield StreamChunk(type="thinking", content=reasoning)
            if delta.content:
                yield StreamChunk(type="text", content=delta.content)
