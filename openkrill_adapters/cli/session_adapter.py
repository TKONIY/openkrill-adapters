"""Session CLI Adapter — routes through bridge to persistent Claude Code sessions.

Unlike the regular CliAdapter which is request-response, this adapter:
- Maintains session continuity via Claude Code's --resume flag
- Supports true streaming (yields chunks as they arrive from bridge)
- Supports interrupt/cancel

Config keys:
    agent_id: str — Agent UUID
    _bridge_stream_send: Callable — Injected by server, returns asyncio.Queue of stream chunks
    _bridge_interrupt: Callable — Injected by server, sends interrupt signal
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    StreamChunk,
)

logger = logging.getLogger(__name__)

# Bridge callable types
BridgeStreamSendFn = Callable[[str, list[dict]], Coroutine[Any, Any, asyncio.Queue]]
BridgeInterruptFn = Callable[[str], Coroutine[Any, Any, None]]


class SessionCliAdapter(BaseAdapter):
    """Adapter for persistent CLI sessions via bridge."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._agent_id = config.get("agent_id", "")
        self._bridge_stream_send: BridgeStreamSendFn | None = config.get("_bridge_stream_send")
        self._bridge_interrupt: BridgeInterruptFn | None = config.get("_bridge_interrupt")

    async def connect(self) -> None:
        if not self._bridge_stream_send:
            raise RuntimeError(
                "Session CLI adapter requires a bridge connection. "
                "Start the bridge daemon with --session-mode."
            )

    async def disconnect(self) -> None:
        pass

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Non-streaming fallback — collects all chunks."""
        parts: list[str] = []
        async for chunk in self.send_stream(messages):
            if chunk.type == "text":
                parts.append(chunk.content)
        return AdapterResponse(content="".join(parts))

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        """Stream response from persistent Claude Code session."""
        if not self._bridge_stream_send:
            raise RuntimeError("No bridge connected")

        msg_dicts = [
            {"role": m.role, "content": m.content, "content_type": m.content_type} for m in messages
        ]

        queue: asyncio.Queue = await self._bridge_stream_send(self._agent_id, msg_dicts)

        while True:
            item = await asyncio.wait_for(queue.get(), timeout=300)
            if item is None:
                break  # Stream ended
            if isinstance(item, dict):
                if item.get("type") == "error":
                    raise RuntimeError(item.get("error", "Bridge stream error"))
                chunk_type = item.get("chunk_type", "text")
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                if chunk_type in ("tool_use", "tool_result"):
                    # Tool events always forwarded (content may be empty)
                    yield StreamChunk(type=chunk_type, content=content, metadata=metadata)
                elif content:
                    yield StreamChunk(type=chunk_type, content=content)

    async def health_check(self) -> bool:
        return self._bridge_stream_send is not None

    @property
    def adapter_type(self) -> str:
        return "cli-session"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L1_RICH
