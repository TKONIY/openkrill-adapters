"""CLI Adapter — routes messages through the Agent Bridge to local CLI tools.

The CliAdapter doesn't directly spawn processes. Instead, it delegates to a
bridge_send callable injected by the server. This keeps the adapter decoupled
from the server's WebSocket infrastructure.

Config keys:
    command: str — CLI command to run (e.g., "claude", "codex")
    args: str — Additional args (e.g., "-p", "exec")
    agent_id: str — Agent UUID (used by bridge_send to route)
    _bridge_send: Callable — Injected by server, sends request to bridge daemon
"""

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

# Type for the bridge send callable injected by the server
BridgeSendFn = Callable[[str, list[dict]], Coroutine[Any, Any, dict]]


class CliAdapter(BaseAdapter):
    """Adapter that routes through Agent Bridge to local CLI tools."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._command = config.get("command", "claude")
        self._args = config.get("args", "-p")
        self._agent_id = config.get("agent_id", "")
        self._bridge_send: BridgeSendFn | None = config.get("_bridge_send")

    async def connect(self) -> None:
        if not self._bridge_send:
            raise RuntimeError(
                "CLI adapter requires a bridge connection. "
                "Start the bridge daemon on your local machine."
            )

    async def disconnect(self) -> None:
        pass

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        if not self._bridge_send:
            raise RuntimeError("No bridge connected")

        msg_dicts = [
            {"role": m.role, "content": m.content, "content_type": m.content_type} for m in messages
        ]

        result = await self._bridge_send(self._agent_id, msg_dicts)
        return AdapterResponse(
            content=result.get("content", ""),
            content_type=result.get("content_type", "text"),
        )

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        # For now, CLI adapter uses non-streaming (full response as single chunk)
        response = await self.send(messages)
        yield StreamChunk(type="text", content=response.content)

    async def health_check(self) -> bool:
        return self._bridge_send is not None

    @property
    def adapter_type(self) -> str:
        return "cli"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L0_TEXT
