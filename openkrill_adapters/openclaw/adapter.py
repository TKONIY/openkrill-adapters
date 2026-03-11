"""OpenClaw adapter — bidirectional message bridge to OpenClaw Gateway.

This is an OUTBOUND adapter: it bridges an OpenKrill channel to an OpenClaw
Gateway connection. Unlike Discord/Telegram adapters that maintain their own
external connections, the OpenClaw adapter uses injected callbacks from the
server's OpenClaw manager (similar to how CLI adapters use bridge callbacks).

Config:
    agent_id: UUID of the agent this adapter serves
    gateway_url: (optional) External OpenClaw gateway URL
    _gateway_send: Injected callback to send messages to the connected OpenClaw client
        async def(agent_id: str, event_type: str, payload: dict) -> None
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class OpenClawAdapter(BaseAdapter):
    """Outbound adapter that bridges OpenKrill channels to OpenClaw Gateway.

    Lifecycle:
        1. __init__(config) — parse agent_id, gateway_url, injected callbacks
        2. connect() — no-op (managed by gateway handler)
        3. send() — forward message to OpenClaw client via gateway callback
        4. send_stream() — forward as single message (streaming handled at handler level)
        5. disconnect() — no-op
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._agent_id: str = config.get("agent_id", "")
        self._gateway_url: str = config.get("gateway_url", "")
        self._gateway_send: Any = config.get("_gateway_send")
        self._on_message_callback: Any = None

    async def connect(self) -> None:
        """No-op — connection is managed by the OpenClaw gateway handler."""

    async def disconnect(self) -> None:
        """No-op — connection is managed by the OpenClaw gateway handler."""

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Forward the latest assistant message to the OpenClaw client.

        In outbound mode, we forward only the latest assistant message.
        The full conversation history is maintained by OpenKrill.
        """
        message = self._pick_outbound_message(messages)
        if not message:
            return AdapterResponse(content="(no message to forward)", content_type="text")

        if self._gateway_send:
            await self._gateway_send(
                self._agent_id,
                "message.new",
                {
                    "content": message.content,
                    "content_type": message.content_type,
                },
            )

        return AdapterResponse(
            content=message.content,
            content_type=message.content_type,
        )

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        """Buffer all chunks and send as a single message.

        Real-time streaming to the OpenClaw client is handled at the gateway
        handler level (openclaw_handler.py), not here. This method is used
        when the adapter is invoked through the normal adapter pipeline.
        """
        response = await self.send(messages)
        yield StreamChunk(type="text", content=response.content)

    async def health_check(self) -> bool:
        """Check if the gateway send callback is available."""
        return self._gateway_send is not None

    @property
    def adapter_type(self) -> str:
        return "openclaw"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L1_RICH

    # ── Inbound: receive messages from OpenClaw ──

    def set_on_message(self, callback: Any) -> None:
        """Register a callback for inbound OpenClaw messages.

        The callback signature:
            async def on_message(text: str, sender_name: str, metadata: dict) -> None
        """
        self._on_message_callback = callback

    # ── Internal helpers ──

    @staticmethod
    def _pick_outbound_message(messages: list[AdapterMessage]) -> AdapterMessage | None:
        """Pick the message to forward outbound (latest assistant, then user)."""
        for m in reversed(messages):
            if m.role == "assistant":
                return m
        for m in reversed(messages):
            if m.role == "user":
                return m
        return None
