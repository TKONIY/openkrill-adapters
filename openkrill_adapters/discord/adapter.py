"""Discord Adapter — forward OpenKrill messages to/from Discord channels.

This is an OUTBOUND adapter: it bridges an OpenKrill channel to a Discord channel.
Messages sent in the OpenKrill channel are forwarded to the Discord channel, and
messages received in the Discord channel are forwarded back to OpenKrill.

Config:
    bot_token: Discord Bot token (from Discord Developer Portal)
    channel_id: Target Discord channel ID (snowflake)
    guild_id: (optional) Discord guild/server ID for validation

Requires: pip install openkrill-adapters[discord]
    (installs discord.py)
"""

import asyncio
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


class DiscordAdapter(BaseAdapter):
    """Outbound adapter that bridges OpenKrill channels to Discord channels.

    Lifecycle:
        1. __init__(config) — parse bot_token, channel_id
        2. connect() — login Discord bot, cache channel reference
        3. send() — forward message to Discord channel
        4. send_stream() — buffer chunks, send as single message
        5. disconnect() — close Discord client
    """

    # Discord message length limit
    MAX_MESSAGE_LENGTH = 2000

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._bot_token: str = config.get("bot_token", "")
        self._channel_id: int = int(config.get("channel_id", 0))
        self._guild_id: int | None = int(config["guild_id"]) if config.get("guild_id") else None
        self._client: Any = None  # discord.Client instance
        self._channel: Any = None  # discord.TextChannel reference
        self._ready_event: asyncio.Event = asyncio.Event()
        # Callback for forwarding inbound Discord messages to OpenKrill
        self._on_message_callback: Any = None
        # Background task running the Discord client
        self._client_task: asyncio.Task | None = None

        if not self._bot_token:
            raise ValueError("Discord adapter requires 'bot_token' in config")
        if not self._channel_id:
            raise ValueError("Discord adapter requires 'channel_id' in config")

    async def connect(self) -> None:
        """Login the Discord bot and cache the target channel."""
        try:
            import discord
        except ImportError as e:
            raise ImportError(
                "discord.py is required for the Discord adapter. "
                "Install with: pip install openkrill-adapters[discord]"
            ) from e

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        adapter = self  # capture for closures

        @self._client.event
        async def on_ready() -> None:
            """Called when the Discord bot has connected and is ready."""
            logger.info("Discord bot connected as %s", self._client.user)
            adapter._channel = self._client.get_channel(adapter._channel_id)
            if adapter._channel is None:
                logger.warning(
                    "Discord channel %d not found — the bot may not have access to this channel",
                    adapter._channel_id,
                )
            adapter._ready_event.set()

        @self._client.event
        async def on_message(message: Any) -> None:
            """Forward incoming Discord messages to OpenKrill."""
            # Ignore messages from the bot itself
            if message.author == self._client.user:
                return
            # Only process messages from the configured channel
            if message.channel.id != adapter._channel_id:
                return
            if adapter._on_message_callback and message.content:
                await adapter._on_message_callback(
                    text=message.content,
                    sender_name=message.author.display_name,
                    metadata={
                        "discord_message_id": message.id,
                        "discord_channel_id": message.channel.id,
                        "discord_user_id": message.author.id,
                        "discord_guild_id": message.guild.id if message.guild else None,
                    },
                )

        # Start the Discord client in a background task
        self._client_task = asyncio.create_task(self._client.start(self._bot_token))

        # Wait for the bot to be ready (with timeout)
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
        except TimeoutError as e:
            raise RuntimeError("Discord bot failed to connect within 30 seconds") from e

        logger.info("Discord adapter connected (channel_id=%d)", self._channel_id)

    async def disconnect(self) -> None:
        """Close the Discord client connection."""
        if self._client is not None:
            await self._client.close()
        if self._client_task is not None:
            self._client_task.cancel()
            self._client_task = None
        self._client = None
        self._channel = None
        self._ready_event.clear()
        logger.info("Discord adapter disconnected")

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Forward the last message to the Discord channel.

        In outbound mode, we forward only the latest assistant message.
        The full conversation history is maintained by OpenKrill.
        """
        if self._channel is None:
            raise RuntimeError("Discord channel not available. Call connect() first.")

        message = self._pick_outbound_message(messages)
        if not message:
            return AdapterResponse(content="(no message to forward)", content_type="text")

        text = self._format_message(message)
        sent = await self._send_text(text)
        return AdapterResponse(
            content=sent,
            content_type="text",
            metadata={"discord_channel_id": self._channel_id},
        )

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        """Buffer all chunks and send as a single Discord message.

        Discord does support message editing, but frequent edits trigger rate
        limits. For simplicity, we buffer the full response and send once.
        """
        response = await self.send(messages)
        yield StreamChunk(type="text", content=response.content)

    async def health_check(self) -> bool:
        """Check if the Discord bot is connected and the channel is accessible."""
        if self._client is None:
            return False
        try:
            return self._client.is_ready() and self._channel is not None
        except Exception:
            logger.exception("Discord health check failed")
            return False

    @property
    def adapter_type(self) -> str:
        return "discord"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L1_RICH

    # ── Inbound: receive messages from Discord ──

    def set_on_message(self, callback: Any) -> None:
        """Register a callback for inbound Discord messages.

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

    def _format_message(self, message: AdapterMessage) -> str:
        """Format an AdapterMessage for Discord.

        Discord uses a subset of Markdown natively. Most Markdown passes through.
        """
        text = message.content
        if message.content_type not in ("text", "markdown"):
            text = f"[{message.content_type}] {text}"
        return text

    async def _send_text(self, text: str) -> str:
        """Send text to the Discord channel, splitting if too long."""
        if self._channel is None:
            raise RuntimeError("Channel not available")

        chunks = self._split_text(text, self.MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            await self._channel.send(chunk)
        return text

    @staticmethod
    def _split_text(text: str, max_length: int) -> list[str]:
        """Split text into chunks that fit within Discord's message limit."""
        if len(text) <= max_length:
            return [text]
        chunks: list[str] = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            # Try to split at a newline for readability
            split_pos = text.rfind("\n", 0, max_length)
            if split_pos == -1:
                split_pos = max_length
            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip("\n")
        return chunks
