"""Telegram Adapter — forward OpenKrill messages to/from Telegram chats.

This is an OUTBOUND adapter: it bridges an OpenKrill channel to a Telegram chat.
Messages sent in the OpenKrill channel are forwarded to the Telegram chat, and
messages received in the Telegram chat are forwarded back to OpenKrill.

Config:
    bot_token: Telegram Bot API token (from @BotFather)
    chat_id: Target Telegram chat ID (group or user)
    parse_mode: Message formatting — "Markdown" or "HTML" (default: "Markdown")

Requires: pip install openkrill-adapters[telegram]
    (installs python-telegram-bot)
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


class TelegramAdapter(BaseAdapter):
    """Outbound adapter that bridges OpenKrill channels to Telegram chats.

    Lifecycle:
        1. __init__(config) — parse bot_token, chat_id, parse_mode
        2. connect() — initialize telegram.Bot client
        3. send() — forward message to Telegram chat
        4. send_stream() — buffer chunks, send as single message (Telegram has no streaming)
        5. disconnect() — stop webhook/polling if active
    """

    # Telegram message length limit
    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._bot_token: str = config.get("bot_token", "")
        self._chat_id: str | int = config.get("chat_id", "")
        self._parse_mode: str = config.get("parse_mode", "Markdown")
        self._bot: Any = None  # telegram.Bot instance
        self._app: Any = None  # telegram.ext.Application (for webhook/polling)
        # Callback for forwarding inbound Telegram messages to OpenKrill
        self._on_message_callback: Any = None

        if not self._bot_token:
            raise ValueError("Telegram adapter requires 'bot_token' in config")
        if not self._chat_id:
            raise ValueError("Telegram adapter requires 'chat_id' in config")

    async def connect(self) -> None:
        """Initialize the Telegram Bot client."""
        try:
            from telegram import Bot
        except ImportError as e:
            raise ImportError(
                "python-telegram-bot is required for the Telegram adapter. "
                "Install with: pip install openkrill-adapters[telegram]"
            ) from e

        self._bot = Bot(token=self._bot_token)
        logger.info("Telegram adapter connected (chat_id=%s)", self._chat_id)

    async def disconnect(self) -> None:
        """Stop polling/webhook and clean up."""
        if self._app is not None:
            await self._app.stop()
            self._app = None
        self._bot = None
        logger.info("Telegram adapter disconnected")

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Forward the last message to the Telegram chat.

        In outbound mode, we typically forward only the latest assistant message.
        The full conversation history is maintained by OpenKrill, not Telegram.
        """
        if not self._bot:
            raise RuntimeError("Telegram bot not initialized. Call connect() first.")

        # Find the last assistant or user message to forward
        message = self._pick_outbound_message(messages)
        if not message:
            return AdapterResponse(content="(no message to forward)", content_type="text")

        text = self._format_message(message)
        sent = await self._send_text(text)
        return AdapterResponse(
            content=sent,
            content_type="text",
            metadata={"telegram_chat_id": self._chat_id},
        )

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        """Buffer all chunks and send as a single Telegram message.

        Telegram does not support streaming/incremental message updates in a
        practical way (editing messages has rate limits), so we buffer the full
        response and send it once complete.
        """
        # Outbound adapters receive the final message via send(), not send_stream().
        # This method is here for interface compliance. We just delegate to send().
        response = await self.send(messages)
        yield StreamChunk(type="text", content=response.content)

    async def health_check(self) -> bool:
        """Call getMe() to verify the bot token is valid and the bot is reachable."""
        if not self._bot:
            return False
        try:
            me = await self._bot.get_me()
            logger.debug("Telegram bot healthy: @%s", me.username)
            return True
        except Exception:
            logger.exception("Telegram health check failed")
            return False

    @property
    def adapter_type(self) -> str:
        return "telegram"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L1_RICH

    # ── Inbound: receive messages from Telegram ──

    def set_on_message(self, callback: Any) -> None:
        """Register a callback for inbound Telegram messages.

        The callback signature:
            async def on_message(text: str, sender_name: str, metadata: dict) -> None
        """
        self._on_message_callback = callback

    async def start_polling(self) -> None:
        """Start long-polling to receive messages from the Telegram chat.

        Call this after connect() to enable bidirectional sync.
        Messages received from Telegram will be forwarded to OpenKrill
        via the registered on_message callback.
        """
        try:
            from telegram.ext import ApplicationBuilder, MessageHandler, filters
        except ImportError as e:
            raise ImportError(
                "python-telegram-bot is required. "
                "Install with: pip install openkrill-adapters[telegram]"
            ) from e

        self._app = ApplicationBuilder().token(self._bot_token).build()

        async def _handle_message(update: Any, context: Any) -> None:
            """Forward incoming Telegram message to OpenKrill."""
            msg = update.message
            if msg is None or msg.text is None:
                return
            # Only forward messages from the configured chat
            if str(msg.chat_id) != str(self._chat_id):
                return
            if self._on_message_callback:
                sender_name = msg.from_user.full_name if msg.from_user else "Unknown"
                await self._on_message_callback(
                    text=msg.text,
                    sender_name=sender_name,
                    metadata={
                        "telegram_message_id": msg.message_id,
                        "telegram_chat_id": msg.chat_id,
                        "telegram_user_id": msg.from_user.id if msg.from_user else None,
                    },
                )

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("Telegram polling started for chat_id=%s", self._chat_id)

    # ── Internal helpers ──

    @staticmethod
    def _pick_outbound_message(messages: list[AdapterMessage]) -> AdapterMessage | None:
        """Pick the message to forward outbound.

        For outbound adapters, we forward the latest assistant message.
        If there is no assistant message, forward the latest user message.
        """
        for m in reversed(messages):
            if m.role == "assistant":
                return m
        for m in reversed(messages):
            if m.role == "user":
                return m
        return None

    def _format_message(self, message: AdapterMessage) -> str:
        """Format an AdapterMessage for Telegram.

        Telegram supports a subset of Markdown. We do minimal conversion:
        - Markdown content_type: pass through (Telegram handles basic Markdown)
        - Other content types: send as plain text
        """
        text = message.content
        if message.content_type not in ("text", "markdown"):
            # For non-text content (images, files), include a description
            text = f"[{message.content_type}] {text}"
        return text

    async def _send_text(self, text: str) -> str:
        """Send text to the Telegram chat, splitting if too long."""
        if not self._bot:
            raise RuntimeError("Bot not initialized")

        # Split long messages at Telegram's 4096-char limit
        chunks = self._split_text(text, self.MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=chunk,
                    parse_mode=self._parse_mode,
                )
            except Exception:
                # Fallback: send without parse_mode if formatting fails
                logger.debug("Telegram parse_mode failed, retrying as plain text")
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=chunk,
                )
        return text

    @staticmethod
    def _split_text(text: str, max_length: int) -> list[str]:
        """Split text into chunks that fit within Telegram's message limit."""
        if len(text) <= max_length:
            return [text]
        chunks: list[str] = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            # Try to split at a newline
            split_pos = text.rfind("\n", 0, max_length)
            if split_pos == -1:
                split_pos = max_length
            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip("\n")
        return chunks
