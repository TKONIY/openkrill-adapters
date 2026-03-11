"""Slack Adapter — forward OpenKrill messages to/from Slack channels.

This is an OUTBOUND adapter: it bridges an OpenKrill channel to a Slack channel.
Messages sent in the OpenKrill channel are forwarded to the Slack channel, and
messages received in the Slack channel (mentioning the bot) are forwarded back
to OpenKrill.

Config:
    bot_token: Slack Bot User OAuth Token (xoxb-...)
    app_token: Slack App-Level Token for Socket Mode (xapp-...)
    signing_secret: Slack app signing secret (for request verification)
    channel_id: Target Slack channel ID (e.g. C01ABCDEF)

Requires: pip install openkrill-adapters[slack]
    (installs slack_bolt[async])
"""

import asyncio
import logging
import re
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


class SlackAdapter(BaseAdapter):
    """Outbound adapter that bridges OpenKrill channels to Slack channels.

    Uses Slack's Socket Mode for real-time event reception (no public URL needed).

    Lifecycle:
        1. __init__(config) — parse bot_token, app_token, signing_secret, channel_id
        2. connect() — initialize AsyncApp + AsyncSocketModeHandler
        3. send() — forward message to Slack channel via chat_postMessage
        4. send_stream() — buffer chunks, send as single message (Slack has no streaming)
        5. disconnect() — stop Socket Mode handler
    """

    # Slack message length limit
    MAX_MESSAGE_LENGTH = 4000  # Slack's actual limit is ~40k but 4000 is practical

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._bot_token: str = config.get("bot_token", "")
        self._app_token: str = config.get("app_token", "")
        self._signing_secret: str = config.get("signing_secret", "")
        self._channel_id: str = config.get("channel_id", "")
        self._app: Any = None  # slack_bolt.async_app.AsyncApp instance
        self._handler: Any = None  # AsyncSocketModeHandler instance
        self._client: Any = None  # AsyncWebClient instance
        self._bot_user_id: str | None = None  # Bot's own user ID (to detect mentions)
        # Callback for forwarding inbound Slack messages to OpenKrill
        self._on_message_callback: Any = None
        # Background task for Socket Mode
        self._handler_task: asyncio.Task | None = None

        if not self._bot_token:
            raise ValueError("Slack adapter requires 'bot_token' in config")
        if not self._app_token:
            raise ValueError("Slack adapter requires 'app_token' in config (for Socket Mode)")
        if not self._signing_secret:
            raise ValueError("Slack adapter requires 'signing_secret' in config")
        if not self._channel_id:
            raise ValueError("Slack adapter requires 'channel_id' in config")

    async def connect(self) -> None:
        """Initialize the Slack AsyncApp and start Socket Mode handler."""
        try:
            from slack_bolt.async_app import AsyncApp
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError as e:
            raise ImportError(
                "slack_bolt is required for the Slack adapter. "
                "Install with: pip install openkrill-adapters[slack]"
            ) from e

        self._client = AsyncWebClient(token=self._bot_token)
        self._app = AsyncApp(
            token=self._bot_token,
            signing_secret=self._signing_secret,
        )

        # Get bot's own user ID for mention detection
        try:
            auth_response = await self._client.auth_test()
            self._bot_user_id = auth_response.get("user_id")
            logger.info(
                "Slack adapter authenticated as user_id=%s", self._bot_user_id
            )
        except Exception:
            logger.exception("Failed to get Slack bot user ID")

        # Register event handlers
        self._register_event_handlers()

        logger.info("Slack adapter connected (channel_id=%s)", self._channel_id)

    async def disconnect(self) -> None:
        """Stop Socket Mode handler and clean up."""
        if self._handler is not None:
            try:
                await self._handler.close_async()
            except Exception:
                logger.debug("Error closing Socket Mode handler", exc_info=True)
            self._handler = None
        if self._handler_task is not None:
            self._handler_task.cancel()
            self._handler_task = None
        self._app = None
        self._client = None
        self._bot_user_id = None
        logger.info("Slack adapter disconnected")

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Forward the last message to the Slack channel.

        In outbound mode, we forward only the latest assistant message.
        The full conversation history is maintained by OpenKrill.
        """
        if self._client is None:
            raise RuntimeError("Slack client not initialized. Call connect() first.")

        message = self._pick_outbound_message(messages)
        if not message:
            return AdapterResponse(content="(no message to forward)", content_type="text")

        text = self._format_message(message)
        sent = await self._send_text(text)
        return AdapterResponse(
            content=sent,
            content_type="text",
            metadata={"slack_channel_id": self._channel_id},
        )

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        """Buffer all chunks and send as a single Slack message.

        Slack supports message editing via chat_update, but frequent updates
        trigger rate limits. For simplicity, we buffer and send once.
        """
        response = await self.send(messages)
        yield StreamChunk(type="text", content=response.content)

    async def health_check(self) -> bool:
        """Call auth.test to verify the bot token is valid."""
        if self._client is None:
            return False
        try:
            response = await self._client.auth_test()
            ok = response.get("ok", False)
            logger.debug("Slack health check: ok=%s", ok)
            return bool(ok)
        except Exception:
            logger.exception("Slack health check failed")
            return False

    @property
    def adapter_type(self) -> str:
        return "slack"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L1_RICH

    # ── Inbound: receive messages from Slack ──

    def set_on_message(self, callback: Any) -> None:
        """Register a callback for inbound Slack messages.

        The callback signature:
            async def on_message(text: str, sender_name: str, metadata: dict) -> None
        """
        self._on_message_callback = callback

    async def start_listening(self) -> None:
        """Start Socket Mode handler to receive messages from Slack.

        Call this after connect() to enable bidirectional sync.
        Messages that mention the bot (@bot) in the configured channel will
        be forwarded to OpenKrill via the registered on_message callback.
        """
        if self._app is None:
            raise RuntimeError("Slack app not initialized. Call connect() first.")

        try:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
        except ImportError as e:
            raise ImportError(
                "slack_bolt with socket mode support is required. "
                "Install with: pip install openkrill-adapters[slack]"
            ) from e

        self._handler = AsyncSocketModeHandler(
            app=self._app,
            app_token=self._app_token,
        )

        # Start in background task so it doesn't block
        self._handler_task = asyncio.create_task(self._handler.start_async())
        logger.info("Slack Socket Mode handler started for channel_id=%s", self._channel_id)

    # ── Internal helpers ──

    def _register_event_handlers(self) -> None:
        """Register Slack event handlers on the AsyncApp."""
        if self._app is None:
            return

        adapter = self  # capture for closure

        @self._app.event("app_mention")
        async def handle_app_mention(event: dict, say: Any) -> None:
            """Handle @bot mentions in Slack channels."""
            channel = event.get("channel", "")
            if channel != adapter._channel_id:
                return

            text = event.get("text", "")
            user_id = event.get("user", "")

            # Strip the bot mention from the text
            text = adapter._strip_bot_mention(text)

            if adapter._on_message_callback and text.strip():
                await adapter._on_message_callback(
                    text=text.strip(),
                    sender_name=user_id,  # Slack user ID; caller can resolve display name
                    metadata={
                        "slack_channel_id": channel,
                        "slack_user_id": user_id,
                        "slack_ts": event.get("ts", ""),
                        "slack_thread_ts": event.get("thread_ts"),
                        "slack_event_type": "app_mention",
                    },
                )

        @self._app.event("message")
        async def handle_message(event: dict, say: Any) -> None:
            """Handle direct messages and channel messages.

            We only process:
            - Messages in the configured channel
            - DMs to the bot
            - Skip bot's own messages and subtypes (joins, edits, etc.)
            """
            # Ignore message subtypes (edits, deletes, bot messages, etc.)
            if event.get("subtype"):
                return

            channel = event.get("channel", "")
            user_id = event.get("user", "")

            # Ignore messages from the bot itself
            if user_id == adapter._bot_user_id:
                return

            # Only process messages from the configured channel
            if channel != adapter._channel_id:
                return

            text = event.get("text", "")
            if not text.strip():
                return

            # For channel messages, only respond if the bot is mentioned
            # (app_mention handler covers explicit @mentions;
            #  this catches DMs in the channel or threads)
            channel_type = event.get("channel_type", "")
            if channel_type == "im":
                # Direct message — always process
                pass
            else:
                # Channel message without @mention — skip (app_mention handles those)
                return

            if adapter._on_message_callback:
                await adapter._on_message_callback(
                    text=text.strip(),
                    sender_name=user_id,
                    metadata={
                        "slack_channel_id": channel,
                        "slack_user_id": user_id,
                        "slack_ts": event.get("ts", ""),
                        "slack_thread_ts": event.get("thread_ts"),
                        "slack_event_type": "message",
                        "slack_channel_type": channel_type,
                    },
                )

    def _strip_bot_mention(self, text: str) -> str:
        """Remove the bot's @mention from message text.

        Slack formats mentions as <@U12345678>. We strip the bot's own mention
        so the forwarded text is clean.
        """
        if self._bot_user_id:
            # Remove <@BOT_USER_ID> pattern
            text = re.sub(rf"<@{re.escape(self._bot_user_id)}>", "", text)
        return text.strip()

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
        """Format an AdapterMessage for Slack.

        Slack uses its own markup (mrkdwn) which is similar to Markdown but
        with some differences (e.g., *bold* instead of **bold**).
        We pass through as-is since most Markdown works reasonably well.
        """
        text = message.content
        if message.content_type not in ("text", "markdown"):
            text = f"[{message.content_type}] {text}"
        return text

    async def _send_text(self, text: str) -> str:
        """Send text to the Slack channel, splitting if too long."""
        if self._client is None:
            raise RuntimeError("Client not initialized")

        chunks = self._split_text(text, self.MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            await self._client.chat_postMessage(
                channel=self._channel_id,
                text=chunk,
            )
        return text

    @staticmethod
    def _split_text(text: str, max_length: int) -> list[str]:
        """Split text into chunks that fit within Slack's message limit."""
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
