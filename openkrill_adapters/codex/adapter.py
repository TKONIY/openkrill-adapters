"""Codex adapter — delegates to CLI or Session CLI adapter.

Branded adapter for OpenAI's Codex coding agent.
Delegates all functionality to CliAdapter (cli mode) or SessionCliAdapter
(session mode) based on the ``mode`` config field.

Config keys:
    mode: str — "cli" or "session" (default: "cli")
    command: str — Override CLI command (default: "codex")
    args: str — Override CLI args
"""

from collections.abc import AsyncIterator

from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    StreamChunk,
)
from openkrill_adapters.cli.adapter import CliAdapter
from openkrill_adapters.cli.session_adapter import SessionCliAdapter


class CodexAdapter(BaseAdapter):
    """Branded adapter for Codex (OpenAI's coding agent)."""

    COMMAND = "codex"
    CLI_ARGS = ""
    SESSION_ARGS = ""
    TYPE_NAME = "codex"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._mode = config.get("mode", "cli")  # "cli" or "session"

        # Build delegate config
        delegate_config = {**config}
        delegate_config["command"] = config.get("command", self.COMMAND)

        if self._mode == "session":
            delegate_config["args"] = config.get("args", self.SESSION_ARGS)
            self._delegate: BaseAdapter = SessionCliAdapter(delegate_config)
        else:
            delegate_config["args"] = config.get("args", self.CLI_ARGS)
            self._delegate = CliAdapter(delegate_config)

    @property
    def adapter_type(self) -> str:
        return self.TYPE_NAME

    @property
    def max_capability(self) -> AdapterCapability:
        return self._delegate.max_capability

    async def connect(self) -> None:
        await self._delegate.connect()

    async def disconnect(self) -> None:
        await self._delegate.disconnect()

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        return await self._delegate.send(messages)

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        async for chunk in self._delegate.send_stream(messages):
            yield chunk

    async def health_check(self) -> bool:
        return await self._delegate.health_check()
