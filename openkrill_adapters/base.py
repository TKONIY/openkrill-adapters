"""Base adapter interface.

All adapters implement this interface. The server doesn't care about the
underlying transport — API calls, CLI stdin/stdout, browser automation,
or agent framework protocols are all normalized to this contract.

Design inspired by OpenClaw's channel adapter pattern:
- Normalize inbound messages to a unified format
- Format outbound messages for the target
- Manage auth/connection lifecycle
- Declare capabilities (L0-L3)
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import IntEnum


class AdapterCapability(IntEnum):
    """Capability levels that a destination platform supports.

    Adapters auto-degrade to the highest level the destination supports.
    """

    L0_TEXT = 0  # Plain text messages (all platforms)
    L1_RICH = 1  # Rich content: images, files, code blocks (most platforms)
    L2_CONTEXT = 2  # Context sharing, drag-and-drop injection (OpenKrill)
    L3_IMMERSIVE = 3  # Agent View immersive UI (OpenKrill)


@dataclass
class AdapterMessage:
    """Normalized message format across all adapters."""

    role: str  # "user", "assistant", or "system"
    content: str
    content_type: str = "text"  # "text", "markdown", "image", "file"
    metadata: dict = field(default_factory=dict)


@dataclass
class AdapterResponse:
    """A complete (non-streaming) response from an adapter."""

    content: str
    content_type: str = "text"
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A single chunk in a streaming response.

    type: "thinking" for CoT/reasoning tokens, "text" for final response.
    """

    type: str  # "thinking" or "text"
    content: str


class BaseAdapter(ABC):
    """Unified interface for all adapter types.

    Lifecycle:
        1. __init__(config) — create with adapter_config from DB
        2. connect() — establish connection (API client, CLI process, browser session)
        3. send() / send_stream() — exchange messages
        4. health_check() — verify connection is alive
        5. disconnect() — clean up resources
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the AI source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up resources."""

    @abstractmethod
    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Send messages and get a complete response."""

    @abstractmethod
    def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        """Send messages and get a streaming response.

        Yields:
            StreamChunk: Each chunk with type ("thinking" or "text") and content.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the connection is alive and usable."""

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """Return the adapter type identifier (e.g., 'api', 'cli', 'web', 'agent')."""

    @property
    def max_capability(self) -> AdapterCapability:
        """The highest capability level this adapter supports.

        Override in subclasses if the adapter supports higher levels.
        Default is L0 (text only).
        """
        return AdapterCapability.L0_TEXT
