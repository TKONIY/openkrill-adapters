"""Agent Protocol — the interface that external agents must implement.

Any agent framework can be adapted to OpenKrill by implementing this protocol.
The protocol is intentionally minimal: receive messages, return a response.

Example usage with a custom agent::

    from openkrill_adapters.agent.protocol import OpenKrillAgent

    class MyResearchAgent(OpenKrillAgent):
        async def run(self, messages: list[dict]) -> str:
            # messages = [{"role": "user", "content": "..."}, ...]
            result = await my_research_pipeline(messages[-1]["content"])
            return result

        async def run_stream(self, messages: list[dict]) -> AsyncIterator[str]:
            async for chunk in my_streaming_pipeline(messages[-1]["content"]):
                yield chunk
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class OpenKrillAgent(ABC):
    """Protocol that external agents must implement to integrate with OpenKrill.

    Message format:
        Each message is a dict with at least:
        - role: "user" | "assistant" | "system"
        - content: str

        Optional fields:
        - content_type: "text" | "markdown" (default: "text")
        - metadata: dict with additional context

    Lifecycle:
        1. __init__(**config) — framework-specific configuration
        2. initialize() — async setup (load models, connect to services)
        3. run() / run_stream() — process messages
        4. cleanup() — async teardown
    """

    @abstractmethod
    async def run(self, messages: list[dict]) -> str:
        """Process messages and return a complete response.

        Args:
            messages: Conversation history as list of dicts.

        Returns:
            The agent's response as a string.
        """

    async def run_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Process messages and yield response chunks for streaming.

        Default implementation falls back to run() and yields the full response.
        Override for true streaming support.

        Args:
            messages: Conversation history as list of dicts.

        Yields:
            String chunks of the response.
        """
        result = await self.run(messages)
        yield result

    async def initialize(self) -> None:  # noqa: B027
        """Async initialization hook. Called once after instantiation.

        Override to set up async resources (HTTP clients, model loading, etc.).
        """

    async def cleanup(self) -> None:  # noqa: B027
        """Async cleanup hook. Called on adapter disconnect.

        Override to release resources (close connections, etc.).
        """

    def health(self) -> bool:
        """Check if the agent is healthy and ready to process messages.

        Returns:
            True if the agent is operational.
        """
        return True
