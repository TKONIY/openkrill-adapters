"""Agent Adapter — connects agent frameworks to OpenKrill.

Dynamically loads an external agent module that implements the OpenKrillAgent
protocol, then proxies messages through it.

Config keys:
    agent_framework: str — framework hint ('langchain', 'crewai', 'autogen', 'custom')
    entry_point: str — dotted module path to the agent class (e.g. 'myapp.agents.ResearchAgent')
    agent_config: dict — framework-specific config passed to the agent's __init__
    system_prompt: str — optional system prompt prepended to messages
"""

import importlib
import logging
from collections.abc import AsyncIterator

from openkrill_adapters.agent.protocol import OpenKrillAgent
from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class AgentAdapter(BaseAdapter):
    """Adapter that delegates to an external agent implementing OpenKrillAgent."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._framework: str = config.get("agent_framework", "custom")
        self._entry_point: str = config.get("entry_point", "")
        self._agent_config: dict = config.get("agent_config", {})
        self._system_prompt: str = config.get("system_prompt", "")
        self._agent: OpenKrillAgent | None = None

    async def connect(self) -> None:
        """Dynamically import the agent class and instantiate it."""
        if not self._entry_point:
            raise RuntimeError(
                "Agent adapter requires 'entry_point' config — "
                "a dotted path to a class implementing OpenKrillAgent "
                "(e.g. 'myapp.agents.ResearchAgent')."
            )

        agent_class = self._load_agent_class(self._entry_point)

        # Verify it implements the protocol
        if not (isinstance(agent_class, type) and issubclass(agent_class, OpenKrillAgent)):
            raise TypeError(
                f"Agent class '{self._entry_point}' must be a subclass of "
                f"OpenKrillAgent. Got: {type(agent_class)}"
            )

        # Instantiate with framework-specific config
        self._agent = agent_class(**self._agent_config)
        await self._agent.initialize()
        logger.info(
            "Agent adapter connected: framework=%s, entry_point=%s",
            self._framework,
            self._entry_point,
        )

    async def disconnect(self) -> None:
        """Clean up the agent instance."""
        if self._agent:
            await self._agent.cleanup()
            self._agent = None
            logger.info("Agent adapter disconnected")

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Send messages to the agent and get a complete response."""
        if not self._agent:
            raise RuntimeError("Agent not initialized. Call connect() first.")

        msg_dicts = self._to_message_dicts(messages)
        content = await self._agent.run(msg_dicts)
        return AdapterResponse(content=content, content_type="markdown")

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[StreamChunk]:
        """Send messages and stream the response."""
        if not self._agent:
            raise RuntimeError("Agent not initialized. Call connect() first.")

        msg_dicts = self._to_message_dicts(messages)
        async for chunk in self._agent.run_stream(msg_dicts):
            yield StreamChunk(type="text", content=chunk)

    async def health_check(self) -> bool:
        """Check if the agent is healthy."""
        if not self._agent:
            return False
        try:
            return self._agent.health()
        except Exception:
            logger.debug("Agent health check failed", exc_info=True)
            return False

    @property
    def adapter_type(self) -> str:
        return "agent"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L2_CONTEXT

    # ── Internal helpers ──

    def _to_message_dicts(self, messages: list[AdapterMessage]) -> list[dict]:
        """Convert AdapterMessages to plain dicts for the agent protocol."""
        result: list[dict] = []
        # Prepend system prompt if configured and not already in messages
        has_system = any(m.role == "system" for m in messages)
        if not has_system and self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        for m in messages:
            result.append(
                {
                    "role": m.role,
                    "content": m.content,
                    "content_type": m.content_type,
                    "metadata": m.metadata,
                }
            )
        return result

    @staticmethod
    def _load_agent_class(entry_point: str) -> type:
        """Dynamically import a class from a dotted module path.

        Supports both 'module.path.ClassName' and 'module.path:ClassName' formats.
        """
        if ":" in entry_point:
            module_path, class_name = entry_point.rsplit(":", 1)
        elif "." in entry_point:
            module_path, class_name = entry_point.rsplit(".", 1)
        else:
            raise ValueError(
                f"Invalid entry_point format: '{entry_point}'. "
                "Expected 'module.path.ClassName' or 'module.path:ClassName'."
            )

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Cannot import agent module '{module_path}': {e}. "
                f"Ensure the package is installed in the server's environment."
            ) from e

        agent_class = getattr(module, class_name, None)
        if agent_class is None:
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{class_name}'. "
                f"Available: {[a for a in dir(module) if not a.startswith('_')]}"
            )
        return agent_class
