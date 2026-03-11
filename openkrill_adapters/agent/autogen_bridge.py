"""AutoGen Bridge — adapts Microsoft AutoGen agents to the OpenKrillAgent protocol.

Usage::

    # In your agent module (e.g. myapp/agents.py):
    from openkrill_adapters.agent.autogen_bridge import AutoGenBridge

    class MyAgent(AutoGenBridge):
        def build_agent(self):
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            model = OpenAIChatCompletionClient(model="gpt-4o")
            return AssistantAgent("assistant", model_client=model)

    # In OpenKrill agent config:
    # entry_point: "myapp.agents.MyAgent"
    # agent_framework: "autogen"
    # agent_config: {"model": "gpt-4o", "api_key": "sk-..."}

AutoGen is an optional dependency — this module only imports it when used.
Supports both AutoGen v0.4+ (autogen-agentchat) and legacy pyautogen.
"""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from openkrill_adapters.agent.protocol import OpenKrillAgent

logger = logging.getLogger(__name__)


class AutoGenBridge(OpenKrillAgent):
    """Base class for AutoGen-based agents in OpenKrill.

    Subclass this and implement build_agent() to return your AutoGen agent.
    The bridge handles message conversion, streaming, and lifecycle.

    Supports two API styles:
    - AutoGen v0.4+ (autogen-agentchat): agent.run() / agent.run_stream()
    - Legacy pyautogen: UserProxyAgent.initiate_chat()
    """

    def __init__(self, **kwargs: Any) -> None:
        self._config = kwargs
        self._agent: Any = None  # AutoGen agent instance

    @abstractmethod
    def build_agent(self) -> Any:
        """Build and return an AutoGen agent.

        This is called during initialize(). Return any AutoGen agent:
        - autogen_agentchat.agents.AssistantAgent (v0.4+)
        - autogen.AssistantAgent (legacy pyautogen)
        - Any object with .run() or .initiate_chat()

        Example (v0.4+)::

            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            def build_agent(self):
                model = OpenAIChatCompletionClient(
                    model=self._config.get("model", "gpt-4o"),
                    api_key=self._config.get("api_key", ""),
                )
                return AssistantAgent("assistant", model_client=model)
        """

    async def initialize(self) -> None:
        """Build the AutoGen agent."""
        try:
            import autogen_agentchat  # noqa: F401

            logger.info("Using AutoGen v0.4+ (autogen-agentchat)")
        except ImportError:
            try:
                import autogen  # noqa: F401

                logger.info("Using legacy AutoGen (pyautogen)")
            except ImportError as e:
                raise ImportError(
                    f"autogen is required for AutoGenBridge: {e}. "
                    "Install with: pip install autogen-agentchat autogen-ext "
                    "or: pip install pyautogen"
                ) from e

        self._agent = self.build_agent()
        logger.info(
            "AutoGen bridge initialized: agent=%s", type(self._agent).__name__
        )

    async def cleanup(self) -> None:
        """Release agent resources."""
        self._agent = None

    async def run(self, messages: list[dict]) -> str:
        """Run the agent with the conversation input."""
        if not self._agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        user_input = self._extract_last_user_message(messages)

        # Try AutoGen v0.4+ API first
        result = await self._try_v04_run(user_input)
        if result is not None:
            return result

        # Fallback to legacy pyautogen API
        return await self._try_legacy_run(user_input)

    async def run_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream output from the agent."""
        if not self._agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        user_input = self._extract_last_user_message(messages)

        # Try AutoGen v0.4+ streaming API
        streamed = False
        try:
            from autogen_agentchat.base import TaskResult
            from autogen_core import CancellationToken

            async for msg in self._agent.run_stream(
                task=user_input,
                cancellation_token=CancellationToken(),
            ):
                if isinstance(msg, TaskResult):
                    break  # Final result marker
                if hasattr(msg, "content") and msg.content:
                    yield str(msg.content)
                    streamed = True
            return
        except (ImportError, AttributeError):
            pass

        if not streamed:
            # No streaming support — fall back to full run
            result = await self.run(messages)
            yield result

    def health(self) -> bool:
        return self._agent is not None

    # ── Internal helpers ──

    @staticmethod
    def _extract_last_user_message(messages: list[dict]) -> str:
        """Extract the last user message from the conversation."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    async def _try_v04_run(self, user_input: str) -> str | None:
        """Try running with AutoGen v0.4+ API.

        Returns:
            The response string, or None if the v0.4+ API is not available.
        """
        try:
            from autogen_agentchat.base import TaskResult
            from autogen_core import CancellationToken

            result = await self._agent.run(
                task=user_input,
                cancellation_token=CancellationToken(),
            )
            if isinstance(result, TaskResult):
                # Get the last assistant message with content
                for msg in reversed(result.messages):
                    if hasattr(msg, "content") and msg.content:
                        return str(msg.content)
                return str(result)
            return str(result)
        except (ImportError, AttributeError):
            return None

    async def _try_legacy_run(self, user_input: str) -> str:
        """Fallback to legacy pyautogen API.

        Creates a temporary UserProxyAgent to drive the conversation.
        """
        try:
            loop = asyncio.get_event_loop()

            def _run_legacy() -> str:
                from autogen import UserProxyAgent

                user_proxy = UserProxyAgent(
                    "user_proxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=0,
                    code_execution_config=False,
                )
                user_proxy.initiate_chat(self._agent, message=user_input)
                last_msg = self._agent.last_message(user_proxy)
                if last_msg:
                    return last_msg.get("content", str(last_msg))
                return "No response generated."

            result = await loop.run_in_executor(None, _run_legacy)
            return str(result)
        except ImportError as e:
            raise RuntimeError(
                f"No compatible AutoGen API found: {e}. "
                "Install autogen-agentchat (v0.4+) or pyautogen (legacy)."
            ) from e
