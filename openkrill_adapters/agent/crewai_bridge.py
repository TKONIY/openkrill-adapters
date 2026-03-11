"""CrewAI Bridge — adapts CrewAI crews to the OpenKrillAgent protocol.

Usage::

    # In your agent module (e.g. myapp/agents.py):
    from openkrill_adapters.agent.crewai_bridge import CrewAIBridge

    class MyCrewAgent(CrewAIBridge):
        def build_crew(self):
            from crewai import Agent, Task, Crew
            researcher = Agent(role="Researcher", ...)
            task = Task(description="{input}", agent=researcher)
            return Crew(agents=[researcher], tasks=[task])

    # In OpenKrill agent config:
    # entry_point: "myapp.agents.MyCrewAgent"
    # agent_framework: "crewai"
    # agent_config: {"openai_api_key": "sk-..."}

CrewAI is an optional dependency — this module only imports it when used.
"""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from openkrill_adapters.agent.protocol import OpenKrillAgent

logger = logging.getLogger(__name__)


class CrewAIBridge(OpenKrillAgent):
    """Base class for CrewAI-based agents in OpenKrill.

    Subclass this and implement build_crew() to return your CrewAI Crew.
    The bridge handles message conversion, async execution, and lifecycle.

    CrewAI crews run synchronously, so the bridge wraps them in an executor
    for async compatibility.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._config = kwargs
        self._crew: Any = None  # crewai.Crew

    @abstractmethod
    def build_crew(self) -> Any:
        """Build and return a crewai.Crew instance.

        This is called during initialize(). Return a Crew configured with
        agents and tasks. The bridge will call crew.kickoff() with the
        conversation input.

        Example::

            from crewai import Agent, Crew, Task

            def build_crew(self):
                researcher = Agent(
                    role="Researcher",
                    goal="Find information on the given topic",
                    backstory="You are an experienced researcher.",
                )
                task = Task(
                    description="Research: {input}",
                    expected_output="Detailed findings",
                    agent=researcher,
                )
                return Crew(agents=[researcher], tasks=[task])
        """

    async def initialize(self) -> None:
        """Build the CrewAI crew."""
        try:
            import crewai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"crewai is required for CrewAIBridge: {e}. "
                "Install with: pip install crewai"
            ) from e

        self._crew = self.build_crew()
        logger.info("CrewAI bridge initialized: crew=%s", type(self._crew).__name__)

    async def cleanup(self) -> None:
        """Release crew resources."""
        self._crew = None

    async def run(self, messages: list[dict]) -> str:
        """Run the crew with the conversation input.

        Extracts the last user message as the primary input, and builds
        a context string from recent conversation history.
        """
        if not self._crew:
            raise RuntimeError("Crew not initialized. Call initialize() first.")

        user_input, context = self._prepare_input(messages)

        inputs: dict[str, str] = {"input": user_input}
        if context:
            inputs["context"] = context

        # CrewAI kickoff is synchronous — run in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._crew.kickoff(inputs=inputs)
        )

        return self._extract_content(result)

    async def run_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream output from the crew.

        CrewAI does not natively support streaming, so this falls back
        to run() and yields the complete result.
        """
        result = await self.run(messages)
        yield result

    def health(self) -> bool:
        return self._crew is not None

    # ── Internal helpers ──

    @staticmethod
    def _prepare_input(messages: list[dict]) -> tuple[str, str]:
        """Extract user input and context from OpenKrill messages.

        Returns:
            A tuple of (last_user_message, context_string).
        """
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break

        # Build context from earlier messages (exclude the last user message)
        context_parts: list[str] = []
        for msg in messages[:-1]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                context_parts.append(f"[{role}]: {content}")

        # Keep last 10 messages to avoid overwhelming the context
        context = "\n".join(context_parts[-10:])
        return user_input, context

    @staticmethod
    def _extract_content(result: Any) -> str:
        """Extract string content from a CrewAI result.

        Handles CrewOutput (has .raw attribute) or plain strings.
        """
        if hasattr(result, "raw"):
            return str(result.raw)
        return str(result)
