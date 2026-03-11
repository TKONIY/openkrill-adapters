"""LangChain Bridge — adapts LangChain agents/chains to the OpenKrillAgent protocol.

Usage::

    # In your agent module (e.g. myapp/agents.py):
    from langchain.agents import create_react_agent
    from openkrill_adapters.agent.langchain_bridge import LangChainBridge

    class MyAgent(LangChainBridge):
        def build_chain(self):
            # Return any LangChain Runnable (chain, agent, etc.)
            return create_react_agent(llm, tools, prompt)

    # In OpenKrill agent config:
    # entry_point: "myapp.agents.MyAgent"
    # agent_framework: "langchain"
    # agent_config: {"temperature": 0.7}

LangChain is an optional dependency — this module only imports it when used.
"""

import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from openkrill_adapters.agent.protocol import OpenKrillAgent

logger = logging.getLogger(__name__)


class LangChainBridge(OpenKrillAgent):
    """Base class for LangChain-based agents in OpenKrill.

    Subclass this and implement build_chain() to return your LangChain Runnable.
    The bridge handles message format conversion and streaming.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._config = kwargs
        self._chain: Any = None  # LangChain Runnable

    @abstractmethod
    def build_chain(self) -> Any:
        """Build and return a LangChain Runnable (chain, agent, or AgentExecutor).

        This is called during initialize(). Return any object that supports
        .ainvoke() and optionally .astream().

        Example::
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate

            def build_chain(self):
                llm = ChatOpenAI(model="gpt-4", temperature=self._config.get("temperature", 0))
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant."),
                    ("placeholder", "{messages}"),
                ])
                return prompt | llm
        """

    async def initialize(self) -> None:
        """Build the LangChain chain."""
        try:
            self._chain = self.build_chain()
            logger.info("LangChain bridge initialized: chain=%s", type(self._chain).__name__)
        except ImportError as e:
            raise ImportError(
                f"LangChain is required for LangChainBridge: {e}. "
                "Install with: pip install langchain langchain-openai"
            ) from e

    async def cleanup(self) -> None:
        """Release chain resources."""
        self._chain = None

    async def run(self, messages: list[dict]) -> str:
        """Invoke the chain with the message history."""
        if not self._chain:
            raise RuntimeError("Chain not initialized. Call initialize() first.")

        chain_input = self._prepare_input(messages)
        result = await self._chain.ainvoke(chain_input)
        return self._extract_content(result)

    async def run_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream output from the chain if supported."""
        if not self._chain:
            raise RuntimeError("Chain not initialized. Call initialize() first.")

        chain_input = self._prepare_input(messages)

        # Try astream first; fall back to ainvoke
        if hasattr(self._chain, "astream"):
            async for event in self._chain.astream(chain_input):
                content = self._extract_content(event)
                if content:
                    yield content
        else:
            result = await self._chain.ainvoke(chain_input)
            yield self._extract_content(result)

    def _prepare_input(self, messages: list[dict]) -> dict:
        """Convert OpenKrill messages to LangChain input format.

        Override this if your chain expects a different input schema.
        Default: passes messages as {"messages": [...]} for ChatPromptTemplate,
        or {"input": last_message} for simple chains.
        """
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            lc_messages = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            last_input = messages[-1].get("content", "") if messages else ""
            return {"messages": lc_messages, "input": last_input}
        except ImportError:
            # Fallback: just pass raw messages
            last_input = messages[-1].get("content", "") if messages else ""
            return {"messages": messages, "input": last_input}

    @staticmethod
    def _extract_content(result: Any) -> str:
        """Extract string content from a LangChain result.

        Handles: str, BaseMessage, dict with 'output'/'content', or str(result).
        """
        if isinstance(result, str):
            return result

        # LangChain BaseMessage
        if hasattr(result, "content"):
            content = result.content
            return content if isinstance(content, str) else str(content)

        # AgentExecutor output dict
        if isinstance(result, dict):
            return str(result.get("output", result.get("content", result)))

        return str(result)

    def health(self) -> bool:
        return self._chain is not None
