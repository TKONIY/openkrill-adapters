"""Agent Adapter — connects agent frameworks (LangChain, CrewAI, AutoGen, custom) to OpenKrill.

Provides a generic protocol (OpenKrillAgent) that any agent framework can implement,
plus built-in bridges for popular frameworks.
"""

from openkrill_adapters.agent.adapter import AgentAdapter
from openkrill_adapters.agent.protocol import OpenKrillAgent

__all__ = ["AgentAdapter", "OpenKrillAgent"]
