"""OpenKrill Adapter Ecosystem.

Pluggable adapters for connecting AI sources and social media platforms.
"""

from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    StreamChunk,
    UsageInfo,
)
from openkrill_adapters.cli.session_adapter import SessionCliAdapter
from openkrill_adapters.registry import AdapterRegistry

__all__ = [
    "AdapterCapability",
    "AdapterMessage",
    "AdapterResponse",
    "AdapterRegistry",
    "BaseAdapter",
    "SessionCliAdapter",
    "StreamChunk",
    "UsageInfo",
]
