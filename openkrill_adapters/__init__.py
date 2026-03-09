"""OpenKrill Adapter Ecosystem.

Pluggable adapters for connecting AI sources and social media platforms.
"""

from openkrill_adapters.base import AdapterCapability, AdapterMessage, AdapterResponse, BaseAdapter
from openkrill_adapters.registry import AdapterRegistry

__all__ = [
    "AdapterCapability",
    "AdapterMessage",
    "AdapterResponse",
    "AdapterRegistry",
    "BaseAdapter",
]
