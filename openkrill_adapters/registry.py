"""Adapter registry — discovery, creation, and lifecycle management.

Inspired by OpenClaw's src/channels/registry.ts pattern.
Supports both programmatic registration and entry-point discovery.
"""

import importlib.metadata
import logging
from typing import ClassVar

from openkrill_adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry of adapter types.

    Usage:
        # Register adapter classes programmatically
        AdapterRegistry.register("api", ApiAdapter)

        # Or discover from installed packages (entry_points)
        AdapterRegistry.discover()

        # Create an adapter instance from DB config
        adapter = AdapterRegistry.create("api", {"provider": "anthropic", ...})
        await adapter.connect()
    """

    _adapters: ClassVar[dict[str, type[BaseAdapter]]] = {}

    @classmethod
    def register(cls, adapter_type: str, adapter_class: type[BaseAdapter]) -> None:
        """Register an adapter class for a given type."""
        if adapter_type in cls._adapters:
            logger.warning("Overriding adapter type '%s'", adapter_type)
        cls._adapters[adapter_type] = adapter_class
        logger.info("Registered adapter type '%s' -> %s", adapter_type, adapter_class.__name__)

    @classmethod
    def create(cls, adapter_type: str, config: dict) -> BaseAdapter:
        """Create an adapter instance from type and config."""
        adapter_class = cls._adapters.get(adapter_type)
        if not adapter_class:
            available = ", ".join(cls._adapters.keys()) or "(none)"
            raise ValueError(f"Unknown adapter type '{adapter_type}'. Available: {available}")
        return adapter_class(config)

    @classmethod
    def discover(cls) -> None:
        """Discover and register adapters from installed packages via entry points."""
        for ep in importlib.metadata.entry_points(group="openkrill.adapters"):
            try:
                adapter_class = ep.load()
                cls.register(ep.name, adapter_class)
            except Exception:
                logger.exception("Failed to load adapter entry point '%s'", ep.name)

    @classmethod
    def available_types(cls) -> list[str]:
        """List all registered adapter types."""
        return list(cls._adapters.keys())

    @classmethod
    def is_registered(cls, adapter_type: str) -> bool:
        """Check if an adapter type is registered."""
        return adapter_type in cls._adapters

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Mainly for testing."""
        cls._adapters.clear()
