"""SiteDriver abstraction — each AI chat website gets its own driver.

A SiteDriver encapsulates DOM selectors and interaction logic for a specific
site (ChatGPT, Gemini, DeepSeek, etc.). The WebAdapter delegates browser
interactions to the appropriate driver.
"""

from abc import ABC, abstractmethod

from playwright.async_api import Page


class SiteDriver(ABC):
    """Base class for site-specific Playwright automation."""

    @property
    @abstractmethod
    def site_name(self) -> str:
        """Unique identifier, e.g. 'chatgpt', 'gemini', 'deepseek'."""

    @property
    @abstractmethod
    def base_url(self) -> str:
        """The chat URL to navigate to."""

    @abstractmethod
    async def is_logged_in(self, page: Page) -> bool:
        """Check whether the current page shows an authenticated session."""

    @abstractmethod
    async def send_message(self, page: Page, text: str) -> None:
        """Type and submit a message in the chat input."""

    @abstractmethod
    async def wait_for_response(self, page: Page, timeout_ms: int = 120_000) -> str:
        """Wait for the AI response to finish streaming, return full text."""

    async def start_new_chat(self, page: Page) -> None:
        """Navigate to a fresh conversation. Override if site supports it."""
        await page.goto(self.base_url, wait_until="domcontentloaded")

    async def select_model(self, page: Page, model_variant: str) -> None:  # noqa: B027
        """Select a specific model variant. Override if site supports it."""


class SiteDriverRegistry:
    """Registry of available site drivers."""

    _drivers: dict[str, type[SiteDriver]] = {}

    @classmethod
    def register(cls, driver_class: type[SiteDriver]) -> type[SiteDriver]:
        """Register a driver class. Can be used as a decorator."""
        # Instantiate temporarily to read the site_name property
        # Instead, use a class-level attribute pattern
        name = driver_class.__dict__.get("_site_name") or driver_class.site_name.fget(None)  # type: ignore[arg-type]
        if name is None:
            raise ValueError(f"Cannot determine site_name for {driver_class.__name__}")
        cls._drivers[name] = driver_class
        return driver_class

    @classmethod
    def register_name(cls, name: str, driver_class: type[SiteDriver]) -> None:
        """Register a driver class with an explicit name."""
        cls._drivers[name] = driver_class

    @classmethod
    def get(cls, site_name: str) -> type[SiteDriver]:
        if site_name not in cls._drivers:
            available = ", ".join(sorted(cls._drivers.keys())) or "(none)"
            raise KeyError(f"Unknown site driver '{site_name}'. Available: {available}")
        return cls._drivers[site_name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._drivers.keys())
