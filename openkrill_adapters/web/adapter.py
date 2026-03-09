"""Web Adapter — drives AI chat websites via headless Playwright browser.

Config schema:
    site: str           — Site driver name ("chatgpt", "gemini", "deepseek")
    cookies: list[dict] — Browser cookies for authentication (exported from browser)
    model_variant: str  — Optional model to select (e.g. "gpt-4o")
    headless: bool      — Run browser headless (default: True)
    response_timeout: int — Max seconds to wait for AI response (default: 120)
"""

import logging
from collections.abc import AsyncIterator

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

# Import sites to trigger registration
import openkrill_adapters.web.sites  # noqa: F401
from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
)
from openkrill_adapters.web.site_driver import SiteDriver, SiteDriverRegistry

logger = logging.getLogger(__name__)


class WebAdapter(BaseAdapter):
    """Adapter that automates AI chat websites using Playwright."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._site_name: str = config.get("site", "chatgpt")
        self._model_variant: str = config.get("model_variant", "")
        headless_raw = config.get("headless", True)
        self._headless: bool = (
            headless_raw if isinstance(headless_raw, bool) else str(headless_raw).lower() != "false"
        )
        self._response_timeout: int = config.get("response_timeout", 120)

        # Cookies can be a JSON string or list of dicts
        raw_cookies = config.get("cookies", [])
        if isinstance(raw_cookies, str) and raw_cookies.strip():
            import json

            try:
                raw_cookies = json.loads(raw_cookies)
            except json.JSONDecodeError:
                logger.warning("Invalid cookies JSON, ignoring")
                raw_cookies = []
        self._cookies: list[dict] = raw_cookies if isinstance(raw_cookies, list) else []

        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._driver: SiteDriver | None = None

    @property
    def adapter_type(self) -> str:
        return "web"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L0_TEXT

    async def connect(self) -> None:
        """Launch browser, restore cookies, navigate to site."""
        driver_class = SiteDriverRegistry.get(self._site_name)
        self._driver = driver_class()

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)

        # Create context with cookies for persistent login
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )

        # Restore auth cookies
        if self._cookies:
            await self._context.add_cookies(self._cookies)

        self._page = await self._context.new_page()

        # Navigate to site
        await self._driver.start_new_chat(self._page)

        # Verify login
        if not await self._driver.is_logged_in(self._page):
            raise RuntimeError(
                f"Not logged in to {self._site_name}. "
                "Please provide valid cookies in adapter_config."
            )

        # Select model if specified
        if self._model_variant:
            await self._driver.select_model(self._page, self._model_variant)

        logger.info("WebAdapter connected to %s", self._site_name)

    async def disconnect(self) -> None:
        """Close browser and clean up."""
        if self._page:
            await self._page.close()
            self._page = None
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._driver = None
        logger.info("WebAdapter disconnected")

    async def send(self, messages: list[AdapterMessage]) -> AdapterResponse:
        """Send the last user message to the AI chat site and return the response.

        Unlike API adapters, web chat sites don't accept full conversation history.
        We only send the latest user message.
        """
        if not self._page or not self._driver:
            raise RuntimeError("WebAdapter not connected — call connect() first")

        # Extract last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        if not last_user_msg:
            return AdapterResponse(content="(no user message to send)")

        # Send message and wait for response
        await self._driver.send_message(self._page, last_user_msg)
        timeout_ms = self._response_timeout * 1000
        response_text = await self._driver.wait_for_response(self._page, timeout_ms=timeout_ms)

        return AdapterResponse(content=response_text, content_type="text")

    async def send_stream(self, messages: list[AdapterMessage]) -> AsyncIterator[str]:
        """Web adapter doesn't support true streaming — returns full response as one chunk."""
        response = await self.send(messages)
        yield response.content

    async def health_check(self) -> bool:
        """Check if browser session is still alive and logged in."""
        if not self._page or not self._driver:
            return False
        try:
            return await self._driver.is_logged_in(self._page)
        except Exception:
            return False
