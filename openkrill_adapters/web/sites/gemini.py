"""Gemini site driver — Playwright automation for gemini.google.com."""

import asyncio
import logging

from playwright.async_api import Page

from openkrill_adapters.web.site_driver import SiteDriver, SiteDriverRegistry

logger = logging.getLogger(__name__)


class GeminiDriver(SiteDriver):
    """Automate Google Gemini web interface."""

    # Selectors — Gemini uses rich-text editor with contenteditable
    INPUT_SELECTOR = '.ql-editor[contenteditable="true"]'
    SEND_BUTTON_SELECTOR = 'button[aria-label="Send message"]'
    # Gemini shows a loading indicator while generating
    LOADING_SELECTOR = ".loading-indicator, .response-streaming"
    # Response container — Gemini renders markdown in message-content divs
    RESPONSE_SELECTOR = "message-content"

    # Fallback selector lists for resilience against DOM changes
    INPUT_SELECTORS = [
        '.ql-editor[contenteditable="true"]',
        'div[contenteditable="true"]',
        'rich-textarea [contenteditable="true"]',
    ]
    SEND_SELECTORS = [
        'button[aria-label="Send message"]',
        'button[aria-label="Send"]',
        'button.send-button',
    ]
    LOADING_SELECTORS = [
        ".loading-indicator",
        ".response-streaming",
        'mat-progress-bar',
    ]
    RESPONSE_SELECTORS = [
        "message-content",
        ".model-response-text",
        ".response-container .markdown",
    ]

    @property
    def site_name(self) -> str:
        return "gemini"

    @property
    def base_url(self) -> str:
        return "https://gemini.google.com/app"

    async def is_logged_in(self, page: Page) -> bool:
        """Check if user is logged in by looking for the chat input."""
        try:
            await self._find_element(page, self.INPUT_SELECTORS, timeout=5000)
            return True
        except Exception:
            return False

    async def send_message(self, page: Page, text: str) -> None:
        """Type message into Gemini and click send."""
        editor = await self._find_element(page, self.INPUT_SELECTORS)
        await editor.click()
        # Gemini uses a Quill-like rich text editor
        await editor.fill(text)
        await asyncio.sleep(0.3)

        send_btn = await self._find_element(page, self.SEND_SELECTORS)
        await send_btn.click()

    async def wait_for_response(self, page: Page, timeout_ms: int = 120_000) -> str:
        """Wait for Gemini to finish generating, return the last response."""
        # Wait for loading to start
        try:
            await self._find_element(page, self.LOADING_SELECTORS, timeout=10_000)
        except Exception:
            logger.debug("Loading indicator not detected — response may be instant")

        # Wait for loading to finish
        try:
            await page.wait_for_selector(
                ", ".join(self.LOADING_SELECTORS), state="detached", timeout=timeout_ms
            )
        except Exception as e:
            logger.warning("Timed out waiting for Gemini response: %s", e)

        # Extract the last response (try each response selector)
        for selector in self.RESPONSE_SELECTORS:
            messages = page.locator(selector)
            count = await messages.count()
            if count > 0:
                last_msg = messages.nth(count - 1)
                return (await last_msg.inner_text()).strip()

        return "(no response received)"

    async def start_new_chat(self, page: Page) -> None:
        """Navigate to Gemini for a fresh conversation."""
        await page.goto(self.base_url, wait_until="domcontentloaded")
        await self._find_element(page, self.INPUT_SELECTORS, timeout=15_000)


SiteDriverRegistry.register_name("gemini", GeminiDriver)
