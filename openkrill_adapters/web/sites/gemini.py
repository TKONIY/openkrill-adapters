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

    @property
    def site_name(self) -> str:
        return "gemini"

    @property
    def base_url(self) -> str:
        return "https://gemini.google.com/app"

    async def is_logged_in(self, page: Page) -> bool:
        """Check if user is logged in by looking for the chat input."""
        try:
            await page.wait_for_selector(self.INPUT_SELECTOR, timeout=5000)
            return True
        except Exception:
            return False

    async def send_message(self, page: Page, text: str) -> None:
        """Type message into Gemini and click send."""
        editor = page.locator(self.INPUT_SELECTOR)
        await editor.click()
        # Gemini uses a Quill-like rich text editor
        await editor.fill(text)
        await asyncio.sleep(0.3)

        send_btn = page.locator(self.SEND_BUTTON_SELECTOR)
        await send_btn.click()

    async def wait_for_response(self, page: Page, timeout_ms: int = 120_000) -> str:
        """Wait for Gemini to finish generating, return the last response."""
        # Wait for loading to start
        try:
            await page.wait_for_selector(self.LOADING_SELECTOR, timeout=10_000)
        except Exception:
            logger.debug("Loading indicator not detected — response may be instant")

        # Wait for loading to finish
        try:
            await page.wait_for_selector(
                self.LOADING_SELECTOR, state="detached", timeout=timeout_ms
            )
        except Exception as e:
            logger.warning("Timed out waiting for Gemini response: %s", e)

        # Extract the last response
        messages = page.locator(self.RESPONSE_SELECTOR)
        count = await messages.count()
        if count == 0:
            return "(no response received)"

        last_msg = messages.nth(count - 1)
        return (await last_msg.inner_text()).strip()

    async def start_new_chat(self, page: Page) -> None:
        """Navigate to Gemini for a fresh conversation."""
        await page.goto(self.base_url, wait_until="domcontentloaded")
        await page.wait_for_selector(self.INPUT_SELECTOR, timeout=15_000)


SiteDriverRegistry.register_name("gemini", GeminiDriver)
