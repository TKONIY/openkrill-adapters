"""DeepSeek site driver — Playwright automation for chat.deepseek.com."""

import asyncio
import logging

from playwright.async_api import Page

from openkrill_adapters.web.site_driver import SiteDriver, SiteDriverRegistry

logger = logging.getLogger(__name__)


class DeepSeekDriver(SiteDriver):
    """Automate DeepSeek web chat interface."""

    # Selectors — DeepSeek uses a textarea input
    INPUT_SELECTOR = "textarea#chat-input, textarea[placeholder]"
    SEND_BUTTON_SELECTOR = 'div[class*="inputBox"] button:last-child'
    # DeepSeek shows a stop button during generation
    STREAMING_SELECTOR = 'div[class*="stopButton"], button[class*="stop"]'
    # Response messages
    RESPONSE_SELECTOR = 'div[class*="markdownContent"], div[class*="assistant"]'

    @property
    def site_name(self) -> str:
        return "deepseek"

    @property
    def base_url(self) -> str:
        return "https://chat.deepseek.com"

    async def is_logged_in(self, page: Page) -> bool:
        """Check if user is logged in by looking for the chat input."""
        try:
            await page.wait_for_selector(self.INPUT_SELECTOR, timeout=5000)
            return True
        except Exception:
            return False

    async def send_message(self, page: Page, text: str) -> None:
        """Type message into DeepSeek and submit."""
        textarea = page.locator(self.INPUT_SELECTOR).first
        await textarea.click()
        await textarea.fill(text)
        await asyncio.sleep(0.3)

        # Try clicking send button, fall back to Enter key
        send_btn = page.locator(self.SEND_BUTTON_SELECTOR)
        if await send_btn.count() > 0:
            await send_btn.click()
        else:
            await textarea.press("Enter")

    async def wait_for_response(self, page: Page, timeout_ms: int = 120_000) -> str:
        """Wait for DeepSeek to finish streaming, return the last response."""
        # Wait for streaming to start
        try:
            await page.wait_for_selector(self.STREAMING_SELECTOR, timeout=10_000)
        except Exception:
            logger.debug("Stop button not detected — response may be instant")

        # Wait for streaming to finish
        try:
            await page.wait_for_selector(
                self.STREAMING_SELECTOR, state="detached", timeout=timeout_ms
            )
        except Exception as e:
            logger.warning("Timed out waiting for DeepSeek response: %s", e)

        # Small delay for final render
        await asyncio.sleep(0.5)

        # Extract the last response
        messages = page.locator(self.RESPONSE_SELECTOR)
        count = await messages.count()
        if count == 0:
            return "(no response received)"

        last_msg = messages.nth(count - 1)
        return (await last_msg.inner_text()).strip()

    async def start_new_chat(self, page: Page) -> None:
        """Navigate to DeepSeek for a fresh conversation."""
        await page.goto(self.base_url, wait_until="domcontentloaded")
        await page.wait_for_selector(self.INPUT_SELECTOR, timeout=15_000)

    async def select_model(self, page: Page, model_variant: str) -> None:
        """DeepSeek supports model toggle (DeepSeek-V3 vs DeepSeek-R1)."""
        if not model_variant:
            return
        logger.info("DeepSeek model selection: %s (manual toggle required)", model_variant)


SiteDriverRegistry.register_name("deepseek", DeepSeekDriver)
