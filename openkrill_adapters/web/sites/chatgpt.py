"""ChatGPT site driver — Playwright automation for chat.openai.com."""

import asyncio
import logging

from playwright.async_api import Page

from openkrill_adapters.web.site_driver import SiteDriver, SiteDriverRegistry

logger = logging.getLogger(__name__)


class ChatGPTDriver(SiteDriver):
    """Automate ChatGPT web interface."""

    # Selectors — update these when ChatGPT changes its DOM
    INPUT_SELECTOR = "#prompt-textarea"
    SEND_BUTTON_SELECTOR = 'button[data-testid="send-button"]'
    # The streaming indicator disappears when response is complete
    STREAMING_SELECTOR = 'button[data-testid="stop-button"]'
    # Assistant messages container
    RESPONSE_SELECTOR = '[data-message-author-role="assistant"]'

    @property
    def site_name(self) -> str:
        return "chatgpt"

    @property
    def base_url(self) -> str:
        return "https://chatgpt.com"

    async def is_logged_in(self, page: Page) -> bool:
        """Check if user is logged in by looking for the chat input."""
        try:
            await page.wait_for_selector(self.INPUT_SELECTOR, timeout=5000)
            return True
        except Exception:
            return False

    async def send_message(self, page: Page, text: str) -> None:
        """Type message into ChatGPT and click send."""
        textarea = page.locator(self.INPUT_SELECTOR)
        await textarea.click()
        # Use fill for reliability (handles contenteditable divs too)
        await textarea.fill(text)
        # Small delay for UI to register the input
        await asyncio.sleep(0.3)

        # Click send button
        send_btn = page.locator(self.SEND_BUTTON_SELECTOR)
        await send_btn.click()

    async def wait_for_response(self, page: Page, timeout_ms: int = 120_000) -> str:
        """Wait for ChatGPT to finish streaming, return the last assistant message."""
        # Wait for streaming to start (stop button appears)
        try:
            await page.wait_for_selector(self.STREAMING_SELECTOR, timeout=10_000)
        except Exception:
            logger.debug("Stop button not found — response may have completed instantly")

        # Wait for streaming to finish (stop button disappears)
        try:
            await page.wait_for_selector(
                self.STREAMING_SELECTOR, state="detached", timeout=timeout_ms
            )
        except Exception as e:
            logger.warning("Timed out waiting for response to finish: %s", e)

        # Extract the last assistant message
        messages = page.locator(self.RESPONSE_SELECTOR)
        count = await messages.count()
        if count == 0:
            return "(no response received)"

        last_msg = messages.nth(count - 1)
        return (await last_msg.inner_text()).strip()

    async def start_new_chat(self, page: Page) -> None:
        """Navigate to ChatGPT homepage for a fresh conversation."""
        await page.goto(self.base_url, wait_until="domcontentloaded")
        await page.wait_for_selector(self.INPUT_SELECTOR, timeout=15_000)

    async def select_model(self, page: Page, model_variant: str) -> None:
        """Select a model variant (e.g. 'gpt-4o', 'o1-preview').

        ChatGPT's model selector changes frequently. This is a best-effort
        implementation that may need updating.
        """
        if not model_variant:
            return
        logger.info("Model selection requested: %s (not yet automated)", model_variant)


# Register the driver
SiteDriverRegistry.register_name("chatgpt", ChatGPTDriver)
