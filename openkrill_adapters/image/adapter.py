"""Image generation adapter — DALL-E and compatible APIs.

Receives text prompts via the standard adapter interface and returns
markdown-formatted image URLs. Supports OpenAI DALL-E 3, DALL-E 2,
and any OpenAI-compatible image generation API.

Config example:
    {
        "api_key": "sk-...",
        "model": "dall-e-3",           # or "dall-e-2"
        "base_url": "https://api.openai.com/v1",  # optional override
        "size": "1024x1024",           # "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
        "quality": "standard",         # "standard" or "hd" (DALL-E 3 only)
        "style": "vivid",             # "vivid" or "natural" (DALL-E 3 only)
    }
"""

import logging
from collections.abc import AsyncIterator

import httpx

from openkrill_adapters.base import (
    AdapterCapability,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    StreamChunk,
    UsageInfo,
)

logger = logging.getLogger(__name__)

# Supported sizes per model
VALID_SIZES = {
    "dall-e-2": ["256x256", "512x512", "1024x1024"],
    "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
}

DEFAULT_SIZE = "1024x1024"


class ImageAdapter(BaseAdapter):
    """Adapter for image generation APIs (DALL-E and compatible)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.api_key: str = config.get("api_key", "")
        self.model: str = config.get("model", "dall-e-3")
        self.base_url: str = config.get(
            "base_url", "https://api.openai.com/v1"
        ).rstrip("/")
        self.size: str = config.get("size", DEFAULT_SIZE)
        self.quality: str = config.get("quality", "standard")
        self.style: str = config.get("style", "vivid")
        self._client: httpx.AsyncClient | None = None

        if not self.api_key:
            raise RuntimeError(
                "ImageAdapter requires 'api_key' in adapter_config"
            )

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(
        self, messages: list[AdapterMessage]
    ) -> AdapterResponse:
        """Generate an image from the last user message prompt."""
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        prompt = self._extract_prompt(messages)
        if not prompt:
            return AdapterResponse(
                content="Please provide a description of the image "
                "you'd like me to generate.",
                content_type="markdown",
            )

        # Validate and fix size
        valid = VALID_SIZES.get(self.model, VALID_SIZES["dall-e-3"])
        size = self.size if self.size in valid else DEFAULT_SIZE

        # Build request
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "url",
        }
        if self.model == "dall-e-3":
            body["quality"] = self.quality
            body["style"] = self.style

        try:
            resp = await self._client.post(
                f"{self.base_url}/images/generations",
                json=body,
            )
            if resp.status_code != 200:
                error_msg = resp.text
                try:
                    error_data = resp.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", resp.text
                    )
                except Exception:
                    pass
                logger.error(
                    "Image generation failed (%d): %s",
                    resp.status_code,
                    error_msg,
                )
                return AdapterResponse(
                    content=f"Image generation failed: {error_msg}",
                    content_type="markdown",
                )

            data = resp.json()
            image_url = data["data"][0]["url"]
            revised_prompt = data["data"][0].get(
                "revised_prompt", prompt
            )

            # Return markdown with image
            content_parts = [
                f"![{prompt[:80]}]({image_url})",
                "",
            ]
            if revised_prompt != prompt:
                content_parts.append(
                    f"*Revised prompt: {revised_prompt}*"
                )

            return AdapterResponse(
                content="\n".join(content_parts),
                content_type="markdown",
                metadata={
                    "image_url": image_url,
                    "revised_prompt": revised_prompt,
                    "model": self.model,
                    "size": size,
                },
            )

        except httpx.TimeoutException:
            return AdapterResponse(
                content="Image generation timed out. "
                "Please try again with a simpler prompt.",
                content_type="markdown",
            )

    async def send_stream(
        self, messages: list[AdapterMessage]
    ) -> AsyncIterator[StreamChunk]:
        """Image generation doesn't support true streaming.

        We yield a "thinking" chunk while generating, then the final
        image as a "text" chunk.
        """
        yield StreamChunk(
            type="thinking",
            content="Generating image...",
        )

        response = await self.send(messages)

        yield StreamChunk(
            type="text",
            content=response.content,
        )

        # Yield usage info (image generation cost estimate)
        cost_map = {
            "dall-e-3": {"standard": 0.04, "hd": 0.08},
            "dall-e-2": {"256x256": 0.016, "512x512": 0.018},
        }
        model_costs = cost_map.get(self.model, {})
        estimated_cost = model_costs.get(
            self.quality, model_costs.get(self.size, 0.04)
        )

        yield StreamChunk(
            type="usage",
            usage=UsageInfo(
                input_tokens=0,
                output_tokens=0,
                model=self.model,
                provider="openai",
                metadata={
                    "estimated_cost": estimated_cost,
                    "image_count": 1,
                },
            ),
        )

    async def health_check(self) -> bool:
        """Check API connectivity."""
        if not self._client:
            return False
        try:
            resp = await self._client.get(
                f"{self.base_url}/models",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def adapter_type(self) -> str:
        return "image"

    @property
    def max_capability(self) -> AdapterCapability:
        return AdapterCapability.L1_RICH

    @staticmethod
    def _extract_prompt(messages: list[AdapterMessage]) -> str:
        """Extract the image generation prompt from messages.

        Uses the last user message as the prompt.
        """
        for msg in reversed(messages):
            if msg.role == "user" and msg.content.strip():
                return msg.content.strip()
        return ""
