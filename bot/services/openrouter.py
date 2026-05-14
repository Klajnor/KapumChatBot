from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any

import aiohttp

from bot.config import config

logger = logging.getLogger(__name__)

# Available models for selection
AVAILABLE_MODELS: list[dict[str, str]] = [
    {"id": "openai/gpt-4o", "name": "GPT-4o"},
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini"},
    {"id": "openai/gpt-4-turbo", "name": "GPT-4 Turbo"},
    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
    {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku"},
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3"},
    {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1"},
    {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash"},
    {"id": "google/gemini-pro", "name": "Gemini Pro"},
    {"id": "mistralai/mistral-large", "name": "Mistral Large"},
    {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B"},
    {"id": "qwen/qwen-2.5-72b-instruct", "name": "Qwen 2.5 72B"},
]

# Models that support vision (image input)
VISION_MODELS: set[str] = {
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-2.0-flash-001",
    "google/gemini-pro",
}


class OpenRouterClient:
    """Async HTTP client for OpenRouter API."""

    def __init__(self) -> None:
        self.base_url = config.openrouter_base_url.rstrip("/")
        self.api_key = config.openrouter_api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/chatbot",
            "X-Title": "Telegram AI Bot",
        }

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str = "openai/gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str | None:
        """Send a chat completion request to OpenRouter.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model identifier string.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            The response text content, or None on failure.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with aiohttp.ClientSession(
                headers=self.headers, timeout=aiohttp.ClientTimeout(total=120)
            ) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions", json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            "OpenRouter API error (status %d): %s",
                            resp.status,
                            error_text,
                        )
                        return None

                    data = await resp.json()
                    choice = data.get("choices", [{}])[0]
                    content = choice.get("message", {}).get("content")
                    return content

        except asyncio.TimeoutError:
            logger.error("OpenRouter API request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error("OpenRouter API request failed: %s", e)
            return None

    async def chat_completion_with_image(
        self,
        text: str,
        image_bytes: bytes,
        messages: list[dict[str, Any]],
        model: str = "openai/gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str | None:
        """Send a chat completion with an image to OpenRouter.

        Args:
            text: Text prompt accompanying the image.
            image_bytes: Raw image bytes (JPEG/PNG).
            messages: Previous conversation messages for context.
            model: Vision-capable model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            The response text content, or None on failure.
        """
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Determine image MIME type
        mime_type = "image/jpeg"
        if image_bytes[:4] == b"\x89PNG":
            mime_type = "image/png"
        elif image_bytes[:2] == b"GI":
            mime_type = "image/gif"
        elif image_bytes[:2] == b"BM":
            mime_type = "image/bmp"
        elif image_bytes[:4] == b"\xff\xd8\xff\xe0":
            mime_type = "image/jpeg"

        data_url = f"data:{mime_type};base64,{image_base64}"

        # Build the user message with text + image
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            },
        ]

        # Append to existing messages
        full_messages = list(messages)
        full_messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with aiohttp.ClientSession(
                headers=self.headers, timeout=aiohttp.ClientTimeout(total=180)
            ) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions", json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            "OpenRouter vision API error (status %d): %s",
                            resp.status,
                            error_text,
                        )
                        return None

                    data = await resp.json()
                    choice = data.get("choices", [{}])[0]
                    content = choice.get("message", {}).get("content")
                    return content

        except asyncio.TimeoutError:
            logger.error("OpenRouter vision API request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error("OpenRouter vision API request failed: %s", e)
            return None

    @staticmethod
    def is_vision_model(model: str) -> bool:
        """Check if the given model supports vision (image input)."""
        return model in VISION_MODELS


openrouter_client = OpenRouterClient()