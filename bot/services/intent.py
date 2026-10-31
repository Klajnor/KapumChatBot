"""AI-based intent classification for messages that carry an image.

Both the chat and the guest handlers ask the model — instead of guessing
from keywords — what to do with an attached photo: analyze it, generate a
new image from it, or enhance the photo itself. One cheap single-word call
(``temperature=0``) decides; any failure falls back to ``"analyze"``, the
behaviour that existed before intent routing.

Все production-вызовы должны предварительно резервировать отдельный суточный
бюджет через UserService.consume_photo_intent_budget (включая guest и альбомы).
"""
from __future__ import annotations

import logging

from bot.services.ai_client import ai_client

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a message that carries an image. "
    "Answer with EXACTLY ONE LINE, starting with one word:\n"
    "- 'generate' — the user wants a NEW image made from the reference "
    "(draw something similar, in another style, reimagine, based on this);\n"
    "- 'enhance' — the user wants THIS photo improved or processed: raise "
    "quality, upscale/enlarge resolution, remove noise/grain, sharpen, fix "
    "colors or brightness, restore an old or faded photo, retouch;\n"
    "- 'analyze' — the user wants to know what is in the image: describe, "
    "explain, read text from it, answer a question about it.\n"
    "For 'enhance' only, you may append ':' and a comma-separated list of "
    "operations from: upscale, denoise, sharpen, color, contrast, "
    "brightness, restore. Examples: 'analyze'. 'generate'. "
    "'enhance: upscale, denoise'. Nothing else."
)

# Vision prompt used to describe a reference image before generation.
VISION_FOR_GEN_PROMPT = (
    "Describe this image in detail: subject, style, colours, lighting, "
    "composition, mood. Output only the description — no introduction, "
    "no summary, no commentary."
)


async def detect_intent(caption: str, model: str) -> tuple[str, list[str]]:
    """Classify the user's intent for an attached image via the AI.

    Returns ``(intent, operations)``: ``intent`` is one of ``"analyze"``,
    ``"generate"`` or ``"enhance"``; ``operations`` (only meaningful for
    ``"enhance"``) is the list of processing steps the model extracted from
    the request (see ``bot.services.image_tools`` and
    ``AIClient.enhance_image_via_ai``). Falls back to ``("analyze", [])``
    on an empty caption or any error.
    """
    if not caption or not caption.strip():
        return "analyze", []

    try:
        result = await ai_client.chat_completion_result(
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": caption},
            ],
            model=model,
            temperature=0.0,
            max_tokens=40,
        )
    except Exception:
        logger.warning("Intent detection failed; defaulting to analyze", exc_info=True)
        return "analyze", []

    answer = (result.content or "").strip().lower()
    head, _, tail = answer.partition(":")
    if "generate" in head:
        return "generate", []
    if "enhance" in head:
        operations = [token.strip() for token in tail.split(",") if token.strip()]
        return "enhance", operations
    return "analyze", []


def generation_prompt(caption: str, vision_description: str | None = None) -> str:
    """Build the image-generation prompt from the user caption and an optional
    vision-derived description of the reference image."""
    base = caption.strip()
    if vision_description:
        base = (
            f"Reference image description: {vision_description}\n\n"
            f"User request: {base}"
        )
    return base
