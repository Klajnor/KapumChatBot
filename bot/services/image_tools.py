"""Local photo enhancement with Pillow: the photo itself is improved.

Unlike generation (a new picture from a reference), enhancement keeps the
content untouched and only fixes quality: upscale, denoise, sharpen, colour,
contrast, brightness, restoration. It is fast, costs no API calls and works
with any model — the intent classifier in ``bot.handlers.chat`` decides
*whether* to enhance and which operations the user asked for; this module
applies them off the event loop.
"""
from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Sequence

from bot.services.pillow_limits import MAX_IMAGE_PIXELS, configure_pillow_limits

logger = logging.getLogger(__name__)

# Longest side of the enhanced result; keeps the upload Telegram-friendly.
MAX_ENHANCED_SIDE = 4096
# Only upscale photos whose longest side is below this: big images already
# carry enough detail, and 2x more of them is slow, heavy and invisible.
UPSCALE_TRIGGER_SIDE = 1600
# Decompression-bomb guard for decoded pixel counts.
MAX_INPUT_PIXELS = MAX_IMAGE_PIXELS

# Operations the classifier may request (single source of truth for the
# intent prompt in bot/handlers/chat.py — keep the two lists in sync).
OPERATIONS = (
    "upscale",    # enlarge 2x (LANCZOS), capped at MAX_ENHANCED_SIDE
    "denoise",    # median filter: speckle/grain cleanup
    "sharpen",    # unsharp mask: recover edge detail
    "color",      # saturation boost
    "contrast",   # auto-levels
    "brightness", # gentle lightening
    "restore",    # full old-photo pipeline: denoise + contrast + colour + sharpen
)

# Conservative pipeline when the model requests "enhance" without details.
DEFAULT_OPERATIONS = ("contrast", "color", "sharpen")

# Canonical application order: tone first, then noise, then geometry, then
# detail — denoise before sharpen, restore expands into its own steps.
_ORDER = ("contrast", "brightness", "denoise", "upscale", "color", "sharpen")

# Lowercase substrings the classifier or user may produce per operation.
_ALIASES: dict[str, tuple[str, ...]] = {
    "upscale": ("upscal", "upscale", "resol", "enlarg", "biggen", "увелич", "разрешени"),
    "denoise": ("denois", "noise", "grain", "шум", "помех"),
    "sharpen": ("sharpen", "sharp", "focus", "резк", "четк", "фокус"),
    "color": ("color", "colour", "saturat", "vibranc", "цвет", "насыщ"),
    "contrast": ("contrast", "контраст"),
    "brightness": ("bright", "luminos", "ярк", "светл"),
    "restore": ("restor", "retro", "faded", "old photo", "реставрац", "стар", "выцв"),
}


def _canonical(token: str) -> str | None:
    """Map one free-form word to a known operation name, or ``None``."""
    token = token.strip().lower().rstrip(".;")
    if token in OPERATIONS:
        return token
    for name, needles in _ALIASES.items():
        if any(needle in token for needle in needles):
            return name
    return None


def canonical_operations(raw: Sequence[str] | None) -> list[str]:
    """Map free-form tokens to known operation names, deduplicated in order.

    Shared by the local pipeline (``normalize_operations``) and the neural
    enhancement prompt builder in ``bot.services.ai_client``.
    """
    out: list[str] = []
    for token in raw or ():
        name = _canonical(str(token))
        if name and name not in out:
            out.append(name)
    return out


def normalize_operations(
    raw: Sequence[str] | None, *, longest_side: int
) -> list[str]:
    """Resolve requested operation names into a deduplicated, ordered list.

    Unknown tokens are dropped; an empty result means the caller gets the
    conservative default pipeline (plus ``upscale`` for small images).
    """
    chosen: set[str] = set(canonical_operations(raw))

    if not chosen:
        chosen = set(DEFAULT_OPERATIONS)
        if longest_side < UPSCALE_TRIGGER_SIDE:
            chosen.add("upscale")

    ordered = [op for op in _ORDER if op in chosen]

    # The model asks for "restore" as a whole pipeline: expand it into its
    # steps at the denoise position so ordering stays sane.
    if "restore" in chosen:
        steps = {"denoise", "contrast", "color", "sharpen"}
        ordered = [op for op in _ORDER if op in (chosen | steps) and op != "restore"]
    # Upscaling an image that is already near the cap changes nothing.
    if "upscale" in chosen and longest_side >= int(MAX_ENHANCED_SIDE * 0.9):
        ordered = [op for op in ordered if op != "upscale"]
    # Everything requested turned out to be a no-op: do the safe default
    # enhancement instead of returning the photo completely untouched.
    if not ordered:
        ordered = [op for op in _ORDER if op in DEFAULT_OPERATIONS]
    return ordered


def _apply_sync(
    data: bytes, operations: Sequence[str] | None
) -> tuple[bytes | None, list[str]]:
    """Run the enhancement pipeline on raw bytes (blocking, worker thread)."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps

        configure_pillow_limits()
    except ImportError:  # pragma: no cover - Pillow is a declared dependency
        logger.error("Pillow is not installed; image enhancement is unavailable")
        return None, []

    try:
        with Image.open(io.BytesIO(data)) as source:
            # Check the pixel count BEFORE anything forces a decode:
            # ``exif_transpose`` loads every pixel, so a decompression bomb (a
            # small file declaring a huge image) would already be expanded into
            # memory by the time a post-decode check could reject it. Pillow's
            # own guard only raises above ~178 Mpx, which is far too late.
            if source.width * source.height > MAX_INPUT_PIXELS:
                logger.error(
                    "Image too large to enhance: %dx%d", source.width, source.height
                )
                return None, []
            image = ImageOps.exif_transpose(source)

            # Work on RGB; keep alpha aside so filters do not smear it.
            has_alpha = image.mode in ("RGBA", "LA", "PA") or (
                image.mode == "P" and "transparency" in image.info
            )
            alpha = None
            if has_alpha:
                image = image.convert("RGBA")
                alpha = image.getchannel("A")
                rgb = image.convert("RGB")
            else:
                rgb = image.convert("RGB")

            ops = normalize_operations(
                operations, longest_side=max(rgb.width, rgb.height)
            )
            for op in ops:
                if op == "contrast":
                    rgb = ImageOps.autocontrast(rgb, cutoff=1)
                elif op == "brightness":
                    rgb = ImageEnhance.Brightness(rgb).enhance(1.12)
                elif op == "denoise":
                    rgb = rgb.filter(ImageFilter.MedianFilter(3))
                elif op == "upscale":
                    factor = min(
                        2.0, MAX_ENHANCED_SIDE / max(rgb.width, rgb.height)
                    )
                    size = (
                        max(1, round(rgb.width * factor)),
                        max(1, round(rgb.height * factor)),
                    )
                    rgb = rgb.resize(size, Image.LANCZOS)
                    if alpha is not None:
                        alpha = alpha.resize(size, Image.LANCZOS)
                elif op == "color":
                    rgb = ImageEnhance.Color(rgb).enhance(1.25)
                elif op == "sharpen":
                    rgb = rgb.filter(
                        ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3)
                    )

            if alpha is not None:
                out = rgb.convert("RGBA")
                out.putalpha(alpha)
                save_kwargs: dict[str, object] = {"format": "PNG", "optimize": True}
            else:
                out = rgb
                save_kwargs = {"format": "JPEG", "quality": 92}

            buffer = io.BytesIO()
            out.save(buffer, **save_kwargs)
            return buffer.getvalue(), ops
    except Exception:
        logger.exception("Image enhancement failed")
        return None, []


async def enhance_image(
    data: bytes, operations: Sequence[str] | None = None
) -> tuple[bytes | None, list[str]]:
    """Improve a photo and return ``(bytes, applied_operations)``.

    ``operations`` accepts free-form tokens (see ``OPERATIONS`` and the alias
    table); unknown or empty means the conservative default pipeline. Returns
    ``(None, [])`` when the input is not a decodable image or Pillow is
    missing, so callers can fall back cleanly.
    """
    return await asyncio.to_thread(_apply_sync, data, operations)
