"""Extract display math and render it locally without a TeX installation."""
from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

# Code takes precedence, including an unfinished fenced code block.
_PARTS = re.compile(
    r"(?P<code>```[\s\S]*?(?:```|\Z)|`[^`\n]*`)"
    r"|(?P<math>\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)"
)
_RENDER_LOCK = threading.Lock()
MAX_FORMULA_CHARS = 2000
MAX_FORMULA_IMAGES = 6


def split_display_math(text: str) -> list[tuple[str, str]]:
    """Return ordered text/math parts, preserving code and math delimiters."""
    parts: list[tuple[str, str]] = []
    start = 0
    for match in _PARTS.finditer(text):
        if not match.group("math"):
            continue
        if match.start() > start:
            parts.append(("text", text[start:match.start()]))
        parts.append(("math", match.group()))
        start = match.end()
    if start < len(text):
        parts.append(("text", text[start:]))
    return parts


@lru_cache(maxsize=64)
def _render_png(block: str) -> bytes | None:
    if len(block) > MAX_FORMULA_CHARS:
        return None
    expression = block[2:-2].strip()
    if not expression or "$" in expression:
        return None
    expression = "$" + expression + "$"
    try:
        # Matplotlib's state and caches are shared: serialize all rendering.
        with _RENDER_LOCK:
            from matplotlib import rc_context
            from matplotlib.font_manager import FontProperties
            from matplotlib.mathtext import MathTextParser, math_to_image
            from PIL import Image, ImageOps

            prop = FontProperties(size=20)
            with rc_context({"text.usetex": False, "mathtext.fontset": "dejavusans",
                             "savefig.facecolor": "white", "savefig.transparent": False}):
                layout = MathTextParser("path").parse(expression, dpi=160, prop=prop)
                if layout.width > 3600 or layout.height > 1600:
                    return None
                output = io.BytesIO()
                math_to_image(expression, output, prop=prop, dpi=160, format="png", color="black")
            output.seek(0)
            with Image.open(output) as image:
                # A white background remains legible in either Telegram theme.
                rgba = image.convert("RGBA")
                canvas = Image.new("RGB", rgba.size, "white")
                canvas.paste(rgba, mask=rgba.getchannel("A"))
                padded = ImageOps.expand(canvas, border=24, fill="white")
                result = io.BytesIO()
                padded.save(result, format="PNG")
                return result.getvalue()
    except Exception as exc:  # malformed or unsupported math must keep its text
        logger.debug("Formula rendering fell back to text: %s", exc)
        return None


async def render_formula(block: str) -> bytes | None:
    """Keep CPU work off the bot's event loop."""
    if len(block) > MAX_FORMULA_CHARS:
        return None
    return await asyncio.to_thread(_render_png, block)
