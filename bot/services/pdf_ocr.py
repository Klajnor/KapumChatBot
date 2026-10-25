"""OCR for scanned PDFs through a vision model.

A PDF without a text layer yields nothing from pdfplumber. Instead of
rejecting it, the first ``PDF_OCR_MAX_PAGES`` pages are rendered to JPEG and
transcribed one request per page. Pages are the unit of cost, so the cap is
hard and the result says how much of the document was actually read.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO

from bot.config import config
from bot.services.ai_client import ai_client

logger = logging.getLogger(__name__)

_RENDER_DPI = 150
_MAX_SIDE = 1600  # longest image side sent to the model, pixels
_CONCURRENCY = 3
_OCR_MAX_TOKENS = 2048
_EMPTY_MARK = "EMPTY"

_OCR_PROMPT = (
    "Transcribe all text on this scanned page exactly as written, in reading "
    "order. Keep the original language. Render tables as rows with ' | ' "
    "between cells. Do not translate, summarize or comment. If the page has "
    f"no readable text, reply with the single word {_EMPTY_MARK}."
)


@dataclass(slots=True)
class OcrResult:
    text: str
    pages_read: int
    total_pages: int

    @property
    def partial(self) -> bool:
        return self.pages_read < self.total_pages


def _render_pages(data: bytes, max_pages: int) -> tuple[list[bytes], int]:
    """JPEG bytes of the first pages and the document's total page count."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber is not installed; PDF OCR is unavailable")
        return [], 0
    images: list[bytes] = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            total = len(pdf.pages)
            for page in pdf.pages[:max_pages]:
                picture = page.to_image(resolution=_RENDER_DPI).original.convert("RGB")
                picture.thumbnail((_MAX_SIDE, _MAX_SIDE))
                buffer = BytesIO()
                picture.save(buffer, format="JPEG", quality=80)
                images.append(buffer.getvalue())
        return images, total
    except Exception as exc:  # noqa: BLE001 - a broken PDF must not kill a reply
        logger.error("PDF page rendering failed: %s", exc)
        return [], 0


async def ocr_pdf(data: bytes) -> OcrResult | None:
    """Transcribe the first pages of a scanned PDF; None when nothing was read."""
    images, total = await asyncio.to_thread(_render_pages, data, config.pdf_ocr_max_pages)
    if not images:
        return None

    gate = asyncio.Semaphore(_CONCURRENCY)

    async def read_page(image: bytes) -> str:
        async with gate:
            try:
                result = await ai_client.chat_with_image_result(
                    _OCR_PROMPT, image, [], temperature=0, max_tokens=_OCR_MAX_TOKENS
                )
            except Exception as exc:  # noqa: BLE001 - one bad page must not sink the rest
                logger.warning("OCR page failed: %s", exc)
                return ""
        text = (result.content or "").strip()
        return "" if text.upper().strip(". ") == _EMPTY_MARK else text

    texts = await asyncio.gather(*(read_page(image) for image in images))
    blocks = [f"## Page {number}\n{text}" for number, text in enumerate(texts, 1) if text]
    if not blocks:
        return None
    return OcrResult(
        text="\n\n".join(blocks), pages_read=len(images), total_pages=total
    )
