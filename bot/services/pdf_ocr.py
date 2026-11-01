"""OCR for scanned PDFs through a vision model.

A PDF without a text layer yields nothing from pdfplumber. Instead of
rejecting it, the first ``PDF_OCR_MAX_PAGES`` pages are rendered to JPEG and
transcribed one request per page. Pages are the unit of cost, so the cap is
hard and the result says how much of the document was actually read.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from io import BytesIO

from bot.config import config
from bot.services.ai_client import ai_client
from bot.services.documents import DocumentParseTimeout, _check_deadline, _run_parser
from bot.services.pdf_limits import PdfResourceLimit, pdf_stream_limits
from bot.services.pdf_pages import open_pdf_pages

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
    total_pages: int | None

    @property
    def partial(self) -> bool:
        return self.total_pages is None or self.pages_read < self.total_pages


def _render_pages(
    data: bytes, max_pages: int, deadline: float | None = None
) -> tuple[list[bytes], int | None]:
    """JPEG первых страниц; total неизвестно при обрезке по лимиту."""
    try:
        import pdfplumber  # noqa: F401 - проверка опциональной зависимости
    except ImportError:
        logger.warning("pdfplumber is not installed; PDF OCR is unavailable")
        return [], 0
    images: list[bytes] = []
    try:
        _check_deadline(deadline)
        with pdf_stream_limits(), open_pdf_pages(
            data, max_pages, lambda: _check_deadline(deadline)
        ) as pdf:
            for page in pdf.pages:
                _check_deadline(deadline)
                # PDFium растеризует MediaBox до обрезки. thumbnail() после
                # to_image не защищает от гигантской исходной аллокации.
                x0, y0, x1, y1 = page.mediabox
                unit = float(page.page_obj.attrs.get("UserUnit", 1))
                sides = [float(page.width), float(page.height), abs(x1 - x0), abs(y1 - y0)]
                if not all(math.isfinite(side) and side > 0 for side in sides):
                    raise ValueError("Invalid PDF page dimensions")
                if not math.isfinite(unit) or unit <= 0:
                    raise ValueError("Invalid PDF UserUnit")
                longest = max(sides) * max(1, unit)
                resolution = min(_RENDER_DPI, _MAX_SIDE * 72 / longest)
                if not math.isfinite(resolution) or resolution < 0.001:
                    raise ValueError("PDF page dimensions exceed rendering limits")
                picture = page.to_image(resolution=resolution).original.convert("RGB")
                picture.thumbnail((_MAX_SIDE, _MAX_SIDE))
                buffer = BytesIO()
                picture.save(buffer, format="JPEG", quality=80)
                images.append(buffer.getvalue())
            total = pdf.total_pages
        return images, total
    except (DocumentParseTimeout, PdfResourceLimit):
        raise
    except Exception as exc:  # noqa: BLE001 - a broken PDF must not kill a reply
        logger.error("PDF page rendering failed: %s", exc)
        return [], 0


async def ocr_pdf(data: bytes, *, model: str | None = None) -> OcrResult | None:
    """Transcribe the first pages of a scanned PDF; None when nothing was read.

    ``model`` is the user's selected model; ``chat_with_image_result`` still
    filters it through ``VISION`` and falls back to a vision-capable model.
    """
    images, total = await _run_parser(_render_pages, data, config.pdf_ocr_max_pages)
    if not images:
        return None

    gate = asyncio.Semaphore(_CONCURRENCY)

    async def read_page(image: bytes) -> str:
        async with gate:
            try:
                result = await ai_client.chat_with_image_result(
                    _OCR_PROMPT,
                    image,
                    [],
                    temperature=0,
                    max_tokens=_OCR_MAX_TOKENS,
                    model=model,
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
