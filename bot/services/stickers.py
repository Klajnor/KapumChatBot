from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bot.config import config

logger = logging.getLogger(__name__)
STICKER_SIZE = 512
STICKER_MAX_BYTES = 512 * 1024
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stickers")
_pending: asyncio.Future | None = None
_session: Any = None


def to_sticker_webp(image_bytes: bytes, size: int = STICKER_SIZE) -> bytes | None:
    """Вписать объект целиком в прозрачный квадрат и соблюсти лимит Telegram."""
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
        bounds = image.getchannel("A").getbbox()
        if bounds is None:
            return None
        image = image.crop(bounds)
        margin = max(1, round(size * 0.04))
        image = ImageOps.contain(image, (size - 2 * margin, size - 2 * margin), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (size, size))
        canvas.alpha_composite(image, ((size - image.width) // 2, (size - image.height) // 2))
        for quality in (90, 80, 70, 60, 50, 40):
            buffer = io.BytesIO()
            canvas.save(buffer, format="WEBP", quality=quality, method=6)
            data = buffer.getvalue()
            if len(data) <= STICKER_MAX_BYTES:
                return data
        return None
    except Exception as exc:  # noqa: BLE001 - ошибка изображения не должна ломать ответ
        logger.warning("Sticker conversion failed: %s", exc)
        return None


def _prepare_sticker(image_bytes: bytes) -> bytes | None:
    """Удалить фон локальной моделью; сессию переиспользует один рабочий поток."""
    global _session
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
        # Сохраняем нативный вырезанный фон; одной общей полупрозрачности недостаточно.
        if image.getchannel("A").getextrema()[0] != 0:
            from rembg import new_session, remove

            if _session is None:
                _session = new_session("u2net", providers=["CPUExecutionProvider"])
            image = remove(image, session=_session).convert("RGBA")
        low, high = image.getchannel("A").getextrema()
        if low != 0 or high == 0:
            logger.warning("Background removal produced no usable transparency")
            return None
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return to_sticker_webp(buffer.getvalue())
    except Exception as exc:  # noqa: BLE001 - включая отсутствие rembg и сбой загрузки модели
        logger.warning("Sticker background removal failed: %s", exc)
        return None


async def prepare_sticker(image_bytes: bytes) -> bytes | None:
    """Обработать стикер вне event loop, без накопления очереди после таймаута."""
    global _pending
    if _pending is not None and not _pending.done():
        logger.warning("Sticker processor is busy")
        return None
    _pending = asyncio.get_running_loop().run_in_executor(_executor, _prepare_sticker, image_bytes)
    try:
        # Отмена ожидания не освобождает слот: ONNX-поток нельзя безопасно прервать.
        return await asyncio.wait_for(asyncio.shield(_pending), timeout=config.sticker_processing_timeout)
    except TimeoutError:
        logger.warning("Sticker processing timed out")
        return None
