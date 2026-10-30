from __future__ import annotations

import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bot.config import config

logger = logging.getLogger(__name__)
STICKER_SIZE = 512
STICKER_MAX_BYTES = 512 * 1024
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stickers")
_pending: asyncio.Future[StickerResult] | None = None
_session: Any = None


@dataclass(slots=True)
class StickerResult:
    """Результат обработки с безопасным кодом ошибки вместо сырого исключения."""

    data: bytes | None = None
    error: str | None = None


_ERROR_MESSAGES = {
    "processor_busy": (
        "Processing: the local sticker processor is busy. A previous background-removal "
        "job may still be running after a timeout. Wait for it to finish and try again."
    ),
    "processing_timeout": (
        "Processing: background removal or WebP encoding exceeded the configured time limit. "
        "The first run also downloads and loads the u2net model (~176 MB). "
        "Administrator: check model download access and CPU/RAM; preload the model or "
        "increase STICKER_PROCESSING_TIMEOUT (up to 600 seconds)."
    ),
    "dependency_pillow": "Image processing: Pillow is missing. Administrator: install the project requirements in the bot's Python environment and restart.",
    "dependency_rembg": "Background removal: rembg is not installed in the bot's Python environment. Administrator: run pip install -r requirements.txt in that environment and restart the bot.",
    "dependency_onnxruntime": "Background removal: ONNX Runtime is missing. Administrator: install rembg[cpu] in the bot's Python environment and restart.",
    "dependencies_missing": "Background removal: a required Python dependency is missing. Administrator: run pip install -r requirements.txt in the bot's Python environment and check the server log for the missing module.",
    "runtime_load_failed": "Background removal: rembg or its native CPU runtime could not be loaded. Administrator: check ONNX Runtime/platform compatibility and the server log, then repair the installation and restart.",
    "invalid_image": "Image decoding: the generated file could not be read as an image. Try generating the sticker again; if this repeats, ask the administrator to check the image provider.",
    "model_load_failed": "Background removal: the u2net model could not be downloaded or loaded. Administrator: check access to the model download host, free disk space and STICKER_MODEL_DIR; see the server log for the underlying error.",
    "model_cache_permission": "Background removal: access to the model directory was denied. Administrator: give the bot service user read/write access to models/rembg, or set STICKER_MODEL_DIR in .env to a writable directory. Restart the bot after changing the configuration.",
    "model_cache_failed": "Background removal: the model directory could not be created or configured. Administrator: check STICKER_MODEL_DIR in .env, free disk space and the server log. The path must be a directory, not a file.",
    "out_of_memory": "Local processing: the processor reported insufficient memory. Administrator: free RAM or increase the server's memory, then try again.",
    "background_removal_failed": "Background removal: the local segmentation model failed while processing the image. Try again; if this repeats, the administrator should check the ONNX Runtime error in the server log.",
    "no_transparency": "Background removal: the model did not produce a usable cutout (the result was opaque or completely empty). Try a single clearly outlined subject on a simple background.",
    "empty_image": "WebP encoding: no visible object remained after background removal. Try a clearer description of a single subject.",
    "webp_too_large": "WebP encoding: the sticker still exceeds Telegram's 512 KB limit after compression. Try a simpler subject with fewer fine details.",
    "encoding_failed": "WebP encoding: the processed image could not be converted to a Telegram sticker. Administrator: check Pillow/WebP support and the server log.",
    "generation_connection": "Image generation: the provider did not return an HTTP response (a connection failure or timeout). Try again later; the administrator should check provider connectivity and IMAGE_TIMEOUT in the server log.",
    "generation_auth": "Image generation: the provider rejected authentication (HTTP 401). Administrator: check the image API key and its provider configuration.",
    "generation_forbidden": "Image generation: the provider denied access (HTTP 403). Administrator: check account permissions and access to the configured image model.",
    "generation_rejected": "Image generation: the provider rejected the request (HTTP 400). Try rephrasing the prompt; the administrator should check moderation, image model and parameters in the server log.",
    "generation_not_found": "Image generation: the provider endpoint or model was not found (HTTP 404). Administrator: check the image API URL and model name.",
    "generation_rate_limit": "Image generation: the provider returned HTTP 429 (rate limit or account quota). Wait and retry; the administrator should check the provider balance and limits.",
    "generation_unavailable": "Image generation: the provider returned a server error (HTTP 5xx). Try again later.",
    "generation_failed": "Image generation: the provider request failed. Administrator: check the HTTP status and provider error in the server log.",
    "generation_invalid_response": "Image generation: the provider returned an empty or unreadable response. Try again later; the administrator should check provider compatibility and the server log.",
    "generation_no_image": "Image generation: the provider returned no image URL. The administrator should check the provider response for moderation or an unsupported response format; try rephrasing the prompt.",
    "image_download_failed": "Image download: an image URL was generated, but the bot could not download the source image. Administrator: check the server log for a download failure, safety rejection or file-size limit. Try generating again.",
}


def sticker_error_message(error: str | None) -> str:
    """Вернуть только доверенный текст: ошибки API и исключения пользователю не передаются."""
    text = _ERROR_MESSAGES.get(error, "Sticker generation failed without a detailed diagnostic. Please try again; the administrator should check the server log.")
    if error == "processing_timeout":
        text += f" Current processing timeout: {config.sticker_processing_timeout} seconds."
    return text


def to_sticker_webp_result(image_bytes: bytes, size: int = STICKER_SIZE) -> StickerResult:
    """Вписать объект целиком в прозрачный квадрат и соблюсти лимит Telegram."""
    stage = "invalid_image"
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
        stage = "encoding_failed"
        bounds = image.getchannel("A").getbbox()
        if bounds is None:
            return StickerResult(error="empty_image")
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
                return StickerResult(data=data)
        return StickerResult(error="webp_too_large")
    except ImportError:
        logger.exception("Sticker conversion dependency is unavailable")
        return StickerResult(error="dependency_pillow")
    except MemoryError:
        logger.exception("Sticker conversion ran out of memory")
        return StickerResult(error="out_of_memory")
    except Exception:  # Ошибка изображения не должна ломать ответ.
        logger.exception("Sticker conversion failed (stage=%s)", stage)
        return StickerResult(error=stage)


def to_sticker_webp(image_bytes: bytes, size: int = STICKER_SIZE) -> bytes | None:
    """Совместимый интерфейс преобразования без диагностических данных."""
    return to_sticker_webp_result(image_bytes, size).data


def configure_model_cache() -> Path:
    """Создать каталог моделей от корня проекта и передать абсолютный путь rembg."""
    path = Path(config.sticker_model_dir).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["U2NET_HOME"] = str(path)
    logger.info("Sticker model directory: %s", path)
    return path


def _prepare_sticker_result(image_bytes: bytes) -> StickerResult:
    """Удалить фон локальной моделью; сессию переиспользует один рабочий поток."""
    global _session
    stage = "dependency_pillow"
    try:
        from PIL import Image, ImageOps

        stage = "invalid_image"
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
        # Сохраняем нативный вырезанный фон; одной общей полупрозрачности недостаточно.
        if image.getchannel("A").getextrema()[0] != 0:
            if _session is None:
                stage = "model_cache_failed"
                configure_model_cache()
            stage = "runtime_load_failed"
            from rembg import new_session, remove

            if _session is None:
                stage = "model_load_failed"
                _session = new_session("u2net", providers=["CPUExecutionProvider"])
            stage = "background_removal_failed"
            image = remove(image, session=_session).convert("RGBA")
        stage = "encoding_failed"
        low, high = image.getchannel("A").getextrema()
        if low != 0 or high == 0:
            logger.warning("Background removal produced no usable transparency")
            return StickerResult(error="no_transparency")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return to_sticker_webp_result(buffer.getvalue())
    except ModuleNotFoundError as exc:
        code = {"PIL": "dependency_pillow", "rembg": "dependency_rembg", "onnxruntime": "dependency_onnxruntime"}.get(exc.name, "dependencies_missing")
        logger.exception("Sticker dependency is missing (stage=%s)", stage)
        return StickerResult(error=code)
    except ImportError:
        logger.exception("Sticker runtime could not be imported (stage=%s)", stage)
        return StickerResult(error="dependency_pillow" if stage == "dependency_pillow" else "runtime_load_failed")
    except SystemExit:
        # Некоторые версии rembg завершают импорт через sys.exit при отсутствии ONNX.
        logger.exception("Sticker library attempted to exit (stage=%s)", stage)
        return StickerResult(error=stage)
    except PermissionError:
        logger.exception("Sticker processing permission error (stage=%s)", stage)
        return StickerResult(error="model_cache_permission" if stage in {"model_cache_failed", "model_load_failed"} else stage)
    except MemoryError:
        logger.exception("Sticker processing ran out of memory (stage=%s)", stage)
        return StickerResult(error="out_of_memory")
    except Exception:  # Включая сбой загрузки модели и ошибки ONNX.
        logger.exception("Sticker processing failed (stage=%s)", stage)
        return StickerResult(error=stage)


def _prepare_sticker(image_bytes: bytes) -> bytes | None:
    """Совместимый синхронный интерфейс обработки без диагностики."""
    return _prepare_sticker_result(image_bytes).data


async def prepare_sticker_result(image_bytes: bytes) -> StickerResult:
    """Обработать стикер вне event loop, без накопления очереди после таймаута."""
    global _pending
    if _pending is not None and not _pending.done():
        logger.warning("Sticker processor is busy")
        return StickerResult(error="processor_busy")
    _pending = asyncio.get_running_loop().run_in_executor(_executor, _prepare_sticker_result, image_bytes)
    try:
        # Отмена ожидания не освобождает слот: ONNX-поток нельзя безопасно прервать.
        return await asyncio.wait_for(asyncio.shield(_pending), timeout=config.sticker_processing_timeout)
    except TimeoutError:
        logger.warning("Sticker processing timed out")
        return StickerResult(error="processing_timeout")


async def prepare_sticker(image_bytes: bytes) -> bytes | None:
    """Совместимый async-интерфейс обработки без диагностических данных."""
    return (await prepare_sticker_result(image_bytes)).data
