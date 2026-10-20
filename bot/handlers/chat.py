from __future__ import annotations

import asyncio
import html
import logging
import time
from typing import Any

from aiogram import F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.types import BufferedInputFile, Message

from bot.bot import get_bot_username
from bot.config import config
from bot.database.session import async_session_factory
from bot.services.access import AccessResult, check_access, guest_hint
from bot.services.ai_client import ChatResult, ai_client, is_vision_model
from bot.services.progress import waiting
from bot.services.telegram_utils import (
    format_timing,
    is_reply_to_bot,
    mentions_bot,
    safe_answer,
    send_formatted,
    strip_bot_mention,
    with_timing,
)
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

router = Router()

MAX_MESSAGE_LENGTH = 4000  # Telegram hard limit is 4096; keep a safety margin.

DEFAULT_VISION_PROMPT = "What's in this image?"
DEFAULT_VOICE_PROMPT = "Answer what the user said in this voice message."


# --- Addressing helpers ----------------------------------------------------


def _is_private(message: Message) -> bool:
    return message.chat.type == ChatType.PRIVATE


def _is_addressed(message: Message) -> bool:
    """Return True if the bot should react to this message.

    In private chats every message is addressed to the bot. In groups the bot
    reacts only when it is mentioned (``@username``), when the message is a
    reply to the bot, or when it starts with a command.
    """
    if _is_private(message):
        return True
    if is_reply_to_bot(message):
        return True
    text = message.text or message.caption or ""
    if mentions_bot(text):
        return True
    return text.lstrip().startswith("/")


# --- Sending helpers -------------------------------------------------------


async def send_long_message(message: Message, text: str) -> bool:
    """Send an AI answer rendered from Markdown, splitting long replies.

    Formatting falls back automatically (MarkdownV2 → HTML → plain text) and
    rate limits are retried, so a reply is not lost because Telegram rejected
    the entities. Returns False only if nothing at all could be delivered.
    """
    delivered = await send_formatted(message, text, limit=MAX_MESSAGE_LENGTH)
    if delivered:
        return True

    logger.error("Answer was not delivered to chat %s; trying a bare fallback", message.chat.id)
    fallback = text.strip()[:MAX_MESSAGE_LENGTH] if text else ""
    if fallback and await safe_answer(
        message, fallback + ("\n…" if len(text.strip()) > MAX_MESSAGE_LENGTH else ""),
        parse_mode=None,
    ):
        return True
    await safe_answer(
        message,
        "⚠️ <b>The answer could not be delivered</b> — Telegram rejected the "
        "message. Please try again in a moment.",
    )
    return False


async def _send_image_by_url(message: Message, url: str, caption: str) -> bool:
    """Send the generated image by URL when its bytes could not be downloaded.

    A presigned URL that this server may not fetch (403, expired signature,
    geo/network restriction) can still be reachable for Telegram's servers. If
    attaching fails too, the link itself is sent — it stays valid for ~24 hours.
    """
    try:
        await message.answer_photo(url, caption=caption)
        logger.info("Image delivered by URL (Telegram fetched it)")
        return True
    except TelegramAPIError as exc:
        logger.warning("Sending the image by URL failed: %s", exc)

    try:
        await message.answer(
            "🎨 <b>The image was generated, but I could not attach it.</b>\n\n"
            f'<a href="{html.escape(url, quote=True)}">Open the image</a> '
            "(the link is valid for about 24 hours)."
        )
        return True
    except TelegramAPIError as exc:
        logger.warning("Sending the image link failed: %s", exc)
        return False


async def _safe_answer_photo(message: Message, photo: Any, caption: str) -> bool:
    """Send a photo, retrying on a rate limit. Never raises."""
    for attempt in range(3):
        try:
            await message.answer_photo(photo, caption=caption)
            return True
        except TelegramRetryAfter as exc:
            if attempt >= 2:
                logger.error("Rate limit persists; the photo was not sent")
                break
            await asyncio.sleep(float(exc.retry_after) + 0.1)
        except TelegramAPIError as exc:
            logger.error("Sending the photo failed: %s", exc)
            break
    await safe_answer(
        message, "⚠️ <b>The image could not be sent.</b> Please try again in a moment."
    )
    return False


async def generate_and_send_image(
    message: Message,
    prompt: str,
    *,
    service: UserService | None = None,
    user_id: int | None = None,
    model: str | None = None,
) -> bool:
    """Generate an image from ``prompt`` and send it. Returns True on success."""
    await message.bot.send_chat_action(
        chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO
    )
    # Image synthesis is slow: show a live "waiting" notice after the delay.
    async with waiting(message):
        result = await ai_client.generate_image_result(prompt)
    image, image_url = result.data, result.url

    caption = f"🎨 <i>{html.escape(prompt[:900])}</i>"
    footer = format_timing(result.elapsed, model=config.image_model)
    if footer:
        caption = f"{caption}\n\n{html.escape(footer)}"

    if image is not None:
        if not await _safe_answer_photo(
            message, BufferedInputFile(image, filename="generated.png"), caption
        ):
            return False
    elif image_url and await _send_image_by_url(message, image_url, caption):
        logger.warning(
            "Image was sent without downloading it: %s", image_url.split("?", 1)[0]
        )
    else:
        await safe_answer(
            message,
            "❌ <b>Could not generate the image.</b>\n\n"
            "The prompt may have been rejected by moderation, or the image "
            "service / model is unavailable for your account.",
        )
        return False

    if service is not None and user_id is not None:
        await service.add_message(
            chat_id=message.chat.id,
            user_id=user_id,
            role="assistant",
            content=f"[Generated image] {prompt}",
            model=model,
        )
    return True


# --- Access helpers --------------------------------------------------------


async def _open_registered(message: Message) -> UserService | None:
    """Register the sender and return a service bound to an open session."""
    user = message.from_user
    if user is None:
        return None

    session = async_session_factory()
    service = UserService(session)
    try:
        await service.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
    except Exception:
        await session.close()
        raise
    return service


async def _consume_access(message: Message, service: UserService) -> AccessResult | None:
    """Run the access check, replying with the reason when denied."""
    user = message.from_user
    if user is None:
        return None
    result = await check_access(service, user.id)
    if not result.allowed:
        await message.answer(result.message or get_random_themed_response())
        return None
    return result


# --- Core answer pipeline --------------------------------------------------


async def _answer_with_ai(
    message: Message,
    service: UserService,
    text: str,
    access: AccessResult,
) -> None:
    """Produce the assistant reply for ``text``: a message and/or an image."""
    user = message.from_user
    if user is None:
        return

    settings = await service.get_user_settings(user.id)
    history = await service.get_history(message.chat.id)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": settings["system_prompt"]},
        *history,
        {"role": "user", "content": text},
    ]

    await service.add_message(
        chat_id=message.chat.id,
        user_id=user.id,
        role="user",
        content=text,
        model=settings["model"],
    )

    if config.enable_image_generation:
        async with waiting(message):
            result: ChatResult = await ai_client.chat_with_tools(
                messages=messages,
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            )
    else:
        async with waiting(message):
            result = await ai_client.chat_completion_result(
                messages=messages,
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            )

    image_call = next(
        (call for call in result.tool_calls if call.name == "generate_image"), None
    )

    # Send the textual part first, if any.
    if result.content:
        await service.add_message(
            chat_id=message.chat.id,
            user_id=user.id,
            role="assistant",
            content=result.content,
            model=settings["model"],
        )
        footer = format_timing(
            result.elapsed,
            model=settings["model"],
            tokens=result.total_tokens,
        )
        await send_long_message(message, with_timing(result.content, footer))

    if image_call is not None:
        prompt = str(image_call.arguments.get("prompt") or text).strip() or text
        await generate_and_send_image(
            message,
            prompt,
            service=service,
            user_id=user.id,
            model=settings["model"],
        )
    elif not result.content:
        await safe_answer(
            message,
            "❌ <b>Error communicating with the AI.</b>\n\n"
            "Please try again later. If the problem persists, check the "
            "API key and balance of the configured Qwen account.",
        )
        return

    hint = guest_hint(access)
    if hint:
        await safe_answer(message, hint)


# --- Handlers --------------------------------------------------------------


@router.message(F.text, ~F.text.startswith("/"))
async def handle_text(message: Message) -> None:
    """Handle text messages: chat with the AI."""
    user = message.from_user
    if user is None or not message.text:
        return
    if not _is_addressed(message):
        return

    text = strip_bot_mention(message.text)
    if not text:
        return

    service = await _open_registered(message)
    if service is None:
        return

    try:
        access = await _consume_access(message, service)
        if access is None:
            return
        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )
        await _answer_with_ai(message, service, text, access)
    finally:
        await service.session.close()


@router.message(F.voice)
async def handle_voice(message: Message) -> None:
    """Handle voice messages: transcribe them, then answer with text or image."""
    user = message.from_user
    if user is None or message.voice is None:
        return
    if not config.enable_voice:
        return
    if not _is_addressed(message):
        return

    if message.voice.duration > 300:
        await safe_answer(message, "⚠️ Voice messages must be at most 5 minutes long.")
        return

    if message.voice.file_size and message.voice.file_size > config.voice_max_bytes:
        await message.answer(
            "⚠️ <b>Voice message is too large.</b>\n\n"
            f"Maximum size for recognition is "
            f"{config.voice_max_bytes // (1024 * 1024)} MB (~5 minutes)."
        )
        return

    service = await _open_registered(message)
    if service is None:
        return

    try:
        access = await _consume_access(message, service)
        if access is None:
            return

        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )
        file = await message.bot.get_file(message.voice.file_id)
        buffer = await message.bot.download_file(file.file_path)
        audio_bytes = buffer.read()

        transcript = None
        asr_started = time.monotonic()
        async with waiting(message):
            transcript = await ai_client.transcribe_audio(
                audio_bytes,
                mime=message.voice.mime_type or "audio/ogg",
            )
        asr_elapsed = time.monotonic() - asr_started

        if not transcript or not transcript.strip():
            await safe_answer(
                message,
                "❌ <b>Could not recognize the voice message.</b>\n\n"
                "Please speak more clearly or type your question.",
            )
            return

        asr_timing = (
            f" <i>({asr_elapsed:.1f} с)</i>"
            if config.show_timing and asr_elapsed > 0
            else ""
        )
        await safe_answer(
            message,
            f"🎤 <b>Recognized</b>{asr_timing}:\n<i>{html.escape(transcript)}</i>",
        )

        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )
        await _answer_with_ai(message, service, transcript, access)
    finally:
        await service.session.close()


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    """Handle photo messages: analyze the image with a vision model."""
    user = message.from_user
    if user is None:
        return
    if not _is_addressed(message):
        return

    caption = strip_bot_mention(message.caption or "") or DEFAULT_VISION_PROMPT

    service = await _open_registered(message)
    if service is None:
        return

    try:
        settings = await service.get_user_settings(user.id)

        # Verify the model supports images *before* consuming quota.
        if not is_vision_model(settings["model"]):
            await message.answer(
                "⚠️ <b>The current model does not support images.</b>\n\n"
                f"Model <code>{html.escape(settings['model'])}</code> cannot "
                f"process images. Switch to a vision-capable model with /models.\n\n"
                "<b>Vision models</b> are marked 🖼️ in the list "
                "(for example <code>qwen3.8-max</code> or <code>qwen3.8-flash</code>)."
            )
            return

        access = await _consume_access(message, service)
        if access is None:
            return

        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )

        photo = message.photo[-1]
        file = await message.bot.get_file(photo.file_id)
        buffer = await message.bot.download_file(file.file_path)
        image_bytes = buffer.read()

        history = await service.get_history(message.chat.id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": settings["system_prompt"]},
            *history,
        ]

        await service.add_message(
            chat_id=message.chat.id,
            user_id=user.id,
            role="user",
            content=f"[Image] {caption}",
            image_url=photo.file_id,
            model=settings["model"],
        )

        async with waiting(message):
            vision_result = await ai_client.chat_with_image_result(
                text=caption,
                image_bytes=image_bytes,
                messages=messages,
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            )
        response = vision_result.content

        if not response:
            await safe_answer(
                message,
                "❌ <b>Error analyzing the image.</b>\n\nPlease try again later.",
            )
            return

        await service.add_message(
            chat_id=message.chat.id,
            user_id=user.id,
            role="assistant",
            content=response,
            model=settings["model"],
        )

        footer = format_timing(
            vision_result.elapsed,
            model=settings["model"],
            tokens=vision_result.total_tokens,
        )
        await send_long_message(message, with_timing(response, footer))

        hint = guest_hint(access)
        if hint:
            await safe_answer(message, hint)
    finally:
        await service.session.close()


@router.message(F.text.startswith("/"))
async def handle_unknown_command(message: Message) -> None:
    """Answer an unknown command instead of dropping it silently.

    This router is registered last, so every real command has already been
    matched by the time we get here. In groups the bot reacts only when the
    command explicitly targets it (``/foo@thisbot``), to stay quiet about other
    bots' commands.
    """
    if message.from_user is None or not message.text:
        return

    first = message.text.strip().split()[0] if message.text.strip() else ""
    if message.chat.type != ChatType.PRIVATE:
        username = get_bot_username()
        if not username or f"@{username.lower()}" not in first.lower():
            return

    command = first.split("@")[0]
    await safe_answer(
        message,
        f"🤔 <b>Unknown command</b> <code>{html.escape(command[:32])}</code>\n\n"
        "See /help for the list of available commands.",
    )
