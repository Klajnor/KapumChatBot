from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from aiogram import Router
from aiogram.types import (
    InlineQueryResult,
    InlineQueryResultArticle,
    InputTextMessageContent,
    LinkPreviewOptions,
    Message,
)

from bot.config import config
from bot.database.session import async_session_factory
from bot.services import limits
from bot.services.access import AccessResult, check_access
from bot.services.ai_client import ChatResult, ai_client, is_vision_model
from bot.services.markdown import render, split_chunks
from bot.services.science import normalize_formulas
from bot.services.telegram_utils import (
    PARSE_MODES,
    format_timing,
    safe_answer_guest,
    strip_bot_mention,
    with_timing,
)
from bot.services.telegraph import needs_telegraph, notice_text, publish_answer
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

router = Router()

MAX_TEXT_LENGTH = 4000
MAX_CAPTION_LENGTH = 1000
DEFAULT_VISION_PROMPT = "What's in this image?"


@dataclass(slots=True)
class GuestReply:
    """A single guest-mode reply: either text or a generated image."""

    text: str | None = None
    mode: str = "markdown"  # markdown | html | plain
    photo_url: str | None = None
    caption: str | None = None


# --- Result builders -------------------------------------------------------


def _to_result(reply: GuestReply, *, mode: str | None = None) -> InlineQueryResult:
    """Convert a reply into the InlineQueryResult expected by answerGuestQuery."""
    if reply.photo_url:
        # Inline photo results require a Telegram-fetchable JPEG. Qwen returns
        # signed PNG URLs, so send the complete link without a preview fetch.
        url = reply.photo_url
        if len(url) > MAX_TEXT_LENGTH - 100:
            text = "Картинка создана, но ссылка слишком длинная. Используйте /image в личном чате с ботом."
        else:
            caption = (reply.caption or "🎨 Картинка готова.")[:MAX_CAPTION_LENGTH]
            caption = caption[:max(0, MAX_TEXT_LENGTH - len(url) - 2)]
            text = f"{caption}\n\n{url}"
        return InlineQueryResultArticle(
            id="image",
            title="Открыть сгенерированное изображение",
            input_message_content=InputTextMessageContent(
                message_text=text,
                parse_mode=None,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            ),
        )

    effective = mode or reply.mode
    text = reply.text or "…"
    # Guest replies accept one result and cannot upload a local formula PNG.
    if config.render_formula_images or config.normalize_formulas:
        text = normalize_formulas(text) or text
    chunks = split_chunks(text, effective, MAX_TEXT_LENGTH - 32)
    body = render(chunks[0], effective) if chunks else "…"
    if len(chunks) > 1:
        body += "\n\n…"
    return InlineQueryResultArticle(
        id="answer",
        title=" ".join(text.split())[:120] or "Answer",
        input_message_content=InputTextMessageContent(
            message_text=body,
            parse_mode=PARSE_MODES.get(effective),
        ),
    )


def _guest_hint(access: AccessResult) -> str:
    """Plain-text reminder about remaining free-trial messages."""
    if access.role == "guest" and access.remaining is not None and access.remaining <= 2:
        return f"\n\n_Free trial messages left: {access.remaining}_"
    return ""


async def _maybe_telegraph_reply(
    question: str, answer: str, footer: str
) -> GuestReply | None:
    """A guest query allows one short reply: links beat broken tables/formulas.

    Returns ``None`` when Telegraph is not needed or unavailable, so the
    caller sends the answer as a normal (truncated) text reply.
    """
    if not config.enable_telegraph or not needs_telegraph(answer):
        return None
    url = await publish_answer(question, answer)
    if not url:
        return None
    return GuestReply(text=with_timing(notice_text(url), footer), mode="plain")


# --- Media helpers ---------------------------------------------------------


async def _download(message: Message, file_id: str) -> bytes | None:
    """Best-effort file download.

    The bot is not a member of the summoning chat, so Telegram may refuse to
    hand over the file; that must not break the reply.
    """
    try:
        file = await message.bot.get_file(file_id)
        buffer = await message.bot.download_file(file.file_path)
        return buffer.read()
    except Exception as exc:  # noqa: BLE001 - guest file access may be denied
        logger.warning("Could not download guest file: %s", exc)
        return None


async def _download_photo(message: Message) -> bytes | None:
    """Return the largest attached (or replied-to) photo, if any."""
    source = message if message.photo else message.reply_to_message
    if source is None or not source.photo:
        return None
    return await _download(message, source.photo[-1].file_id)


def _voice_of(message: Message):
    """Return the voice note attached to the message or to the replied one."""
    if message.voice is not None:
        return message.voice
    reply = message.reply_to_message
    return reply.voice if reply is not None else None


def _extract_text(message: Message) -> str:
    """Summoning text with the ``@bot`` mention removed."""
    return strip_bot_mention(message.text or message.caption or "").strip()


# --- Reply pipeline --------------------------------------------------------


async def _build_reply(
    message: Message,
    service: UserService,
    caller,
    chat_id: int,
) -> GuestReply:
    """Produce the single guest reply for a summoning message."""
    caller_id = caller.id
    await service.register_user(
        telegram_id=caller_id,
        username=caller.username,
        first_name=caller.first_name,
        last_name=caller.last_name,
    )

    access = await check_access(service, caller_id)
    if not access.allowed:
        return GuestReply(
            text=access.message or get_random_themed_response(), mode="html"
        )

    text = _extract_text(message)

    # Voice message: recognize it first, then answer like a text message.
    voice = _voice_of(message)
    if voice is not None and config.enable_voice:
        if voice.duration > 300:
            return GuestReply(text="⚠️ Voice messages must be at most 5 minutes long.", mode="plain")
        if voice.file_size and voice.file_size > config.voice_max_bytes:
            return GuestReply(
                text="⚠️ The voice message is too large to recognize.", mode="plain"
            )
        audio = await _download(message, voice.file_id)
        transcript = None
        if audio:
            transcript = await ai_client.transcribe_audio(
                audio, mime=voice.mime_type or "audio/ogg"
            )
        if not transcript or not transcript.strip():
            return GuestReply(
                text="❌ Could not recognize the voice message.", mode="plain"
            )
        text = f"{text}\n{transcript}".strip() if text else transcript.strip()

    if not text:
        return GuestReply(
            text=(
                "Summon me with a text message, for example: "
                "`@bot what is the capital of France?`"
            ),
            mode="plain",
        )

    settings = await service.get_user_settings(caller_id)
    history = await service.get_history(chat_id, user_id=caller_id)
    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": settings["system_prompt"]},
        *history,
    ]

    await service.add_message(
        chat_id=chat_id,
        user_id=caller_id,
        role="user",
        content=text,
        model=settings["model"],
    )

    async def _save(content: str) -> None:
        await service.add_message(
            chat_id=chat_id,
            user_id=caller_id,
            role="assistant",
            content=content,
            model=settings["model"],
        )

    # Attached (or replied-to) photo -> vision answer.
    photo_bytes = await _download_photo(message)
    if photo_bytes is not None:
        if not is_vision_model(settings["model"]):
            return GuestReply(
                text=(
                    f"⚠️ Model `{settings['model']}` cannot analyze images. "
                    "Switch to a vision model with /models."
                ),
                mode="plain",
            )
        vision = await ai_client.chat_with_image_result(
            text or DEFAULT_VISION_PROMPT,
            photo_bytes,
            base_messages,
            model=settings["model"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
        )
        response = vision.content if vision else None
        if not response:
            return GuestReply(
                text="❌ Error analyzing the image. Please try again later.",
                mode="plain",
            )
        await _save(response)
        footer = format_timing(
            vision.elapsed, model=settings["model"], tokens=vision.total_tokens
        )
        telegraph_reply = await _maybe_telegraph_reply(text, response, footer)
        if telegraph_reply is not None:
            return telegraph_reply
        return GuestReply(text=with_timing(response + _guest_hint(access), footer))

    # Text request -> the model may answer with a generated image.
    messages = [*base_messages, {"role": "user", "content": text}]
    if config.enable_image_generation:
        result: ChatResult = await ai_client.chat_with_tools(
            messages=messages,
            model=settings["model"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
        ) or ChatResult()
    else:
        result = await ai_client.chat_completion_result(
            messages=messages,
            model=settings["model"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
        )
        result = result or ChatResult()

    image_call = next(
        (call for call in result.tool_calls if call.name == "generate_image"), None
    )
    if image_call is not None:
        prompt = str(image_call.arguments.get("prompt") or text).strip() or text
        image = await ai_client.generate_image_result(prompt, download=False)
        image_url = image.url if image else None
        if image_url:
            await _save(f"[Generated image] {prompt}")
            caption = result.content or f"🎨 {prompt}"
            footer = format_timing(
                image.elapsed if image else 0.0, model=config.image_model
            )
            return GuestReply(
                photo_url=image_url, caption=with_timing(caption, footer)
            )
        if not result.content:
            return GuestReply(
                text="❌ Could not generate the image. Please try again.",
                mode="plain",
            )

    if not result.content:
        return GuestReply(
            text="❌ Error communicating with the AI. Please try again later.",
            mode="plain",
        )

    await _save(result.content)
    footer = format_timing(
        result.elapsed, model=settings["model"], tokens=result.total_tokens
    )
    telegraph_reply = await _maybe_telegraph_reply(text, result.content, footer)
    if telegraph_reply is not None:
        return telegraph_reply
    return GuestReply(text=with_timing(result.content + _guest_hint(access), footer))


# --- Handler ---------------------------------------------------------------


@router.guest_message()
async def handle_guest_message(message: Message) -> None:
    """Handle a guest-mode summon: ``@bot`` in any chat or group.

    The bot is not a member of that chat, so the only way to reply is
    ``answerGuestQuery`` — and exactly one reply is allowed per summon.
    """
    if not config.enable_guest_mode:
        return
    if not getattr(message, "guest_query_id", None):
        return

    caller = getattr(message, "guest_bot_caller_user", None) or message.from_user
    if caller is None:
        return
    chat = getattr(message, "guest_bot_caller_chat", None) or message.chat
    if chat is None:
        return

    with limits.slot(caller.id) as granted:
        if granted:
            reply = await _run_guest_reply(message, caller, chat.id)
        else:
            reply = GuestReply(
                text="⏳ Your previous request is still being processed.",
                mode="plain",
            )

    # Try the configured formatting; if Telegram rejects it (bad entities) the
    # answer is retried once as plain text, so a guest reply is never lost to
    # formatting. An expired query simply fails both attempts and is logged.
    if await safe_answer_guest(message, _to_result(reply)):
        return
    if reply.mode != "plain":
        await safe_answer_guest(message, _to_result(reply, mode="plain"))


async def _run_guest_reply(message: Message, caller, chat_id: int) -> GuestReply:
    """Build the single reply within the time budget; only cancellation propagates."""
    session = async_session_factory()
    service = UserService(session)
    try:
        reply = await asyncio.wait_for(
            _build_reply(message, service, caller, chat_id),
            timeout=config.guest_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Guest request exceeded the total processing budget (%ss)",
            config.guest_timeout,
        )
        reply = GuestReply(
            text=(
                "⏳ Не удалось завершить запрос за отведённое время. "
                "Попробуйте ещё раз. Для генерации картинок откройте личный "
                "чат с ботом и используйте /image с описанием — там можно "
                "дождаться более долгой генерации."
            ),
            mode="plain",
        )
    except Exception as exc:  # noqa: BLE001 - always produce a single reply
        logger.exception("Guest message handling failed: %s", exc)
        reply = GuestReply(
            text="❌ Something went wrong. Please try again.", mode="plain"
        )
    finally:
        await session.close()
    return reply
