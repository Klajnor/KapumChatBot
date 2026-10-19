from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from aiogram import Router
from aiogram.types import (
    InlineQueryResult,
    InlineQueryResultArticle,
    InlineQueryResultPhoto,
    InputTextMessageContent,
    Message,
)

from bot.config import config
from bot.database.session import async_session_factory
from bot.services.access import AccessResult, check_access
from bot.services.ai_client import ChatResult, ai_client, is_vision_model
from bot.services.markdown import render
from bot.services.telegram_utils import (
    PARSE_MODES,
    safe_answer_guest,
    strip_bot_mention,
)
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
        return InlineQueryResultPhoto(
            id="image",
            photo_url=reply.photo_url,
            thumbnail_url=reply.photo_url,
            caption=(reply.caption or "")[:MAX_CAPTION_LENGTH] or None,
        )

    effective = mode or reply.mode
    text = reply.text or "…"
    return InlineQueryResultArticle(
        id="answer",
        title=" ".join(text.split())[:120] or "Answer",
        input_message_content=InputTextMessageContent(
            message_text=render(text, effective)[:MAX_TEXT_LENGTH],
            parse_mode=PARSE_MODES.get(effective),
        ),
    )


def _guest_hint(access: AccessResult) -> str:
    """Plain-text reminder about remaining free-trial messages."""
    if access.role == "guest" and access.remaining is not None and access.remaining <= 2:
        return f"\n\n_Free trial messages left: {access.remaining}_"
    return ""


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


async def _guarded(coro, default: Any = None) -> Any:
    """Await a coroutine with a hard timeout (a guest query allows one reply)."""
    try:
        return await asyncio.wait_for(coro, timeout=config.guest_timeout)
    except asyncio.TimeoutError:
        logger.warning("Guest request timed out after %ss", config.guest_timeout)
        return default


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
        if voice.file_size and voice.file_size > config.voice_max_bytes:
            return GuestReply(
                text="⚠️ The voice message is too large to recognize.", mode="plain"
            )
        audio = await _download(message, voice.file_id)
        transcript = None
        if audio:
            transcript = await _guarded(
                ai_client.transcribe_audio(
                    audio, mime=voice.mime_type or "audio/ogg"
                )
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
    history = await service.get_history(chat_id)
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
        response = await _guarded(
            ai_client.chat_completion_with_image(
                text or DEFAULT_VISION_PROMPT,
                photo_bytes,
                base_messages,
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            )
        )
        if not response:
            return GuestReply(
                text="❌ Error analyzing the image. Please try again later.",
                mode="plain",
            )
        await _save(response)
        return GuestReply(text=response + _guest_hint(access))

    # Text request -> the model may answer with a generated image.
    messages = [*base_messages, {"role": "user", "content": text}]
    if config.enable_image_generation:
        result: ChatResult = await _guarded(
            ai_client.chat_with_tools(
                messages=messages,
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            ),
            default=ChatResult(),
        ) or ChatResult()
    else:
        content = await _guarded(
            ai_client.chat_completion(
                messages=messages,
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            )
        )
        result = ChatResult(content=content)

    image_call = next(
        (call for call in result.tool_calls if call.name == "generate_image"), None
    )
    if image_call is not None:
        prompt = str(image_call.arguments.get("prompt") or text).strip() or text
        image_url = await _guarded(ai_client.generate_image_url(prompt))
        if image_url:
            await _save(f"[Generated image] {prompt}")
            return GuestReply(
                photo_url=image_url, caption=result.content or f"🎨 {prompt}"
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
    return GuestReply(text=result.content + _guest_hint(access))


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

    session = async_session_factory()
    service = UserService(session)
    try:
        reply = await _build_reply(message, service, caller, chat.id)
    except Exception as exc:  # noqa: BLE001 - always produce a single reply
        logger.exception("Guest message handling failed: %s", exc)
        reply = GuestReply(
            text="❌ Something went wrong. Please try again.", mode="plain"
        )
    finally:
        await session.close()

    if await safe_answer_guest(message, _to_result(reply)):
        return

    # Most likely a formatting rejection: retry without any markup.
    await safe_answer_guest(message, _to_result(reply, mode="plain"))