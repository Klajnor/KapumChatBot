from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from aiogram import Router
from aiogram.types import (
    InlineQueryResult,
    InlineQueryResultArticle,
    InputRichMessage,
    InputRichMessageContent,
    InputTextMessageContent,
    LinkPreviewOptions,
    Message,
)

from bot.config import config
from bot.database.session import async_session_factory
from bot.handlers.chat import (
    DEFAULT_LINK_PROMPT,
    DEFAULT_VIDEO_PROMPT,
    _compose_document_context,
    read_links,
)
from bot.services import limits
from bot.services.access import AccessResult, check_access, refund_access
from bot.services.ai_client import ChatResult, ai_client, is_vision_model
from bot.services.documents import detect_file_type, extract_document_text
from bot.services.intent import (
    VISION_FOR_GEN_PROMPT,
    detect_intent,
    generation_prompt,
)
from bot.services.markdown import render, split_chunks
from bot.services.rich import RICH_LIMIT, adapt_markdown
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
DEFAULT_VISION_PROMPT = "Что изображено на этой картинке?"


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
            text = (
                "Image generated, but the link is too long. "
                "Use /image in a private chat with the bot."
            )
        else:
            caption = (reply.caption or "🎨 Image is ready.")[:MAX_CAPTION_LENGTH]
            caption = caption[:max(0, MAX_TEXT_LENGTH - len(url) - 2)]
            text = f"{caption}\n\n{url}"
        return InlineQueryResultArticle(
            id="image",
            title="Open the generated image",
            input_message_content=InputTextMessageContent(
                message_text=text,
                parse_mode=None,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            ),
        )

    effective = mode or reply.mode
    text = reply.text or "…"
    if effective == "rich":
        # Native rich message: Telegram renders the model's Markdown (tables,
        # $$LaTeX$$) itself; the 32768-character budget beats plain text.
        body = adapt_markdown(text)
        if len(body) > RICH_LIMIT:
            body = body[: RICH_LIMIT - 1] + "…"
        return InlineQueryResultArticle(
            id="answer",
            title=" ".join(text.split())[:120] or "Answer",
            input_message_content=InputRichMessageContent(
                rich_message=InputRichMessage(markdown=body)
            ),
        )
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
    if config.rich_messages:
        # Rich guest answers render tables/formulas natively: no link needed.
        return None
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


def _document_of(message: Message):
    """Return the document attached to the message or to the replied one."""
    if message.document is not None:
        return message.document
    reply = message.reply_to_message
    return reply.document if reply is not None else None


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

    text = _extract_text(message)

    # Enforce maximum summon-text length to prevent abuse.
    if len(text) > config.max_guest_text_length:
        return GuestReply(
            text=(
                f"⚠️ The message is too long "
                f"({len(text)} characters; limit {config.max_guest_text_length}). "
                f"Please shorten it."
            ),
            mode="plain",
        )

    # Validate the voice attachment *before* consuming quota: a too-long or
    # too-large note must not cost the guest one of their free messages.
    voice = _voice_of(message)
    voice_ok = voice is not None and config.enable_voice
    if voice_ok:
        if voice.duration > 300:
            return GuestReply(
                text="⚠️ Voice messages must be at most 5 minutes long.",
                mode="plain",
            )
        if voice.file_size and voice.file_size > config.voice_max_bytes:
            return GuestReply(
                text="⚠️ The voice message is too large to recognize.",
                mode="plain",
            )

    # Same for an attached (or replied-to) document: reject an unsupported or
    # oversized file before it costs a free message.
    document = _document_of(message) if config.enable_document_analysis else None
    file_type = None
    if document is not None:
        file_type = detect_file_type(document.file_name, document.mime_type)
        limit_mb = config.max_document_bytes // (1024 * 1024)
        if file_type is None:
            return GuestReply(
                text=(
                    "⚠️ This file type cannot be analyzed. Supported: PDF, DOCX, "
                    "XLSX, PPTX, plain text and code."
                ),
                mode="plain",
            )
        if document.file_size and document.file_size > config.max_document_bytes:
            return GuestReply(
                text=f"⚠️ The file is too large. The limit is {limit_mb} MB.",
                mode="plain",
            )

    access = await check_access(service, caller_id)
    if not access.allowed:
        return GuestReply(
            text=access.message or get_random_themed_response(), mode="html"
        )

    # Document: extract its text (after the quota has been taken) and fold it,
    # together with the pages linked in the text, into the request.
    if document is not None:
        data = await _download(message, document.file_id)
        if not data:
            await refund_access(service, caller_id, access)
            return GuestReply(
                text=(
                    "❌ Telegram did not give me access to the attached file in "
                    "this chat. Send it to the bot in a private chat instead."
                ),
                mode="plain",
            )
        extracted = None
        if len(data) <= config.max_document_bytes:
            extracted = await extract_document_text(file_type, data)
        if not extracted:
            await refund_access(service, caller_id, access)
            return GuestReply(
                text="❌ Could not extract any text from this file.", mode="plain"
            )
        link_blocks: list[str] = []
        if config.enable_link_analysis and text:
            link_blocks, remainder, _ = await read_links(text, require_short=False)
            if link_blocks:
                text = remainder
        text = _compose_document_context(
            document.file_name or "document", extracted, text, link_blocks
        )
    elif config.enable_link_analysis and text:
        # A summons that is essentially bare links reads those pages.
        link_blocks, remainder, videos = await read_links(text)
        if link_blocks:
            request = remainder or (
                DEFAULT_VIDEO_PROMPT if videos == len(link_blocks) else DEFAULT_LINK_PROMPT
            )
            text = "\n\n".join(link_blocks) + f"\n\n---\n\nUser request: {request}"

    # Voice message: recognize it now (after the quota has been taken).
    if voice_ok:
        audio = await _download(message, voice.file_id)
        transcript = None
        if audio:
            transcript = await ai_client.transcribe_audio(
                audio, mime=voice.mime_type or "audio/ogg"
            )
        if not transcript or not transcript.strip():
            await refund_access(service, caller_id, access)
            return GuestReply(
                text="❌ Could not recognize the voice message.", mode="plain"
            )
        text = f"{text}\n{transcript}".strip() if text else transcript.strip()

    if not text:
        # Nothing to send to the model: give the quota unit back.
        await refund_access(service, caller_id, access)
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

    # Attached (or replied-to) photo: classify the intent, then answer.
    # Guest mode allows exactly one inline result and a guest bot cannot
    # upload files, so enhancement here is always neural: the re-drawn photo
    # comes back as a generated URL and is delivered as a link.
    photo_bytes = await _download_photo(message)
    if photo_bytes is not None:
        intent, enhance_ops = "analyze", []
        if text and config.enable_image_generation:
            intent, enhance_ops = await detect_intent(text, settings["model"])

        if intent == "enhance":
            result = await ai_client.enhance_image_via_ai(
                photo_bytes, operations=enhance_ops, caption=text, download=False
            )
            if result.url:
                await _save(f"[Enhanced image: neural] {text}")
                footer = format_timing(result.elapsed, model=config.image_model)
                return GuestReply(
                    photo_url=result.url,
                    caption=with_timing(
                        f"🧠 Enhanced by the neural network: {text[:200]}", footer
                    ),
                )
            await refund_access(service, caller_id, access)
            return GuestReply(
                text="❌ Could not enhance the image. Please try again.",
                mode="plain",
            )

        if intent == "generate":
            if not is_vision_model(settings["model"]):
                await refund_access(service, caller_id, access)
                return GuestReply(
                    text=(
                        f"⚠️ Model `{settings['model']}` cannot analyze the "
                        "reference. Ask in a private chat with a vision model."
                    ),
                    mode="plain",
                )
            vision = await ai_client.chat_with_image_result(
                VISION_FOR_GEN_PROMPT,
                photo_bytes,
                base_messages,
                model=settings["model"],
                temperature=0.3,
                max_tokens=1024,
            )
            description = (vision.content or "").strip() if vision else ""
            gen_prompt = generation_prompt(text, description or None)
            image = await ai_client.generate_image_result(
                gen_prompt, download=False, reference_images=[photo_bytes]
            )
            if image.url:
                await _save(f"[Generated image] {gen_prompt}")
                footer = format_timing(image.elapsed, model=config.image_model)
                return GuestReply(
                    photo_url=image.url,
                    caption=with_timing(f"🎨 {text[:200]}", footer),
                )
            await refund_access(service, caller_id, access)
            return GuestReply(
                text="❌ Could not generate the image. Please try again.",
                mode="plain",
            )

        if not is_vision_model(settings["model"]):
            await refund_access(service, caller_id, access)
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
            await refund_access(service, caller_id, access)
            return GuestReply(
                text="❌ Error analyzing the image. Please try again later.",
                mode="plain",
            )
        await _save(response)
        footer = format_timing(
            vision.elapsed,
            model=vision.model or settings["model"],
            tokens=vision.total_tokens,
            input_tokens=vision.input_tokens,
            output_tokens=vision.output_tokens,
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
            await refund_access(service, caller_id, access)
            return GuestReply(
                text="❌ Could not generate the image. Please try again.",
                mode="plain",
            )

    if not result.content:
        await refund_access(service, caller_id, access)
        return GuestReply(
            text="❌ Error communicating with the AI. Please try again later.",
            mode="plain",
        )

    await _save(result.content)
    footer = format_timing(
        result.elapsed,
        model=result.model or settings["model"],
        tokens=result.total_tokens,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
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

    # Attempt order: rich (native Markdown) → the configured formatting →
    # plain text, so a rejection of one form never loses the single allowed
    # guest reply. An expired query simply fails every attempt and is logged.
    if (
        config.rich_messages
        and reply.photo_url is None
        and reply.mode == "markdown"
        and reply.text
    ) and await safe_answer_guest(message, _to_result(reply, mode="rich")):
        return
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
    except TimeoutError:
        logger.warning(
            "Guest request exceeded the total processing budget (%ss)",
            config.guest_timeout,
        )
        reply = GuestReply(
            text=(
                "⏳ The request could not be completed within the time limit. "
                "Please try again. For image generation, open a private chat "
                "with the bot and use /image with a description — longer "
                "generation times are supported there."
            ),
            mode="plain",
        )
    except Exception:
        logger.exception("Guest message handling failed")
        reply = GuestReply(
            text="❌ Something went wrong. Please try again.", mode="plain"
        )
    finally:
        await session.close()
    return reply
