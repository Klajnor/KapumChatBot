from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from typing import Any

from aiogram import F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    Message,
    ReplyParameters,
    User,
)

from bot.bot import get_bot_username
from bot.config import config
from bot.database.session import async_session_factory
from bot.keyboards.inline import (
    get_ask_other_model_keyboard,
    get_resend_model_keyboard,
)
from bot.services import limits
from bot.services.access import AccessResult, check_access, guest_hint, refund_access
from bot.services.ai_client import (
    IMAGE_TOOL,
    AIStreamError,
    AIClient,
    ChatResult,
    ai_client,
    is_vision_model,
)
from bot.services.formulas import has_table, promote_complex_inline
from bot.services.progress import waiting
from bot.services.rich import (
    RICH_LIMIT,
    next_draft_id,
    send_rich,
    send_rich_draft,
    send_text_draft,
)
from bot.services.telegram_utils import (
    edit_formatted,
    format_timing,
    is_reply_to_bot,
    mentions_bot,
    safe_answer,
    safe_answer_callback,
    safe_delete,
    safe_edit_text,
    send_formatted,
    strip_bot_mention,
    throttle,
    with_timing,
)
from bot.services.telegraph import needs_telegraph, notice_text, publish_answer
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

router = Router()

MAX_MESSAGE_LENGTH = 4000  # Telegram hard limit is 4096; keep a safety margin.

DEFAULT_VISION_PROMPT = "What's in this image?"
MAX_PHOTO_BYTES = 10 * 1024 * 1024

DEFAULT_VOICE_PROMPT = "Answer what the user said in this voice message."

# Display math still reaches the PNG renderer, so the streamed text must not
# be edited into a formatted message (that would skip formula images).
_DISPLAY_MATH_RE = re.compile(r"\$\$|\\\[")


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


# --- "Ask another model" registry ------------------------------------------

# Last user question per private chat, kept in memory on purpose: after a bot
# restart the button simply asks the user to repeat the question.
_last_questions: dict[int, str] = {}
_LAST_QUESTIONS_CAP = 500


def _remember_question(chat_id: int, text: str) -> None:
    if not text:
        return
    _last_questions.pop(chat_id, None)
    _last_questions[chat_id] = text
    while len(_last_questions) > _LAST_QUESTIONS_CAP:
        _last_questions.pop(next(iter(_last_questions)))


def _retry_markup(message: Message) -> Any:
    """Inline keyboard offering to re-ask, private chats only (never guests)."""
    if config.enable_ask_other_model and _is_private(message):
        return get_ask_other_model_keyboard()
    return None


# --- Sending helpers -------------------------------------------------------


async def send_long_message(message: Message, text: str, *, reply_markup: Any = None) -> bool:
    """Send an AI answer rendered from Markdown, splitting long replies.

    Formatting falls back automatically (MarkdownV2 → HTML → plain text) and
    rate limits are retried, so a reply is not lost because Telegram rejected
    the entities. Returns False only if nothing at all could be delivered.
    """
    delivered = await send_formatted(
        message, text, limit=MAX_MESSAGE_LENGTH, reply_markup=reply_markup
    )
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


async def _open_registered_user(user: User) -> UserService:
    """Register ``user`` and return a service bound to an open session."""
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


async def _open_registered(message: Message) -> UserService | None:
    """Register the sender and return a service bound to an open session."""
    user = message.from_user
    if user is None:
        return None
    return await _open_registered_user(user)


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


# --- Streaming presentation ------------------------------------------------


class _StreamPrinter:
    """Shows streamed answer chunks live.

    In private chats with rich messages the preview is an ephemeral *rich
    draft* (``sendRichMessageDraft``): Telegram animates the partial Markdown
    itself and the draft vanishes once the final message is sent. Everywhere
    else the older behaviour stays: one placeholder message edited as plain
    text at a throttled pace, replaced by the rendered answer in the end.
    """

    def __init__(self, message: Message, notifier: Any = None) -> None:
        self.origin = message
        self.message: Message | None = None
        self._notifier = notifier
        self._last_edit = 0.0
        self._broken = False
        self.draft_mode = bool(config.rich_messages and _is_private(message))
        self._draft_id = next_draft_id() if self.draft_mode else 0
        self._thread_id = getattr(message, "message_thread_id", None)
        self.drafted = False

    async def on_delta(self, full: str) -> None:
        """Client callback: awaited with the text accumulated so far."""
        if self._broken or not full:
            return
        if self.draft_mode:
            if time.monotonic() - self._last_edit < config.stream_edit_interval:
                return
            await self._draft(full)
            return
        if self.message is None:
            await self._send_initial(full)
            self._last_edit = time.monotonic()
            if self._notifier is not None:
                # The live answer replaces the "waiting" notice.
                await self._notifier.stop()
            return
        if time.monotonic() - self._last_edit < config.stream_edit_interval:
            return
        if await safe_edit_text(self.message, self._clip(full), parse_mode=None):
            self._last_edit = time.monotonic()
        else:
            self._broken = True

    async def _draft(self, full: str) -> None:
        """Push one draft update; degrade to plain text, then stop trying."""
        if not self.drafted:
            self.drafted = True
            if self._notifier is not None:
                await self._notifier.stop()
        bot = getattr(self.origin, "bot", None)
        if bot is None:  # pragma: no cover - real messages always carry a bot
            self._broken = True
            return
        chat_id = self.origin.chat.id
        if await send_rich_draft(
            bot,
            chat_id,
            self._draft_id,
            self._clip(full, RICH_LIMIT),
            message_thread_id=self._thread_id,
        ):
            self._last_edit = time.monotonic()
        elif await send_text_draft(
            bot, chat_id, self._draft_id, self._clip(full),
            message_thread_id=self._thread_id,
        ):
            self._last_edit = time.monotonic()
        else:
            self._broken = True

    def _clip(self, full: str, limit: int = MAX_MESSAGE_LENGTH) -> str:
        if len(full) <= limit:
            return full
        return full[: limit - 1] + "…"

    async def _send_initial(self, text: str) -> None:
        kwargs: dict[str, Any] = {"parse_mode": None}
        if not _is_private(self.origin):
            kwargs["reply_parameters"] = ReplyParameters(
                message_id=self.origin.message_id
            )
        try:
            await throttle.wait(self.origin.chat.id)
            self.message = await self.origin.answer(self._clip(text), **kwargs)
        except Exception as exc:  # noqa: BLE001 - streaming is an optimization
            logger.warning("Streaming placeholder could not be sent: %s", exc)
            self.message = None
            self._broken = True

    async def finalize(self, raw_markdown: str, *, reply_markup: Any = None) -> bool:
        """Replace the placeholder with the fully formatted answer."""
        if self.message is None:
            return False
        return await edit_formatted(self.message, raw_markdown, reply_markup=reply_markup)

    async def discard(self) -> None:
        if self.message is not None:
            await safe_delete(self.message)
            self.message = None


# --- Delivery --------------------------------------------------------------


async def deliver_answer(
    message: Message,
    printer: _StreamPrinter | None,
    text: str,
    footer: str,
    *,
    title_source: str | None = None,
    reply_markup: Any = None,
) -> bool:
    """Deliver an AI answer.

    Preferred path: one native rich message (Bot API 10.1) — Telegram renders
    the model's Markdown itself: tables, LaTeX, headings, footnotes. When
    rich messages are off or Telegram rejects the content, the old chain
    takes over: Telegraph link for what Telegram used to be unable to show,
    formula PNGs, and the MarkdownV2/HTML formatting fallback — so the
    answer is never lost.
    """
    if config.rich_messages:
        if printer is not None and printer.message is not None:
            # The placeholder served the stream; the rich answer is a new
            # message, so remove the stale preview before it duplicates.
            await printer.discard()
        body = with_timing(text, footer)
        if await send_rich(message, body, reply_markup=reply_markup):
            return True
        logger.warning("Rich delivery failed; falling back to the legacy chain")

    if config.render_formula_images:
        text = promote_complex_inline(text)
    if config.enable_telegraph and needs_telegraph(
        text, tables_as_images=config.render_formula_images
    ):
        url = await publish_answer(title_source or text, text)
        if url:
            body = with_timing(notice_text(url), footer)
            if printer is not None and printer.message is not None:
                if await safe_edit_text(
                    printer.message, body, parse_mode=None, reply_markup=reply_markup
                ):
                    return True
                await printer.discard()
            return await safe_answer(
                message, body, parse_mode=None, reply_markup=reply_markup
            )
        logger.warning("Telegraph publishing failed; delivering the answer in Telegram as is")

    body = with_timing(text, footer)
    if (
        printer is not None
        and printer.message is not None
        # Formula PNGs are produced only on the send path, so answers still
        # containing display math must not be finalized by a plain edit.
        and not (
            config.render_formula_images
            and (_DISPLAY_MATH_RE.search(body) or has_table(body))
        )
    ):
        if await printer.finalize(body, reply_markup=reply_markup):
            return True
        await printer.discard()
    return await send_long_message(message, body, reply_markup=reply_markup)


# --- Core answer pipeline --------------------------------------------------


async def _single_shot(
    messages: list[dict[str, Any]],
    model: str,
    settings: dict[str, Any],
    tools: list[dict[str, Any]] | None,
) -> ChatResult:
    """One non-streaming request (also the fallback when a stream fails)."""
    if tools is not None:
        return await ai_client.chat_with_tools(
            messages=messages,
            model=model,
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
            tools=tools,
        )
    return await ai_client.chat_completion_result(
        messages=messages,
        model=model,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )


async def _answer_with_ai(
    message: Message,
    service: UserService,
    text: str,
    access: AccessResult,
    *,
    user: User | None = None,
    model_override: str | None = None,
) -> None:
    """Produce the assistant reply for ``text``: a message and/or an image.

    ``model_override`` answers once with another model without touching the
    user's stored setting; ``user`` lets callback handlers pass the pressing
    user explicitly (a bot message's ``from_user`` is the bot itself).
    """
    user = user or message.from_user
    if user is None:
        return

    settings = await service.get_user_settings(user.id)
    model = model_override or settings["model"]
    history = await service.get_history(message.chat.id, user_id=user.id)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": settings["system_prompt"]},
        *history,
        {"role": "user", "content": text},
    ]

    if _is_private(message):
        _remember_question(message.chat.id, text)

    await service.add_message(
        chat_id=message.chat.id,
        user_id=user.id,
        role="user",
        content=text,
        model=model,
    )

    tools = [IMAGE_TOOL] if config.enable_image_generation else None
    printer: _StreamPrinter | None = None

    async with waiting(message) as notifier:
        if config.stream_responses:
            printer = _StreamPrinter(message, notifier)
            try:
                result: ChatResult = await ai_client.stream_chat_completion(
                    messages,
                    model=model,
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                    tools=tools,
                    on_delta=printer.on_delta,
                )
            except AIStreamError as exc:
                logger.warning(
                    "Streaming failed (%s); retrying as a single request", exc
                )
                await printer.discard()
                result = await _single_shot(messages, model, settings, tools)
        else:
            result = await _single_shot(messages, model, settings, tools)

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
            model=model,
        )
        footer = format_timing(
            result.elapsed,
            model=model,
            tokens=result.total_tokens,
        )
        await deliver_answer(
            message,
            printer,
            result.content,
            footer,
            title_source=text,
            reply_markup=_retry_markup(message),
        )
    elif image_call is None and printer is not None:
        await printer.discard()

    if image_call is not None:
        prompt = str(image_call.arguments.get("prompt") or text).strip() or text
        if not await generate_and_send_image(
            message,
            prompt,
            service=service,
            user_id=user.id,
            model=model,
        ) and not result.content:
            await refund_access(service, user.id, access)
    elif not result.content:
        await refund_access(service, user.id, access)
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

    with limits.slot(user.id) as granted:
        if not granted:
            await safe_answer(message, limits.BUSY_MESSAGE)
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


async def _guarded(message: Message, inner: Any) -> None:
    """Run ``inner(message)`` inside the per-user concurrency cap."""
    user = message.from_user
    if user is None:
        return
    with limits.slot(user.id) as granted:
        if not granted:
            await safe_answer(message, limits.BUSY_MESSAGE)
            return
        await inner(message)


@router.message(F.voice)
async def handle_voice(message: Message) -> None:
    """Voice messages, one at a time per user."""
    await _guarded(message, _handle_voice)


async def _handle_voice(message: Message) -> None:
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
            await refund_access(service, user.id, access)
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
    """Photos, one request at a time per user."""
    await _guarded(message, _handle_photo)


async def _handle_photo(message: Message) -> None:
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
        if photo.file_size and photo.file_size > MAX_PHOTO_BYTES:
            await refund_access(service, user.id, access)
            await safe_answer(message, "⚠️ <b>The image is too large</b> (limit 10 MB).")
            return
        file = await message.bot.get_file(photo.file_id)
        buffer = await message.bot.download_file(file.file_path)
        image_bytes = buffer.read()

        history = await service.get_history(message.chat.id, user_id=user.id)
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

        printer: _StreamPrinter | None = None
        async with waiting(message) as notifier:
            if config.stream_responses:
                printer = _StreamPrinter(message, notifier)
                try:
                    vision_result: ChatResult = await ai_client.stream_chat_completion(
                        AIClient.vision_messages(
                            messages, caption, image_bytes
                        ),
                        model=settings["model"],
                        temperature=settings["temperature"],
                        max_tokens=settings["max_tokens"],
                        on_delta=printer.on_delta,
                    )
                except AIStreamError as exc:
                    logger.warning(
                        "Vision streaming failed (%s); retrying as a single request",
                        exc,
                    )
                    await printer.discard()
                    vision_result = await ai_client.chat_with_image_result(
                        text=caption,
                        image_bytes=image_bytes,
                        messages=messages,
                        model=settings["model"],
                        temperature=settings["temperature"],
                        max_tokens=settings["max_tokens"],
                    )
            else:
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
            if printer is not None:
                await printer.discard()
            await refund_access(service, user.id, access)
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
        # No "Ask another model" button: re-asking needs the image attached.
        await deliver_answer(
            message, printer, response, footer, title_source=caption
        )

        hint = guest_hint(access)
        if hint:
            await safe_answer(message, hint)
    finally:
        await service.session.close()


# --- "Ask another model" callbacks -----------------------------------------


@router.callback_query(F.data == "askother")
async def cb_ask_other_model(callback: CallbackQuery) -> None:
    """Offer the model picker to re-ask the last question once."""
    if callback.from_user is None or callback.message is None:
        await safe_answer_callback(callback)
        return

    origin = callback.message
    if not _is_private(origin):
        await safe_answer_callback(
            callback, "⚠️ This button works in private chats only.", show_alert=True
        )
        return

    question = _last_questions.get(origin.chat.id)
    if not question:
        await safe_answer_callback(
            callback,
            "⚠️ I cannot find your previous question here — please ask it again.",
            show_alert=True,
        )
        return

    await safe_answer_callback(callback, "Choose a model")
    preview = question if len(question) <= 200 else question[:200] + "…"
    await origin.answer(
        "🤖 <b>Ask the same question with another model:</b>\n\n"
        f"<blockquote>{html.escape(preview)}</blockquote>\n\n"
        "The question is re-asked once; your default model stays unchanged.",
        reply_markup=get_resend_model_keyboard(),
    )


@router.callback_query(F.data.startswith("resend:"))
async def cb_resend_with_model(callback: CallbackQuery) -> None:
    """Re-send the stored last question to the chosen model (one-time)."""
    if callback.data is None or callback.from_user is None or callback.message is None:
        await safe_answer_callback(callback)
        return

    model_id = callback.data.split(":", 1)[1]
    origin = callback.message
    if not _is_private(origin):
        await safe_answer_callback(
            callback, "⚠️ This button works in private chats only.", show_alert=True
        )
        return

    available = {m["id"]: m["name"] for m in ai_client.available_models()}
    if model_id not in available:
        await safe_answer_callback(
            callback, "⚠️ This model is not available for your account.", show_alert=True
        )
        return

    question = _last_questions.get(origin.chat.id)
    if not question:
        await safe_answer_callback(
            callback,
            "⚠️ I cannot find your previous question here — please ask it again.",
            show_alert=True,
        )
        return

    await safe_answer_callback(callback, f"Asking {available[model_id]}…")
    await safe_edit_text(
        origin,
        "🔀 <b>Re-asking with</b> "
        f"<code>{html.escape(available[model_id])}</code>…",
    )

    with limits.slot(callback.from_user.id) as granted:
        if not granted:
            await safe_answer(origin, limits.BUSY_MESSAGE)
            return
        service = await _open_registered_user(callback.from_user)
        try:
            access = await check_access(service, callback.from_user.id)
            if not access.allowed:
                await origin.answer(access.message or get_random_themed_response())
                return
            await _answer_with_ai(
                origin,
                service,
                question,
                access,
                user=callback.from_user,
                model_override=model_id,
            )
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
