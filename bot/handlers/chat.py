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

from bot.bot import get_bot_id, get_bot_username
from bot.config import config
from bot.database.session import async_session_factory
from bot.keyboards.inline import (
    get_answer_keyboard,
    get_enhance_choice_keyboard,
    get_resend_model_keyboard,
)
from bot.services import limits
from bot.services.access import AccessResult, check_access, guest_hint, refund_access
from bot.services.ai_client import (
    IMAGE_TOOL,
    STICKER_TOOL,
    AIClient,
    AIStreamError,
    ChatResult,
    ToolCall,
    _image_data_url,
    ai_client,
    is_vision_model,
)
from bot.services.documents import (
    FetchedPage,
    clip_extracted,
    detect_file_type,
    extract_document_text,
    fetch_url_text,
)
from bot.services.formulas import has_table, promote_complex_inline
from bot.services.image_tools import enhance_image
from bot.services.intent import (
    VISION_FOR_GEN_PROMPT,
    detect_intent,
    generation_prompt,
)
from bot.services.mcp import (
    apply_mcp_hint,
    mcp_available,
    mcp_manager,
    remove_mcp_hint,
)
from bot.services.pdf_ocr import ocr_pdf
from bot.services.progress import waiting
from bot.services.rich import (
    RICH_LIMIT,
    next_draft_id,
    send_rich,
    send_rich_draft,
    send_text_draft,
)
from bot.services.search import (
    SEARCH_TOOL,
    apply_search_hint,
    build_assistant_tool_message,
    build_tool_response,
    format_search_results,
    search_configured,
    search_web,
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
from bot.services.untrusted import (
    apply_untrusted_hint,
    messages_have_untrusted_content,
    wrap_untrusted,
)
from bot.services.user_service import UserService
from bot.services.youtube import (
    YouTubeError,
    describe,
    fetch_transcript,
    youtube_video_id,
)

logger = logging.getLogger(__name__)

router = Router()

MAX_MESSAGE_LENGTH = 4000  # Telegram hard limit is 4096; keep a safety margin.

DEFAULT_VISION_PROMPT = "Что изображено на этой картинке?"

DEFAULT_VOICE_PROMPT = "Ответь на то, что пользователь сказал в этом голосовом сообщении."

# Default ask when a document/link arrives without the user's own question.
DEFAULT_VIDEO_PROMPT = (
    "Сделай краткое изложение этого видео по субтитрам: основные темы, "
    "ключевые моменты (с таймкодами) и выводы."
)
DEFAULT_DOCUMENT_PROMPT = (
    "Сделай краткое изложение этого документа: основные темы, ключевые "
    "моменты и выводы."
)

DEFAULT_DOCUMENT_LINK_PROMPT = (
    "Сделай краткое изложение этого документа и приложенных страниц по "
    "ссылкам: основные темы, ключевые моменты и выводы по каждому источнику."
)

DEFAULT_LINK_PROMPT = (
    "Сделай краткое изложение этой страницы: основные темы, ключевые моменты "
    "и выводы."
)

MSG_UNSUPPORTED_DOCUMENT = (
    "📎 <b>This file type cannot be analyzed.</b>\n\n"
    "Supported formats: <b>PDF</b>, <b>DOCX</b>, <b>XLSX</b>, <b>PPTX</b>, plain text and code "
    "(TXT, MD, CSV, JSON, PY, …).\n"
    "Scanned PDFs without a text layer are not supported — send the page "
    "as a photo for vision analysis instead.\n\n"
    "Maximum file size: {limit} MB."
)

# Shown when the selected model cannot process images.
MSG_NO_VISION = (
    "⚠️ <b>The current model does not support images.</b>\n\n"
    "Model <code>{model}</code> cannot process images. "
    "Switch to a vision-capable model with /models.\n\n"
    "<b>Vision models</b> are marked 🖼️ in the list "
    "(for example <code>qwen3.8-max</code> or <code>qwen3.8-flash</code>)."
)

MSG_ENHANCE_CHOICE = (
    "🔧 <b>How should I improve the photo?</b>\n\n"
    "🖥️ <b>Server</b> — processed locally (fast, no AI cost): upscale, "
    "denoise, sharpen, colour; the content is never altered.\n"
    "🧠 <b>Neural network</b> — the photo is re-drawn to restore lost "
    "detail; slower, may slightly change small details."
)

# Display math still reaches the PNG renderer, so the streamed text must not
# be edited into a formatted message (that would skip formula images).
_DISPLAY_MATH_RE = re.compile(r"\$\$|\\\[")


# --- Media group (album) support -------------------------------------------

# Collect photos that arrive in a single album; each group_id maps to a dict
# with ``images`` (list of bytes) and an ``event`` that processing awaits.
_album_images: dict[str, dict[str, Any]] = {}
_album_lock: asyncio.Lock = asyncio.Lock()
_album_total_bytes: dict[str, int] = {}  # per-key running total for size checks
_ALBUM_COLLECT_SECONDS = 1.5  # how long to wait for the next photo
_ALBUM_CLEANUP_SECONDS = 300  # discard stale album entries after 5 minutes
# Album buffers hold raw photo bytes in RAM for up to _ALBUM_CLEANUP_SECONDS
# and are collected before any per-user slot is taken, so a flood of media
# groups must not be able to exhaust memory. New albums are refused once the
# global budget is spent; every appended photo re-checks the real byte count.
_ALBUM_MAX_CONCURRENT = 8
_ALBUM_MAX_PER_USER = 2
_ALBUM_MAX_TOTAL_BYTES = 200 * 1024 * 1024
_ALBUM_BUDGET_NOTICE = (
    "⏳ <b>Too many albums are being processed.</b>\n\n"
    "Please wait a moment and send the photos again."
)


def _album_bytes_in_flight() -> int:
    """Total bytes currently buffered across all in-flight albums."""
    return sum(_album_total_bytes.values())


async def _album_cleanup() -> None:
    """Periodically remove stale album entries to prevent memory leaks."""
    while True:
        await asyncio.sleep(_ALBUM_CLEANUP_SECONDS)
        cutoff = time.monotonic() - _ALBUM_CLEANUP_SECONDS
        async with _album_lock:
            stale = [k for k, v in _album_images.items() if v["last"] < cutoff]
            for key in stale:
                _album_images.pop(key, None)
                _album_total_bytes.pop(key, None)


async def _touch_album(key: str, stamp: float) -> None:
    """Mark an album as recently touched so this handler may finalize it.

    Called for rejected photos too: without it an album whose *last* photo was
    oversized would never be processed and the valid photos would be lost.
    """
    async with _album_lock:
        entry = _album_images.get(key)
        if entry is not None:
            entry["last"] = stamp


def _album_key(chat_id: int, group_id: str) -> str:
    return f"{chat_id}:{group_id}"


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


# --- Quoted / forwarded context ---------------------------------------------

# A quoted message is included in the AI request verbatim; keep it bounded so
# one huge forward cannot crowd out the conversation history.
_MAX_QUOTED_CHARS = 4000


def _sender_name(message: Message) -> str:
    """Human-readable sender of a message (for quoted/forwarded context)."""
    sender_chat = getattr(message, "sender_chat", None)
    if sender_chat is not None:
        return sender_chat.title or "channel"
    user = message.from_user
    if user is None:
        return "unknown"
    return user.first_name or user.username or "user"


def _describe_quoted(reply: Message | None) -> str | None:
    """Render the replied-to message as a context block for the AI.

    Returns None when there is nothing worth quoting: no reply at all, a
    reply to a photo (the vision pipeline attaches the image itself) or a
    reply to the bot's own answer (already part of the user's history).
    """
    if reply is None or reply.photo:
        return None
    bot_id = get_bot_id()
    if bot_id != 0 and reply.from_user is not None and reply.from_user.id == bot_id:
        return None

    parts: list[str] = []
    text = reply.text or reply.caption or ""
    if text:
        parts.append(text)
    if reply.voice is not None:
        parts.append("[Voice message]")
    if reply.video is not None:
        parts.append("[Video]")
    if reply.animation is not None:
        parts.append("[Animation]")
    if reply.document is not None:
        parts.append(f"[Document: {reply.document.file_name or 'file'}]")
    if reply.sticker is not None:
        emoji = reply.sticker.emoji or ""
        parts.append(f"[Sticker{' ' + emoji if emoji else ''}]")
    if not parts:
        return None

    body = "\n".join(parts)
    if len(body) > _MAX_QUOTED_CHARS:
        body = body[:_MAX_QUOTED_CHARS] + "…"
    # The quoted text was written by somebody else: frame it as data.
    return f"[Quoted message from {_sender_name(reply)}]\n{wrap_untrusted('quoted', body)}"


def _forward_source_name(message: Message) -> str | None:
    """Where a forwarded message came from, when Telegram tells us."""
    origin = getattr(message, "forward_origin", None)
    if origin is None:
        return None
    sender = getattr(origin, "sender_user", None)
    if sender is not None:
        return sender.first_name or getattr(sender, "username", None) or "user"
    chat = getattr(origin, "sender_chat", None) or getattr(origin, "chat", None)
    if chat is not None:
        return getattr(chat, "title", None) or "chat"
    return None


def _with_quoted_context(message: Message, text: str) -> str:
    """Prepend quoted-reply and forwarding provenance to the question.

    The combined text is what the AI sees and what is stored in history, so
    follow-up questions keep working after the original leaves the window.
    Both the quoted body and a forwarded message are third-party content and
    are wrapped as untrusted (see ``bot/services/untrusted.py``).
    """
    quoted = _describe_quoted(message.reply_to_message)
    if quoted:
        text = f"{quoted}\n\n---\n\n{text}"
    if message.forward_date is not None:
        source = _forward_source_name(message)
        label = (
            f"[Forwarded message from {source}]" if source else "[Forwarded message]"
        )
        text = f"{label}\n{wrap_untrusted('forward', text)}"
    return text


# --- Document & link analysis -------------------------------------------------

# URLs inside a text message; trailing punctuation is stripped by _clean_url.
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_MAX_LINKS_PER_MESSAGE = 2
# A message with more non-URL text than this is a discussion that merely
# contains links, not a "read this" request — fetch nothing, pollute nothing.
_LINK_REMAINDER_LIMIT = 300


def _clean_url(candidate: str) -> str:
    """Strip sentence punctuation from a URL match, keeping balanced brackets.

    ``https://example.com/a_(b).`` → ``https://example.com/a_(b)``: a closing
    bracket is only removed when it has no opening partner inside the URL.
    """
    openers = {")": "(", "]": "[", "}": "{"}
    cleaned = candidate
    while cleaned:
        last = cleaned[-1]
        if last in ".,;:!?\"'":
            drop = True
        elif last in openers:
            drop = cleaned.count(openers[last]) < cleaned.count(last)
        else:
            drop = False
        if not drop:
            break
        cleaned = cleaned[:-1]
    return cleaned


def _compose_document_context(
    name: str,
    text: str,
    question: str,
    link_blocks: list[str] | None = None,
) -> str:
    """The AI request for an uploaded document: content plus the user's ask.

    The extracted document body is third-party content, so it is wrapped in
    untrusted-content tags (see ``bot/services/untrusted.py``): a malicious
    PDF must not be able to inject instructions into the conversation.
    ``link_blocks`` are pages fetched from links in the caption; they travel
    with the document so both are analyzed together.
    """
    default = DEFAULT_DOCUMENT_LINK_PROMPT if link_blocks else DEFAULT_DOCUMENT_PROMPT
    request = question.strip() or default
    document = wrap_untrusted("document", f"[Document: {name}]\n\n{text}")
    parts = [document, *(link_blocks or [])]
    return "\n\n".join(parts) + f"\n\n---\n\nUser request: {request}"


def _find_urls(text: str) -> list[str]:
    """Distinct cleaned URLs of ``text`` in order of appearance."""
    urls: list[str] = []
    for candidate in _URL_RE.findall(text):
        cleaned = _clean_url(candidate)
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


async def read_links(
    text: str,
    notify: Any = None,
    *,
    require_short: bool = True,
) -> tuple[list[str], str, int]:
    """Fetch the pages linked in ``text``: ``(blocks, remainder, videos)``.

    ``blocks`` are untrusted-wrapped page bodies (YouTube links give
    transcripts); ``remainder`` is the text without the URLs, i.e. the user's
    own question. Pages reached from a linked page (one level deep, at most
    ``LINK_CRAWL_PAGES`` per message) follow their parent. ``notify`` is an
    optional ``async (str)`` for progress notes. With ``require_short`` a
    message that is mostly prose yields nothing: it merely mentions links.
    """
    urls = _find_urls(text)
    remainder = _URL_RE.sub(" ", text).strip()
    if not urls or (require_short and len(remainder) > _LINK_REMAINDER_LIMIT):
        return [], remainder, 0
    urls = urls[:_MAX_LINKS_PER_MESSAGE]

    async def say(note: str) -> None:
        if notify is not None:
            await notify(note)

    listed = "\n".join(f"<code>{html.escape(u[:120])}</code>" for u in urls)
    await say(f"🔗 <b>Reading {len(urls)} link(s)…</b>\n{listed}")

    blocks: list[str] = []
    videos = 0
    seen = set(urls)
    crawl_left = config.link_crawl_pages
    crawled = 0
    for url in urls:
        video_id = youtube_video_id(url) if config.enable_youtube else None
        if video_id is not None:
            try:
                transcript = await fetch_transcript(video_id)
            except YouTubeError as exc:
                await say(f"⚠️ <b>YouTube:</b> {html.escape(str(exc))}")
                continue
            videos += 1
            blocks.append(
                wrap_untrusted("web", f"{describe(transcript)}\n\n{transcript.text}")
            )
            continue
        page = await fetch_url_text(url, with_links=crawl_left > 0)
        if page is None:
            continue
        header = f"[Web page: {page.url}]"
        if page.title:
            header += f"\nTitle: {page.title}"
        blocks.append(wrap_untrusted("web", f"{header}\n\n{page.text}"))

        # One level deeper: the pages this page links to.
        children = [
            link for link in page.links
            if link not in seen and youtube_video_id(link) is None
        ][:crawl_left]
        if not children:
            continue
        seen.update(children)
        crawl_left -= len(children)
        sub_limit = max(2000, config.max_extracted_chars // 3)
        subpages = await asyncio.gather(
            *(fetch_url_text(child) for child in children), return_exceptions=True
        )
        for child in subpages:
            if not isinstance(child, FetchedPage):
                continue
            crawled += 1
            sub_header = f"[Linked page: {child.url}]\n(found on {page.url})"
            if child.title:
                sub_header += f"\nTitle: {child.title}"
            blocks.append(
                wrap_untrusted("web", f"{sub_header}\n\n{child.text[:sub_limit]}")
            )
    if crawled:
        await say(f"🔗 Also read <b>{crawled}</b> page(s) linked from there.")
    return blocks, remainder, videos


async def _expand_links_with_content(message: Message, text: str) -> str:
    """Replace a URL-centric message with the fetched page content.

    Returns ``text`` unchanged when there is nothing worth fetching, the
    message is mostly prose, or every fetch failed — the model still answers,
    just without the page body. Fetching happens after the access check, so
    only authorized users can make the server issue HTTP requests.

    Page bodies are wrapped as untrusted content: a page that tries to steer
    the model ("ignore your rules…") must stay data.
    """
    async def notify(note: str) -> None:
        await safe_answer(message, note)

    blocks, remainder, videos = await read_links(text, notify)
    if not blocks:
        if _find_urls(text) and len(remainder) <= _LINK_REMAINDER_LIMIT:
            await safe_answer(
                message,
                "⚠️ <b>Could not read the linked page(s)</b> — answering without them.",
            )
        return text

    request = remainder or (
        DEFAULT_VIDEO_PROMPT if videos == len(blocks) else DEFAULT_LINK_PROMPT
    )
    return "\n\n".join(blocks) + f"\n\n---\n\nUser request: {request}"


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


# --- Edited-message notices -------------------------------------------------

# Editing is free and repeatable, so the "message edited" notice is limited to
# one per chat per cooldown: without it a user editing a message in a loop
# would make the bot answer once per second indefinitely.
_edit_notice_last: dict[int, float] = {}
_EDIT_NOTICE_COOLDOWN = 30.0
_EDIT_NOTICE_CAP = 500


def _edit_notice_allowed(chat_id: int) -> bool:
    """Return True (and record the time) when an edit notice may be sent."""
    now = time.monotonic()
    last = _edit_notice_last.get(chat_id, 0.0)
    if now - last < _EDIT_NOTICE_COOLDOWN:
        return False
    _edit_notice_last.pop(chat_id, None)
    _edit_notice_last[chat_id] = now
    while len(_edit_notice_last) > _EDIT_NOTICE_CAP:
        _edit_notice_last.pop(next(iter(_edit_notice_last)))
    return True


# --- Pending "server or neural?" enhance choices ----------------------------

# Telegram user id -> clarify-question context (photo file ids, caption,
# operations). Bytes are not cached: the stored ``file_id`` values are
# re-downloaded when the user presses a button, which works long after the
# original upload. One active question per user — a newer photo replaces it.
_pending_enhance: dict[int, dict[str, Any]] = {}
_PENDING_ENHANCE_TTL_SECONDS = 600


def _remember_pending_enhance(
    user_id: int, file_ids: list[str], caption: str, operations: list[str]
) -> None:
    now = time.monotonic()
    _pending_enhance[user_id] = {
        "file_ids": file_ids,
        "caption": caption,
        "operations": operations,
        "expires": now + _PENDING_ENHANCE_TTL_SECONDS,
    }
    # Opportunistic cleanup of everything that has gone stale.
    for uid in [u for u, item in _pending_enhance.items() if item["expires"] < now]:
        _pending_enhance.pop(uid, None)


def _pop_pending_enhance(user_id: int) -> dict[str, Any] | None:
    entry = _pending_enhance.pop(user_id, None)
    if entry is None or entry["expires"] < time.monotonic():
        return None
    return entry


def _retry_markup(message: Message, export_id: int | None = None) -> Any:
    """Inline keyboard under an answer: 📄/📝 export + re-ask (private only).

    ``export_id`` is the database id of the saved assistant message; the
    export buttons embed it in their callback data so the handler can reload
    the exact answer with an owner check.
    """
    if not _is_private(message):
        return None
    return get_answer_keyboard(
        export_id=export_id if config.enable_export else None,
        ask_other=config.enable_ask_other_model,
    )


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


async def generate_and_send_sticker(
    message: Message,
    prompt: str,
    *,
    service: UserService | None = None,
    user_id: int | None = None,
    model: str | None = None,
) -> bool:
    """Generate a sticker from ``prompt`` and send it. Returns True on success.

    The image is produced by the model, then converted to a 512×512 WebP and
    sent as a real Telegram sticker (``answer_sticker``), so the user can save
    it or add it to a sticker set. Stickers carry no caption, so the timing
    footer (when enabled) is sent as a short follow-up reply.
    """
    await message.bot.send_chat_action(
        chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO
    )
    async with waiting(message):
        result = await ai_client.generate_sticker(prompt)
    sticker = result.data

    if sticker is None:
        await safe_answer(
            message,
            "❌ <b>Could not generate the sticker.</b>\n\n"
            "The prompt may have been rejected by moderation, or the image "
            "service / model is unavailable for your account.",
        )
        return False

    sent = False
    for attempt in range(2):
        try:
            await throttle.wait(message.chat.id)
            await message.answer_sticker(
                BufferedInputFile(sticker, filename="sticker.webp")
            )
            sent = True
            break
        except TelegramRetryAfter as exc:
            if attempt >= 1:
                break
            await asyncio.sleep(float(exc.retry_after) + 0.1)
        except TelegramAPIError as exc:
            logger.error("Sending the sticker failed: %s", exc)
            break

    if not sent:
        await safe_answer(
            message,
            "⚠️ <b>The sticker was generated but could not be sent.</b> "
            "Please try again in a moment.",
        )
        return False

    footer = format_timing(result.elapsed, model=config.image_model)
    if footer:
        await safe_answer(
            message,
            f"🎨 <i>{html.escape(prompt[:300])}</i>\n\n{html.escape(footer)}",
        )

    if service is not None and user_id is not None:
        await service.add_message(
            chat_id=message.chat.id,
            user_id=user_id,
            role="assistant",
            content=f"[Generated sticker] {prompt}",
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
        ) or await send_text_draft(
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


def _mcp_notice(calls: list[ToolCall]) -> str:
    """Short user-facing line describing the pending MCP tool calls."""
    labels: list[str] = []
    for call in calls:
        tool = mcp_manager.tool_info(call.name)
        label = tool.name if tool else call.name
        detail = call.arguments.get("query") or call.arguments.get("path")
        if detail:
            label = f"{label}: {detail}"
        labels.append(label)
    shown = ", ".join(labels)
    suffix = f": <i>{html.escape(shown[:200])}</i>" if shown else "…"
    return f"🧠 <b>Second brain</b>{suffix}"


async def _resolve_tool_round(
    message: Message,
    messages: list[dict[str, Any]],
    result: ChatResult,
    calls: list[ToolCall],
    *,
    search_ready: bool,
    search_allowed: bool,
    mcp_allowed: bool,
) -> None:
    """Run the model's search/MCP calls and append the tool exchange.

    Adds the assistant turn that requested the calls, one ``tool`` response
    per call (the chat API rejects an assistant turn with unanswered
    tool_calls), and a short notice for the user. A call whose round budget is
    spent still gets an answer explaining that, so every ``tool_call`` is
    closed. ``messages`` is extended in place so the next round sees the
    results. The exchange lives **only** in this request: nothing here is
    written to the database — the history stores the user question and the
    final answer, never tool calls or tool output.
    """
    messages.append(build_assistant_tool_message(result.content, calls))

    search_calls = [
        call for call in calls
        if call.name == "search_web" and search_ready
    ]
    mcp_calls = [call for call in calls if mcp_manager.is_mcp_tool(call.name)]

    notices: list[str] = []
    queries = [
        str(call.arguments.get("query") or "").strip() for call in search_calls
    ]
    shown = ", ".join(query for query in queries if query)
    if shown:
        notices.append(
            f"🔎 <b>Searching the web:</b> <i>{html.escape(shown[:200])}</i>"
        )
    if mcp_calls:
        notices.append(_mcp_notice(mcp_calls))
    if notices:
        await safe_answer(message, "\n".join(notices))

    for call in calls:
        if call.name == "search_web" and search_ready:
            if not search_allowed:
                messages.append(
                    build_tool_response(
                        call.id,
                        "Лимит веб-поиска на этот вопрос исчерпан. Ответь из "
                        "уже собранной информации.",
                    )
                )
                continue
            query = str(call.arguments.get("query") or "").strip()
            max_results = call.arguments.get("max_results")
            if not isinstance(max_results, int) or isinstance(max_results, bool):
                max_results = None
            results = await search_web(query, max_results) if query else []
            messages.append(
                build_tool_response(call.id, format_search_results(query, results))
            )
            continue

        if not mcp_allowed:
            messages.append(
                build_tool_response(
                    call.id,
                    "Лимит обращений к MCP на этот вопрос исчерпан. Ответь из "
                    "уже собранной информации.",
                )
            )
            continue
        output = await mcp_manager.call_tool(call.name, call.arguments)
        messages.append(build_tool_response(call.id, output))


def _without_mcp_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the MCP (private notes) tools from the catalog.

    Used when the conversation contains untrusted content: a fetched page or
    document must not be able to steer the model into reading the admin's
    personal notes.
    """
    return [
        tool
        for tool in tools
        if not str((tool.get("function") or {}).get("name", "")).startswith("mcp__")
    ]


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

    tools: list[dict[str, Any]] = []
    if config.enable_image_generation:
        tools.append(IMAGE_TOOL)
    if config.enable_sticker_generation:
        tools.append(STICKER_TOOL)
    search_ready = search_configured()
    if search_ready:
        tools.append(SEARCH_TOOL)
        apply_search_hint(messages)
    # MCP tools (the Obsidian "second brain") are exposed to admins only by
    # default; a server that fails to start simply contributes no tools.
    mcp_ready = mcp_available(user.id)
    if mcp_ready:
        mcp_tools = await mcp_manager.openai_tools()
        if mcp_tools:
            tools.extend(mcp_tools)
        else:
            mcp_ready = False
    tools_arg: list[dict[str, Any]] | None
    printer: _StreamPrinter | None = None
    total_elapsed = 0.0
    search_rounds = 0
    mcp_rounds = 0
    result: ChatResult

    async with waiting(message) as notifier:
        while True:
            # Third-party content (documents, fetched pages, quoted/forwarded
            # messages, search results) must not drive the private-note tools.
            # It can be present from the start (and persists in the stored
            # history) or appear after a search round — re-check every round.
            if messages_have_untrusted_content(messages):
                apply_untrusted_hint(messages)
                if mcp_ready:
                    mcp_ready = False
                    remove_mcp_hint(messages)
                    logger.info(
                        "MCP tools withheld: the request contains untrusted content"
                    )
            if mcp_ready:
                apply_mcp_hint(messages)
            tools_arg = (tools if mcp_ready else _without_mcp_tools(tools)) or None
            if config.stream_responses:
                printer = _StreamPrinter(message, notifier)
                try:
                    result = await ai_client.stream_chat_completion(
                        messages,
                        model=model,
                        temperature=settings["temperature"],
                        max_tokens=settings["max_tokens"],
                        tools=tools_arg,
                        on_delta=printer.on_delta,
                    )
                except AIStreamError as exc:
                    logger.warning(
                        "Streaming failed (%s); retrying as a single request", exc
                    )
                    await printer.discard()
                    printer = None
                    result = await _single_shot(messages, model, settings, tools_arg)
            else:
                result = await _single_shot(messages, model, settings, tools_arg)

            total_elapsed += result.elapsed

            image_call = next(
                (call for call in result.tool_calls if call.name == "generate_image"),
                None,
            )
            sticker_call = next(
                (call for call in result.tool_calls if call.name == "generate_sticker"),
                None,
            )
            search_calls = (
                [call for call in result.tool_calls if call.name == "search_web"]
                if search_ready
                else []
            )
            mcp_calls = (
                [
                    call for call in result.tool_calls
                    if mcp_manager.is_mcp_tool(call.name)
                ]
                if mcp_ready
                else []
            )
            handled_calls = [*search_calls, *mcp_calls]
            can_search = (
                bool(search_calls)
                and search_rounds < config.max_search_rounds
            )
            can_mcp = bool(mcp_calls) and mcp_rounds < config.mcp_max_rounds
            # Search and MCP are multi-round tool loops: run the calls and ask
            # again — unless the same turn also asked for an image/sticker
            # (those win) or every requested tool type spent its budget.
            if (
                handled_calls
                and image_call is None
                and sticker_call is None
                and (can_search or can_mcp)
            ):
                if can_search:
                    search_rounds += 1
                if can_mcp:
                    mcp_rounds += 1
                if printer is not None:
                    await printer.discard()
                    printer = None
                await _resolve_tool_round(
                    message,
                    messages,
                    result,
                    handled_calls,
                    search_ready=search_ready,
                    search_allowed=can_search,
                    mcp_allowed=can_mcp,
                )
                continue
            if handled_calls and not result.content:
                # Budget spent and the model still only wants to use tools: one
                # final tools-less request so the user gets an answer from the
                # results already gathered.
                logger.info("Tool round limit reached; forcing a final answer")
                result = await _single_shot(messages, model, settings, None)
                total_elapsed += result.elapsed
            break

    # Send the textual part first, if any.
    if result.content:
        saved = await service.add_message(
            chat_id=message.chat.id,
            user_id=user.id,
            role="assistant",
            content=result.content,
            model=model,
        )
        footer = format_timing(
            total_elapsed,
            model=result.model or model,
            tokens=result.total_tokens,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        await deliver_answer(
            message,
            printer,
            result.content,
            footer,
            title_source=text,
            reply_markup=_retry_markup(message, saved.id if saved else None),
        )
    elif image_call is None and sticker_call is None and printer is not None:
        await printer.discard()

    if sticker_call is not None:
        prompt = str(sticker_call.arguments.get("prompt") or text).strip() or text
        if not await generate_and_send_sticker(
            message,
            prompt,
            service=service,
            user_id=user.id,
            model=model,
        ) and not result.content:
            await refund_access(service, user.id, access)
    elif image_call is not None:
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

    # A question sent as a reply to a photo analyzes (or regenerates from)
    # the quoted image instead of chatting about text alone.
    reply = message.reply_to_message
    if reply is not None and reply.photo:
        await _guarded(
            message,
            lambda m: _route_single_photo(m, photo_from=reply, question=text),
        )
        return

    with limits.slot(user.id) as granted:
        if not granted:
            await safe_answer(message, limits.BUSY_MESSAGE)
            return
        service = await _open_registered(message)
        if service is None:
            return

        access: AccessResult | None = None
        try:
            access = await _consume_access(message, service)
            if access is None:
                return
            # A message that is essentially bare links is expanded with the
            # fetched page content. Runs after the access check on purpose:
            # only authorized users may make the server issue HTTP requests.
            if config.enable_link_analysis:
                text = await _expand_links_with_content(message, text)
            # Replies to text/media messages and forwards carry their
            # provenance into the AI request so the model knows what the
            # user is referring to.
            text = _with_quoted_context(message, text)
            await message.bot.send_chat_action(
                chat_id=message.chat.id, action=ChatAction.TYPING
            )
            await _answer_with_ai(message, service, text, access)
        except Exception:
            # A crash after the quota was taken must not bill the user for a
            # request that never produced an answer; refund_access is a no-op
            # when nothing was consumed or the unit is already given back.
            if access is not None:
                await refund_access(service, user.id, access)
            raise
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

    access: AccessResult | None = None
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
            f" <i>({asr_elapsed:.1f} s)</i>"
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
    except Exception:
        if access is not None:
            await refund_access(service, user.id, access)
        raise
    finally:
        await service.session.close()


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    """Photos: route to album (media group), vision analysis, or generation."""
    # Media groups (albums) are collected and processed together.
    if message.media_group_id:
        await _handle_album_photo(message)
        return
    await _guarded(message, _route_single_photo)


# --- Document analysis --------------------------------------------------------


@router.message(F.document)
async def handle_document(message: Message) -> None:
    """Uploaded files: extract the text and let the AI analyze it."""
    await _guarded(message, _handle_document)


async def _handle_document(message: Message) -> None:
    """Download an uploaded document, extract its text and ask the AI.

    The extracted text (clipped to ``MAX_EXTRACTED_CHARS``) is stored in
    history together with the user's caption as the question, so follow-ups
    ("what does section 2 say?") keep the document in context.
    """
    user = message.from_user
    doc = message.document
    if user is None or doc is None:
        return
    if not config.enable_document_analysis:
        return
    if not _is_addressed(message):
        return

    file_type = detect_file_type(doc.file_name, doc.mime_type)
    if file_type is None:
        await safe_answer(
            message,
            MSG_UNSUPPORTED_DOCUMENT.format(
                limit=config.max_document_bytes // (1024 * 1024)
            ),
        )
        return
    if doc.file_size and doc.file_size > config.max_document_bytes:
        await safe_answer(
            message,
            f"⚠️ <b>The file is too large.</b> The limit is "
            f"{config.max_document_bytes // (1024 * 1024)} MB.",
        )
        return

    service = await _open_registered(message)
    if service is None:
        return
    access: AccessResult | None = None
    try:
        access = await _consume_access(message, service)
        if access is None:
            return
        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )
        try:
            file = await message.bot.get_file(doc.file_id)
            buffer = await message.bot.download_file(file.file_path)
            data = buffer.read()
        except Exception as exc:  # noqa: BLE001 - any download failure is final
            logger.warning("Document download failed: %s", exc)
            await refund_access(service, user.id, access)
            await safe_answer(
                message,
                "❌ <b>Could not download the file.</b> Please try again.",
            )
            return
        if len(data) > config.max_document_bytes:
            await refund_access(service, user.id, access)
            await safe_answer(
                message,
                f"⚠️ <b>The file is too large.</b> The limit is "
                f"{config.max_document_bytes // (1024 * 1024)} MB.",
            )
            return

        ocr_note = ""
        async with waiting(message):
            extracted = await extract_document_text(file_type, data)
            if not extracted and file_type == "pdf" and config.enable_pdf_ocr:
                await safe_answer(
                    message,
                    "🔍 <b>No text layer — this looks like a scan.</b> Reading up "
                    f"to {config.pdf_ocr_max_pages} page(s) with a vision model…",
                )
                ocr = await ocr_pdf(data)
                if ocr is not None:
                    extracted = clip_extracted(ocr.text)
                    ocr_note = (
                        f"[Scanned PDF read by OCR: pages 1-{ocr.pages_read} of "
                        f"{ocr.total_pages}"
                        + ("; the remaining pages were NOT read" if ocr.partial else "")
                        + ". OCR may contain recognition errors.]\n\n"
                    )
                    if ocr.partial:
                        await safe_answer(
                            message,
                            f"ℹ️ Read the first {ocr.pages_read} of {ocr.total_pages} "
                            "pages (OCR page limit).",
                        )
        if not extracted:
            await refund_access(service, user.id, access)
            await safe_answer(
                message,
                "❌ <b>Could not extract any text from this file.</b>\n\n"
                "Password-protected and empty documents are not supported, and "
                "scans need a vision model (and ENABLE_PDF_OCR). You can also "
                "send a scanned page as a photo.",
            )
            return

        caption = strip_bot_mention(message.caption or "")
        link_blocks: list[str] = []
        if config.enable_link_analysis and caption:
            # Links in the caption belong to the same request as the file:
            # read them too instead of silently dropping them.
            async def notify(note: str) -> None:
                await safe_answer(message, note)

            async with waiting(message):
                link_blocks, remainder, _ = await read_links(
                    caption, notify, require_short=False
                )
            if link_blocks:
                caption = remainder
        combined = _compose_document_context(
            doc.file_name or "document",
            ocr_note + extracted,
            caption,
            link_blocks,
        )
        await _answer_with_ai(message, service, combined, access)
    except Exception:
        if access is not None:
            await refund_access(service, user.id, access)
        raise
    finally:
        await service.session.close()


async def _route_single_photo(
    message: Message,
    *,
    photo_from: Message | None = None,
    question: str | None = None,
) -> None:
    """Decide whether a single photo needs analysis, generation or enhancement.

    ``photo_from`` points to the message carrying the image when it is not
    ``message`` itself (a question sent as a reply to a photo); ``question``
    is the user's text in that case — the triggering message has no caption.
    """
    if not _is_addressed(message):
        return
    if question is not None:
        caption = question
    else:
        caption = strip_bot_mention(message.caption or "").strip()

    # Detect intent via AI before consuming quota.
    service = await _open_registered(message)
    if service is None:
        return
    try:
        settings = await service.get_user_settings(message.from_user.id)  # type: ignore[union-attr]
        intent = "analyze"
        enhance_ops: list[str] = []
        if caption:
            # Intent classification is itself a PAID model call, so gate it
            # behind a non-consuming access check: otherwise anyone — even a
            # user with no access at all — could farm the API by sending
            # captioned photos in a loop. The real quota unit is still consumed
            # later, by the handler that produces the answer.
            gate = await check_access(service, message.from_user.id, consume=False)  # type: ignore[union-attr]
            if not gate.allowed:
                await safe_answer(
                    message, gate.message or get_random_themed_response()
                )
                return
            intent, enhance_ops = await detect_intent(caption, settings["model"])
            # The caption is user content: log the decision only, not the text.
            logger.info("Photo intent: %s %s", intent, " ".join(enhance_ops))
    finally:
        await service.session.close()

    if intent == "enhance":
        # Local processing: no vision model needed, no generation quota.
        await _handle_enhance_photo(
            message,
            photo_from=photo_from,
            question=question,
            operations=enhance_ops,
        )
    elif intent == "generate" and config.enable_image_generation:
        await _handle_vision_and_generate(
            message, photo_from=photo_from, question=question
        )
    else:
        await _handle_photo(message, photo_from=photo_from, question=question)


async def _handle_photo(
    message: Message,
    *,
    photo_from: Message | None = None,
    question: str | None = None,
) -> None:
    """Handle photo messages: analyze the image with a vision model."""
    user = message.from_user
    if user is None:
        return
    if not _is_addressed(message):
        return

    photo_msg = message if photo_from is None else photo_from
    if question is not None:
        caption = question
    else:
        caption = strip_bot_mention(message.caption or "") or DEFAULT_VISION_PROMPT
        # A photo sent as a reply to a text message: quote it for context.
        quoted = _describe_quoted(message.reply_to_message)
        if quoted:
            caption = f"{quoted}\n\n---\n\n{caption}"

    service = await _open_registered(message)
    if service is None:
        return

    access: AccessResult | None = None
    try:
        settings = await service.get_user_settings(user.id)

        # Verify the model supports images *before* consuming quota.
        if not is_vision_model(settings["model"]):
            await message.answer(MSG_NO_VISION.format(model=html.escape(settings["model"])))
            return

        access = await _consume_access(message, service)
        if access is None:
            return

        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )

        photo = photo_msg.photo[-1]
        if photo.file_size and photo.file_size > config.max_photo_bytes:
            await refund_access(service, user.id, access)
            await safe_answer(
                message,
                f"⚠️ <b>The image is too large</b> "
                f"(limit {config.max_photo_bytes // (1024 * 1024)} MB).",
            )
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

        saved = await service.add_message(
            chat_id=message.chat.id,
            user_id=user.id,
            role="assistant",
            content=response,
            model=settings["model"],
        )

        footer = format_timing(
            vision_result.elapsed,
            model=vision_result.model or settings["model"],
            tokens=vision_result.total_tokens,
            input_tokens=vision_result.input_tokens,
            output_tokens=vision_result.output_tokens,
        )
        # No "Ask another model" button: re-asking needs the image attached.
        # The 📄/📝 export buttons still apply to the answer text.
        export_markup = (
            get_answer_keyboard(export_id=saved.id if saved else None)
            if config.enable_export and _is_private(message)
            else None
        )
        await deliver_answer(
            message,
            printer,
            response,
            footer,
            title_source=caption,
            reply_markup=export_markup,
        )

        hint = guest_hint(access)
        if hint:
            await safe_answer(message, hint)
    except Exception:
        if access is not None:
            await refund_access(service, user.id, access)
        raise
    finally:
        await service.session.close()


# --- Vision -> generation pipeline ----------------------------------------


async def _handle_vision_and_generate(
    message: Message,
    *,
    photo_from: Message | None = None,
    question: str | None = None,
) -> None:
    """Analyze the attached photo with vision, then generate a new image.

    The vision model produces a detailed description of the reference image;
    that description and the user's prompt are combined for image generation.
    ``photo_from``/``question`` allow the reference to come from a replied-to
    photo while the prompt is the text of the triggering message.
    """
    user = message.from_user
    if user is None:
        return

    if question is not None:
        caption = question
    else:
        caption = strip_bot_mention(message.caption or "").strip()
    photo_msg = message if photo_from is None else photo_from
    service = await _open_registered(message)
    if service is None:
        return

    access: AccessResult | None = None
    try:
        access = await _consume_access(message, service)
        if access is None:
            return
        settings = await service.get_user_settings(user.id)

        if not is_vision_model(settings["model"]):
            await message.answer(
                "⚠️ <b>Image generation with a reference needs a vision model.</b>\n\n"
                f"Model <code>{html.escape(settings['model'])}</code> cannot "
                f"process images. Switch to a vision-capable model with /models.\n\n"
                "<b>Vision models</b> are marked 🖼️ in the list "
                "(for example <code>qwen3.8-max</code> or <code>qwen3.8-flash</code>)."
            )
            return

        photo = photo_msg.photo[-1]
        if photo.file_size and photo.file_size > config.max_photo_bytes:
            await refund_access(service, user.id, access)
            await safe_answer(
                message,
                f"⚠️ <b>The image is too large</b> "
                f"(limit {config.max_photo_bytes // (1024 * 1024)} MB).",
            )
            return
        file = await message.bot.get_file(photo.file_id)
        buffer = await message.bot.download_file(file.file_path)
        image_bytes = buffer.read()

        # Step 1: vision analysis
        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )
        await safe_answer(message, "🔍 <b>Analyzing the reference image…</b>")

        history = await service.get_history(message.chat.id, user_id=user.id)
        base_messages: list[dict[str, Any]] = [
            {"role": "system", "content": settings["system_prompt"]},
            *history,
        ]

        async with waiting(message):
            vision_result = await ai_client.chat_with_image_result(
                text=VISION_FOR_GEN_PROMPT,
                image_bytes=image_bytes,
                messages=base_messages,
                model=settings["model"],
                temperature=0.3,
                max_tokens=1024,
            )
        vision_description = (vision_result.content or "").strip()
        if not vision_description:
            await refund_access(service, user.id, access)
            await safe_answer(message, "❌ <b>Could not analyze the reference image.</b>\n\nPlease try again.")
            return

        # Step 2: generate
        gen_prompt = generation_prompt(caption, vision_description)

        await service.add_message(
            chat_id=message.chat.id,
            user_id=user.id,
            role="user",
            content=f"[Image->Generate] {caption}",
            image_url=photo.file_id,
            model=settings["model"],
        )

        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO
        )
        await safe_answer(
            message,
            "🎨 <b>Generating an image based on the reference…</b>\n\n"
            f"<blockquote expandable>{html.escape(gen_prompt[:500])}</blockquote>",
        )

        async with waiting(message):
            result = await ai_client.generate_image_with_references(
                prompt=gen_prompt,
                reference_images=[image_bytes],
            )
        gen_image, gen_url = result.data, result.url

        gen_caption = f"🎨 <i>{html.escape(caption[:900])}</i>"
        footer = format_timing(result.elapsed, model=config.image_model)
        if footer:
            gen_caption = f"{gen_caption}\n\n{html.escape(footer)}"

        if gen_image is not None:
            sent = await _safe_answer_photo(
                message,
                BufferedInputFile(gen_image, filename="generated.png"),
                gen_caption,
            )
        elif gen_url:
            sent = await _send_image_by_url(message, gen_url, gen_caption)
        else:
            sent = False
            await safe_answer(
                message,
                "❌ <b>Could not generate the image.</b>\n\n"
                "The prompt may have been rejected by moderation, or the image "
                "service / model is unavailable for your account.",
            )

        if sent:
            await service.add_message(
                chat_id=message.chat.id,
                user_id=user.id,
                role="assistant",
                content=f"[Generated from reference] {caption}",
                model=config.image_model,
            )
        else:
            await refund_access(service, user.id, access)

        hint = guest_hint(access)
        if hint:
            await safe_answer(message, hint)

    except Exception:
        if access is not None:
            await refund_access(service, user.id, access)
        raise
    finally:
        await service.session.close()


# --- Photo enhancement ----------------------------------------------------


async def _handle_enhance_photo(
    message: Message,
    *,
    photo_from: Message | None = None,
    question: str | None = None,
    operations: list[str] | None = None,
) -> None:
    """Improve the attached photo — on the server or with the neural network.

    In private chats the user is asked which path to take (the question
    itself consumes no quota; the answer arrives via the ``enh:`` callback).
    Groups — and private chats with generation switched off, where the
    neural option does not exist — get the fast, free server pipeline
    without prompting. No vision model is required for either path: the
    classifier already picked the operations, the server uses Pillow and
    the neural path talks to the image-synthesis API directly.
    """
    user = message.from_user
    if user is None:
        return

    if question is not None:
        caption = question
    else:
        caption = strip_bot_mention(message.caption or "").strip()
    photo_msg = message if photo_from is None else photo_from

    photo = photo_msg.photo[-1]
    if photo.file_size and photo.file_size > config.max_photo_bytes:
        await safe_answer(
            message,
            f"⚠️ <b>The image is too large</b> "
            f"(limit {config.max_photo_bytes // (1024 * 1024)} MB).",
        )
        return

    if _is_private(message) and config.enable_image_generation:
        # The neural option means the generation API: no question without it.
        _remember_pending_enhance(
            user.id, [photo.file_id], caption, operations or []
        )
        await safe_answer(
            message, MSG_ENHANCE_CHOICE, reply_markup=get_enhance_choice_keyboard()
        )
        return

    try:
        file = await message.bot.get_file(photo.file_id)
        buffer = await message.bot.download_file(file.file_path)
        image_bytes = buffer.read()
    except Exception:
        logger.warning("Could not download photo to enhance", exc_info=True)
        await safe_answer(
            message, "❌ <b>Could not download the photo.</b> Please try again."
        )
        return

    # Quota is consumed inside _run_enhancement (and refunded on failure).
    await _run_enhancement(
        message,
        user,
        [(photo.file_id, image_bytes)],
        caption,
        operations or [],
        via="server",
    )


async def _run_enhancement(
    target: Message,
    user: User,
    items: list[tuple[str, bytes]],
    caption: str,
    operations: list[str],
    *,
    via: str,
) -> None:
    """Enhance one or more photos and deliver the results.

    Shared by the group path (``via="server"``, straight away) and by the
    private clarify-callback (``via`` comes from the pressed button).
    ``target`` is the message to answer into — the photo itself or the bot's
    clarify question. Quota is consumed here: check_access counts the
    request, refund_access gives it back when nothing could be delivered.
    """
    service = await _open_registered_user(user)
    access: AccessResult | None = None
    try:
        access = await check_access(service, user.id)
        if not access.allowed:
            await target.answer(access.message or get_random_themed_response())
            return

        settings = await service.get_user_settings(user.id)
        await service.add_message(
            chat_id=target.chat.id,
            user_id=user.id,
            role="user",
            content=f"[Image→Enhance:{via} {len(items)}] {caption}",
            image_url=items[0][0],
            model=settings["model"],
        )

        await target.bot.send_chat_action(
            chat_id=target.chat.id, action=ChatAction.UPLOAD_PHOTO
        )

        done = 0
        for index, (file_id, image_bytes) in enumerate(items, 1):
            if via == "neural":
                async with waiting(target):
                    result = await ai_client.enhance_image_via_ai(
                        image_bytes, operations=operations, caption=caption
                    )
                data, url, elapsed = result.data, result.url, result.elapsed
                label = "🧠 <b>Enhanced by the neural network</b>"
                timing_model = config.image_model
            else:
                started = time.monotonic()
                data, applied = await enhance_image(image_bytes, operations)
                url = None
                elapsed = time.monotonic() - started
                label = "🔧 <b>Enhanced:</b> " + html.escape(", ".join(applied))
                timing_model = "Pillow"

            if data is None and url is None:
                logger.warning(
                    "Enhance (%s) failed for photo %s of %s", via, index, file_id
                )
                continue

            photo_caption = (
                f"({index}/{len(items)}) {label}" if len(items) > 1 else label
            )
            footer = format_timing(elapsed, model=timing_model)
            if footer:
                photo_caption = f"{photo_caption}\n\n{html.escape(footer)}"

            if data is not None:
                filename = "enhanced.png" if data[:4] == b"\x89PNG" else "enhanced.jpg"
                sent = await _safe_answer_photo(
                    target,
                    BufferedInputFile(data, filename=filename),
                    photo_caption,
                )
            else:
                sent = await _send_image_by_url(target, url or "", photo_caption)
            if sent:
                done += 1

        if done == 0:
            await refund_access(service, user.id, access)
            await safe_answer(
                target,
                "❌ <b>Could not enhance the image.</b>\n\n"
                "The file may be damaged or in an unsupported format, or the "
                "image service is temporarily unavailable.",
            )
            return

        await service.add_message(
            chat_id=target.chat.id,
            user_id=user.id,
            role="assistant",
            content=f"[Enhanced via {via}] {caption}",
            model=settings["model"],
        )

        hint = guest_hint(access)
        if hint:
            await safe_answer(target, hint)

    except Exception:
        if access is not None:
            await refund_access(service, user.id, access)
        raise
    finally:
        await service.session.close()


# --- Album (media group) handling ----------------------------------------


async def _handle_album_photo(message: Message) -> None:
    """Collect photos from a media group and process the album once complete."""
    group_id = str(message.media_group_id)
    key = _album_key(message.chat.id, group_id)

    photo = message.photo[-1]
    now = time.monotonic()
    sender = getattr(message, "from_user", None)
    sender_id = sender.id if sender is not None else 0

    # Reserve room BEFORE downloading: Telegram reports the size up front, so
    # an oversized photo — or an album whose budget is already spent — costs
    # neither bandwidth nor memory. On top of the per-album cap, the global
    # in-flight budget (albums, bytes, per-user albums) stops a flood of media
    # groups from exhausting RAM: buffers live for up to _ALBUM_CLEANUP_SECONDS
    # and are collected before any per-user request slot is taken.
    declared = photo.file_size or 0
    over_budget = False
    async with _album_lock:
        if key not in _album_images:
            same_user = sum(
                1
                for entry in _album_images.values()
                if entry.get("user_id") == sender_id
            )
            if (
                len(_album_images) >= _ALBUM_MAX_CONCURRENT
                or same_user >= _ALBUM_MAX_PER_USER
                or _album_bytes_in_flight() + declared > _ALBUM_MAX_TOTAL_BYTES
            ):
                over_budget = True
            else:
                _album_images[key] = {
                    "images": [],
                    "message": message,
                    "last": 0.0,
                    "user_id": sender_id,
                }
                _album_total_bytes[key] = 0
        if not over_budget:
            entry = _album_images.get(key)
            if entry is not None and not entry["message"].caption and message.caption:
                entry["message"] = message
            total_before = _album_total_bytes.get(key, 0)

    if over_budget:
        await safe_answer(message, _ALBUM_BUDGET_NOTICE)
        return

    over_single = bool(declared) and declared > config.max_photo_bytes
    over_album = total_before + declared > config.max_album_bytes

    if over_single:
        await safe_answer(
            message,
            f"⚠️ <b>Image too large.</b> Maximum size per image is "
            f"{config.max_photo_bytes // (1024 * 1024)} MB. "
            f"This image is {declared / (1024 * 1024):.1f} MB — skipped.",
        )
        await _touch_album(key, now)
    elif over_album:
        await safe_answer(
            message,
            f"⚠️ <b>Album size limit reached.</b> The maximum total is "
            f"{config.max_album_bytes // (1024 * 1024)} MB — this image was skipped.",
        )
        await _touch_album(key, now)
    else:
        try:
            file = await message.bot.get_file(photo.file_id)
            buffer = await message.bot.download_file(file.file_path)
            image_bytes = buffer.read()
        except Exception:
            logger.warning("Could not download album photo %s", photo.file_id, exc_info=True)
            await _touch_album(key, now)
            image_bytes = None

        if image_bytes is not None:
            img_size = len(image_bytes)
            if img_size > config.max_photo_bytes:
                await safe_answer(
                    message,
                    f"⚠️ <b>Image too large.</b> Maximum size per image is "
                    f"{config.max_photo_bytes // (1024 * 1024)} MB. "
                    f"This image is {img_size / (1024 * 1024):.1f} MB — skipped.",
                )
                await _touch_album(key, now)
            else:
                # The declared size can be missing or understated, so the real
                # byte count is checked again before the buffer is kept.
                appended = False
                gone = False
                async with _album_lock:
                    entry = _album_images.get(key)
                    if entry is None:
                        gone = True
                    else:
                        others = sum(
                            value
                            for other, value in _album_total_bytes.items()
                            if other != key
                        )
                        if (
                            others + _album_total_bytes.get(key, 0) + img_size
                            <= _ALBUM_MAX_TOTAL_BYTES
                        ):
                            entry["images"].append((photo.file_id, image_bytes))
                            entry["last"] = now
                            _album_total_bytes[key] = (
                                _album_total_bytes.get(key, 0) + img_size
                            )
                            appended = True
                if gone:
                    return
                if not appended:
                    await safe_answer(message, _ALBUM_BUDGET_NOTICE)
                    await _touch_album(key, now)

    await asyncio.sleep(_ALBUM_COLLECT_SECONDS)
    await asyncio.sleep(0)  # yield to let other album handlers update the entry

    async with _album_lock:
        entry = _album_images.get(key)
        if entry is None or entry["last"] > now:
            return
        images: list[tuple[str, bytes]] = entry.pop("images", [])
        first_message: Message = entry.pop("message", message)
        total_size = _album_total_bytes.pop(key, 0)
        _album_images.pop(key, None)

    if not images:
        return

    if total_size > config.max_album_bytes:
        await safe_answer(
            first_message,
            f"⚠️ <b>Album too large.</b> The total album size is "
            f"{total_size / (1024 * 1024):.1f} MB; the maximum is "
            f"{config.max_album_bytes // (1024 * 1024)} MB. "
            f"Please send fewer or smaller images.",
        )
        return

    await _guarded(first_message, lambda m: _process_album(m, images))


async def _process_album(
    message: Message, images: list[tuple[str, bytes]]
) -> None:
    """Process a collected album: vision analysis, generation or enhancement."""
    user = message.from_user
    if user is None:
        return
    if not _is_addressed(message):
        return

    caption = strip_bot_mention(message.caption or "")

    service = await _open_registered(message)
    if service is None:
        return

    access: AccessResult | None = None
    try:
        access = await _consume_access(message, service)
        if access is None:
            return
        settings = await service.get_user_settings(user.id)

        # Detect intent via AI before processing (safe: defaults to 'analyze').
        intent, enhance_ops = "analyze", []
        if caption:
            intent, enhance_ops = await detect_intent(caption, settings["model"])
            # Log the decision only: the caption is user content.
            logger.info("Album intent: %s %s", intent, " ".join(enhance_ops))

        # Enhancement runs locally: it needs neither a vision model nor
        # the generation service, so only the other paths check for vision.
        wants_gen = intent == "generate" and config.enable_image_generation
        if intent != "enhance" and not is_vision_model(settings["model"]):
            await message.answer(MSG_NO_VISION.format(model=html.escape(settings["model"])))
            return

        valid = [(fid, img) for fid, img in images if len(img) <= config.max_photo_bytes]
        valid_images = [img for _, img in valid]
        if len(valid_images) < len(images):
            await safe_answer(
                message,
                f"⚠️ <b>{len(images) - len(valid_images)} image(s) too large</b> "
                f"(limit {config.max_photo_bytes // (1024 * 1024)} MB). Skipped.",
            )
        if not valid_images:
            await refund_access(service, user.id, access)
            return

        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )

        if wants_gen:
            # Album -> generation
            await safe_answer(
                message,
                f"🔍🎨 <b>Analyzing {len(valid_images)} reference image(s) and generating…</b>",
            )

            history = await service.get_history(message.chat.id, user_id=user.id)
            base_messages: list[dict[str, Any]] = [
                {"role": "system", "content": settings["system_prompt"]},
                *history,
            ]

            descriptions: list[str] = []
            for i, img in enumerate(valid_images, 1):
                async with waiting(message):
                    vis = await ai_client.chat_with_image_result(
                        text=VISION_FOR_GEN_PROMPT,
                        image_bytes=img,
                        messages=base_messages,
                        model=settings["model"],
                        temperature=0.3,
                        max_tokens=512,
                    )
                desc = (vis.content or "").strip()
                if desc:
                    descriptions.append(f"Image {i}: {desc}")

            if not descriptions:
                await refund_access(service, user.id, access)
                await safe_answer(message, "❌ Could not analyze the reference images.")
                return

            combined_desc = "\n\n".join(descriptions)
            gen_prompt = generation_prompt(
                caption or "Generate an image inspired by these references",
                combined_desc,
            )

            await service.add_message(
                chat_id=message.chat.id,
                user_id=user.id,
                role="user",
                content=f"[Album->Generate {len(valid_images)} images] {caption}",
                model=settings["model"],
            )

            await message.bot.send_chat_action(
                chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO
            )
            async with waiting(message):
                result = await ai_client.generate_image_with_references(
                    prompt=gen_prompt,
                    reference_images=valid_images,
                )
            gen_image, gen_url = result.data, result.url

            gen_caption = f"🎨 <i>{html.escape((caption or 'Generated image')[:900])}</i>"
            footer = format_timing(result.elapsed, model=config.image_model)
            if footer:
                gen_caption = f"{gen_caption}\n\n{html.escape(footer)}"

            sent = False
            if gen_image is not None:
                sent = await _safe_answer_photo(
                    message,
                    BufferedInputFile(gen_image, filename="generated.png"),
                    gen_caption,
                )
            elif gen_url:
                sent = await _send_image_by_url(message, gen_url, gen_caption)

            if sent:
                await service.add_message(
                    chat_id=message.chat.id,
                    user_id=user.id,
                    role="assistant",
                    content=f"[Generated from {len(valid_images)} references] {caption}",
                    model=config.image_model,
                )
            else:
                await refund_access(service, user.id, access)
                await safe_answer(message, "❌ <b>Could not generate the image.</b>")
        elif intent == "enhance":
            # Album -> enhancement. In private chats ask first: server or
            # neural? The consumed quota is refunded and re-charged when the
            # user presses a button (the photos' file ids stay fetchable).
            if _is_private(message) and config.enable_image_generation:
                _remember_pending_enhance(
                    user.id,
                    [fid for fid, _ in valid],
                    caption,
                    enhance_ops,
                )
                await refund_access(service, user.id, access)
                await safe_answer(
                    message,
                    MSG_ENHANCE_CHOICE,
                    reply_markup=get_enhance_choice_keyboard(),
                )
                return

            await safe_answer(
                message,
                f"🖥️ <b>Enhancing {len(valid_images)} image(s) on the server…</b>",
            )
            await service.add_message(
                chat_id=message.chat.id,
                user_id=user.id,
                role="user",
                content=f"[Album→Enhance {len(valid_images)} images] {caption}",
                model=settings["model"],
            )

            done = 0
            for index, img in enumerate(valid_images, 1):
                enhanced, applied = await enhance_image(img, enhance_ops)
                if not enhanced:
                    await safe_answer(
                        message, f"⚠️ <b>Image {index} could not be processed.</b>"
                    )
                    continue
                photo_caption = (
                    f"🔧 <b>Enhanced {index}/{len(valid_images)}:</b> "
                    + html.escape(", ".join(applied))
                )
                filename = (
                    "enhanced.png" if enhanced[:4] == b"\x89PNG" else "enhanced.jpg"
                )
                if not await _safe_answer_photo(
                    message, BufferedInputFile(enhanced, filename=filename), photo_caption
                ):
                    break
                done += 1

            if done == 0:
                await refund_access(service, user.id, access)
                await safe_answer(message, "❌ <b>Could not process the images.</b>")
                return
            await service.add_message(
                chat_id=message.chat.id,
                user_id=user.id,
                role="assistant",
                content=f"[Enhanced {done} images] {caption}",
                model=settings["model"],
            )
        else:
            # Album -> vision analysis
            await safe_answer(
                message,
                f"🔍 <b>Analyzing {len(valid_images)} image(s)…</b>",
            )

            history = await service.get_history(message.chat.id, user_id=user.id)
            base_messages = [
                {"role": "system", "content": settings["system_prompt"]},
                *history,
            ]

            full_messages = [dict(entry) for entry in base_messages]
            content_parts: list[dict[str, Any]] = [
                {"type": "text", "text": caption or f"Describe these {len(valid_images)} images."}
            ]
            for img in valid_images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(img)},
                })
            full_messages.append({"role": "user", "content": content_parts})

            await service.add_message(
                chat_id=message.chat.id,
                user_id=user.id,
                role="user",
                content=f"[Album {len(valid_images)} images] {caption}",
                model=settings["model"],
            )

            async with waiting(message):
                album_result = await ai_client.chat_completion_result(
                    full_messages,
                    model=settings["model"],
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                )
            response = album_result.content or ""

            if not response:
                await refund_access(service, user.id, access)
                await safe_answer(message, "❌ <b>Error analyzing the images.</b>")
                return

            await service.add_message(
                chat_id=message.chat.id,
                user_id=user.id,
                role="assistant",
                content=response,
                model=settings["model"],
            )

            footer = format_timing(
                album_result.elapsed,
                model=album_result.model or settings["model"],
                tokens=album_result.total_tokens,
                input_tokens=album_result.input_tokens,
                output_tokens=album_result.output_tokens,
            )
            await deliver_answer(message, None, response, footer, title_source=caption)

        hint = guest_hint(access)
        if hint:
            await safe_answer(message, hint)

    except Exception:
        if access is not None:
            await refund_access(service, user.id, access)
        raise
    finally:
        await service.session.close()


# --- Enhance choice callbacks ----------------------------------------------


@router.callback_query(F.data.startswith("enh:"))
async def cb_enhance_choice(callback: CallbackQuery) -> None:
    """Finish a private enhance request after the server/neural choice.

    The photo is re-downloaded from the stored ``file_id`` (still valid
    after the question), quota is consumed now — not when the question was
    asked — and the chosen pipeline runs via the shared enhancement flow.
    """
    if callback.data is None or callback.from_user is None or callback.message is None:
        await safe_answer_callback(callback)
        return
    via = "neural" if callback.data == "enh:ai" else "server"

    entry = _pop_pending_enhance(callback.from_user.id)
    if entry is None:
        await safe_answer_callback(
            callback,
            "This enhance request has expired — please send the photo again.",
            show_alert=True,
        )
        return

    await safe_answer_callback(
        callback,
        "🧠 Enhancing with the neural network…"
        if via == "neural"
        else "🖥️ Enhancing on the server…",
    )
    origin = callback.message
    await safe_edit_text(
        origin,
        "🧠 <b>Enhancing with the neural network…</b>"
        if via == "neural"
        else "🖥️ <b>Enhancing on the server…</b>",
    )

    with limits.slot(callback.from_user.id) as granted:
        if not granted:
            await safe_answer(origin, limits.BUSY_MESSAGE)
            return

        items: list[tuple[str, bytes]] = []
        for file_id in entry["file_ids"]:
            try:
                file = await origin.bot.get_file(file_id)
                buffer = await origin.bot.download_file(file.file_path)
                items.append((str(file_id), buffer.read()))
            except Exception:
                logger.warning(
                    "Could not re-download pending photo %s", file_id, exc_info=True
                )
        if not items:
            await safe_answer(
                origin,
                "❌ <b>Could not fetch the photo anymore.</b> Please resend it.",
            )
            return

        await _run_enhancement(
            origin,
            callback.from_user,
            items,
            str(entry["caption"]),
            list(entry["operations"]),
            via=via,
        )


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
        access: AccessResult | None = None
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
        except Exception:
            if access is not None:
                await refund_access(service, callback.from_user.id, access)
            raise
        finally:
            await service.session.close()


@router.edited_message(F.text, ~F.text.startswith("/"))
async def handle_edited_text(message: Message) -> None:
    """Handle a user editing an already-sent question.

    The answer above was produced from the previous version of the message, so
    silently accepting the edit would desynchronise the visible conversation
    from the stored context. Editing does **not** consume quota: the user is
    told about it and the remembered question is refreshed so the
    "ask another model" button re-asks the edited text.

    Notices are rate-limited per chat: repeated edits must not turn into a
    flood of bot messages.
    """
    user = message.from_user
    if user is None or not message.text:
        return
    if not _is_addressed(message):
        return

    text = strip_bot_mention(message.text)
    if not text:
        return

    private = _is_private(message)
    if private:
        _remember_question(message.chat.id, _with_quoted_context(message, text))

    if not _edit_notice_allowed(message.chat.id):
        return

    note = (
        "✏️ <b>Message edited.</b>\n\n"
        "The answer above was based on the previous version. "
        "Send the text again to get an answer to the new version."
    )
    if private and config.enable_ask_other_model:
        note += " Or use the button below to re-ask it."
    await safe_answer(message, note, reply_markup=_retry_markup(message))


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
