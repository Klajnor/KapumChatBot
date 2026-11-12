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
    _describe_reply_context,
    _without_mcp_tools,
    _without_search_tool,
    _without_untrusted_side_effects,
    read_links,
)
from bot.services import limits
from bot.services.access import AccessResult, check_access, refund_access
from bot.services.ai_client import (
    IMAGE_TOOL,
    ChatResult,
    ai_client,
    is_vision_model,
)
from bot.services.documents import (
    DocumentParseTimeout,
    detect_file_type,
    extract_document_text,
)
from bot.services.intent import (
    VISION_FOR_GEN_PROMPT,
    detect_intent,
    generation_prompt,
)
from bot.services.markdown import render, split_chunks
from bot.services.mcp import (
    apply_mcp_hint,
    isolate_mcp_history,
    mcp_available,
    mcp_manager,
    remove_mcp_hint,
)
from bot.services.operations import stage, tracked
from bot.services.pdf_limits import PdfResourceLimit
from bot.services.request_scope import WEB_MARKER_RE, route
from bot.services.response_metadata import AnswerMetadata, metadata_for_result
from bot.services.retrieval_answer import (
    ensure_note_lookup,
    final_answer_messages,
    is_retrieval_promise,
    retrieval_fallback,
)
from bot.services.retrieval_plan import apply_combined_hint, isolate_combined_history
from bot.services.rich import RICH_LIMIT, adapt_markdown
from bot.services.science import normalize_formulas
from bot.services.search import (
    SEARCH_TOOL,
    apply_search_hint,
    build_assistant_tool_message,
    build_tool_response,
    format_search_results,
    remove_search_hint,
    search_configured,
    search_web,
)
from bot.services.telegram_quotes import DEFAULT_QUOTE_QUESTION, has_quote
from bot.services.telegram_utils import (
    PARSE_MODES,
    format_timing,
    is_forwarded,
    safe_answer_guest,
    strip_bot_mention,
    with_timing,
)
from bot.services.telegraph import needs_telegraph, notice_text, publish_answer
from bot.services.themed_responses import get_random_themed_response
from bot.services.tool_budget import CALL_LIMIT_NOTICE, ToolBudget
from bot.services.untrusted import (
    apply_untrusted_hint,
    messages_have_untrusted_content,
    remove_untrusted_hint,
    wrap_untrusted,
)
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
    question: str, answer: str, footer: str, *, allow: bool = True
) -> GuestReply | None:
    """A guest query allows one short reply: links beat broken tables/formulas.

    Returns ``None`` when Telegraph is not needed or unavailable, so the
    caller sends the answer as a normal (truncated) text reply.
    ``allow=False`` keeps answers based on documents/pages/forwards inside
    Telegram instead of publishing them publicly.
    """
    if not allow:
        return None
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
    if message.photo:
        source = message
    elif has_quote(message):
        # Выделена подпись: не скачиваем всё изображение из replied-сообщения.
        source = None
    else:
        source = message.reply_to_message
    if source is None or not source.photo:
        return None
    return await _download(message, source.photo[-1].file_id)


def _voice_of(message: Message):
    """Return the voice note attached to the message or to the replied one."""
    if message.voice is not None:
        return message.voice
    if has_quote(message):
        return None
    reply = message.reply_to_message
    return reply.voice if reply is not None else None


def _document_of(message: Message):
    """Return the document attached to the message or to the replied one."""
    if message.document is not None:
        return message.document
    if has_quote(message):
        return None
    reply = message.reply_to_message
    return reply.document if reply is not None else None


def _attachment_foreign(
    message: Message, source: Message | None, caller_id: int
) -> bool:
    """True when an attachment was written/forwarded by somebody else.

    Guest replies may reference a photo or voice note from the replied-to
    message. That content is third-party even though the summon itself was
    written by the caller. Unknown authors are treated as foreign (fail closed).
    """
    if source is None:
        return False
    if is_forwarded(source):
        return True
    if source is message:
        return False
    author = getattr(source, "from_user", None)
    return author is None or getattr(author, "id", None) != caller_id


def _extract_text(message: Message) -> str:
    """Summoning text with the ``@bot`` mention removed."""
    return strip_bot_mention(message.text or message.caption or "").strip()


# --- Reply pipeline --------------------------------------------------------


@stage("Preparing guest answer")
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
    # Forwarded summon text is somebody else's words: it is data, not a request.
    text_tainted = is_forwarded(message)
    text_external = text_tainted
    external = text_tainted

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
    crawl_notes: list[str] = []
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

    if text_tainted and text:
        text = f"[Forwarded message]\n{wrap_untrusted('forward', text)}"

    # Document: extract its text (after the quota has been taken) and fold it,
    # together with the pages linked in the text, into the request.
    if document is not None:
        # Uploaded/replied documents and their OCR/link text are third-party data.
        external = True
        text_external = True
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
            try:
                extracted = await extract_document_text(file_type, data)
            except PdfResourceLimit:
                await refund_access(service, caller_id, access)
                return GuestReply(
                    text="⚠️ The PDF exceeds safe processing limits. "
                    "Try a smaller file. Your question was not counted.",
                    mode="plain",
                )
            except DocumentParseTimeout:
                await refund_access(service, caller_id, access)
                return GuestReply(
                    text=(
                        "⏱️ The file took too long to process "
                        f"(limit {config.document_parse_timeout:g} s). "
                        "Try a smaller file."
                    ),
                    mode="plain",
                )
        if not extracted:
            await refund_access(service, caller_id, access)
            return GuestReply(
                text="❌ Could not extract any text from this file.", mode="plain"
            )
        link_blocks: list[str] = []
        if config.enable_link_analysis and text:
            link_blocks, remainder, _ = await read_links(
                text, require_short=False, notes=crawl_notes
            )
            if link_blocks:
                external = True
                text = remainder
        text = _compose_document_context(
            document.file_name or "document", extracted, text, link_blocks
        )
    elif config.enable_link_analysis and text:
        # A summons that is essentially bare links reads those pages.
        link_blocks, remainder, videos = await read_links(text, notes=crawl_notes)
        if link_blocks:
            external = True
            text_external = True
            request = remainder or (
                DEFAULT_VIDEO_PROMPT if videos == len(link_blocks) else DEFAULT_LINK_PROMPT
            )
            text = "\n\n".join(link_blocks) + f"\n\n---\n\nUser request: {request}"

    # Voice message: recognize it now (after the quota has been taken).
    voice_foreign = voice_ok and _attachment_foreign(
        message,
        message
        if getattr(message, "voice", None) is not None
        else getattr(message, "reply_to_message", None),
        caller_id,
    )
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
        if voice_foreign:
            # A replied-to/forwarded voice note is data, exactly like a forward.
            external = True
            text_external = True
            transcript = (
                "[Voice message from another participant]\n"
                f"{wrap_untrusted('voice', transcript)}"
            )
        text = f"{text}\n{transcript}".strip() if text else transcript.strip()

    # Цитату добавляем ПОСЛЕ чтения ссылок/ASR: URL и инструкции внутри неё
    # являются данными, не запросом на загрузку или платное действие. Один
    # разделитель сохраняет структуру document/link-контекста без второго
    # «User request» внутри него.
    quoted = _describe_reply_context(message, text_only=True)
    if quoted:
        text = f"{quoted}\n\n---\n\n{text or DEFAULT_QUOTE_QUESTION}"
        external = True
        text_external = True

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

    # A replied-to/forwarded photo is third-party too. Its content cannot be
    # wrapped (it is an image), so the history record gets a marker instead and
    # the next question keeps the taint gate active.
    photo_source = message if getattr(message, "photo", None) else None
    replied = getattr(message, "reply_to_message", None)
    if photo_source is None and not has_quote(message) and getattr(replied, "photo", None):
        photo_source = replied
    photo_foreign = _attachment_foreign(message, photo_source, caller_id)
    external = external or photo_foreign
    # Происхождение текущего запроса сохраняем отдельно от taint истории:
    # изоляция разрешена только новому собственному тексту пользователя.
    current_external = external
    # The stored history may already contain an untrusted document/page from
    # an earlier summon. Re-check it every time: a clean current message must
    # not re-enable the paid image tool for text that is still in context.
    if messages_have_untrusted_content(base_messages):
        external = True
    if external:
        apply_untrusted_hint(base_messages)

    history_text = text
    if photo_foreign:
        history_text = (
            f"{history_text}\n{wrap_untrusted('photo', '[image contents]')}".strip()
        )

    await service.add_message(
        chat_id=chat_id,
        user_id=caller_id,
        role="user",
        content=history_text,
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
        if (
            text
            and not text_external
            and config.enable_image_generation
            and await service.consume_photo_intent_budget(caller_id)
        ):
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
            public=True,
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
            metadata=metadata_for_result(vision),
        )
        telegraph_reply = await _maybe_telegraph_reply(
            text, response, footer, allow=not external
        )
        if telegraph_reply is not None:
            return telegraph_reply
        return GuestReply(text=with_timing(response + _guest_hint(access), footer))

    # Text request -> the model may read the "second brain" (read-only MCP
    # tools) and answer with a generated image. With external content in the
    # request both are withheld: third-party text must not steer a paid
    # generation call or reach the private notes.
    messages = [*base_messages, {"role": "user", "content": text}]
    mcp_ready = mcp_available(caller_id)
    # Каталог проверяем до изоляции: сбой сервера не должен вырезать историю.
    mcp_tools: list[dict[str, Any]] = []
    if mcp_ready and not current_external:
        mcp_tools = await mcp_manager.openai_tools()
        mcp_ready = bool(mcp_tools)
    combined = mcp_ready and not current_external and isolate_combined_history(messages)
    mcp_only = combined or (mcp_ready and not current_external and isolate_mcp_history(messages))
    if mcp_only and not messages_have_untrusted_content(messages):
        # Подсказка вешалась на base_messages до изоляции (она нужна и для
        # vision-пути выше). После вырезания истории тегов больше нет.
        remove_untrusted_hint(messages)
    tools: list[dict[str, Any]] = []
    # Prefer the provider's built-in search for models that support it; the
    # local search_web tool is the fallback for the others. Search is available
    # in every guest request; external content still withholds MCP, image
    # generation and file sending.
    builtin_search = config.builtin_web_search(settings["model"])
    explicit_web = bool(WEB_MARKER_RE.search(text))
    search_ready = search_configured() and (
        not builtin_search or explicit_web
    )
    if search_ready:
        tools.append(SEARCH_TOOL)
        apply_search_hint(messages)
    if config.enable_image_generation and not external and not mcp_only:
        tools.append(IMAGE_TOOL)
    mcp_ready = mcp_ready and (not external or mcp_only)
    if mcp_ready:
        tools.extend(mcp_tools)
    if combined:
        apply_combined_hint(
            messages,
            web_available=search_ready or builtin_search,
            notes_available=mcp_ready,
        )

    budget = ToolBudget()
    search_rounds = 0
    mcp_rounds = 0
    result = ChatResult()
    metadata = AnswerMetadata()
    while True:
        # Note text returned by MCP is framed as untrusted; once it is in the
        # context, neither the notes tools nor the paid image tool may stay
        # offered (a note must not steer them). Only the final answer is
        # delivered: a guest query cannot send progress notices.
        untrusted = messages_have_untrusted_content(messages)
        if untrusted:
            apply_untrusted_hint(messages)
            if mcp_ready:
                mcp_ready = False
                remove_mcp_hint(messages)
        if mcp_ready:
            apply_mcp_hint(messages)
        offered_tools = tools
        if untrusted:
            offered_tools = _without_untrusted_side_effects(offered_tools)
        if not mcp_ready:
            offered_tools = _without_mcp_tools(offered_tools)

        if offered_tools:
            result = await ai_client.chat_with_tools(
                messages=messages,
                tools=offered_tools,
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                public=True,
            ) or ChatResult()
        else:
            result = await ai_client.chat_completion_result(
                messages=messages,
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                public=True,
            ) or ChatResult()

        metadata.add(result)
        # The model's turn is untrusted input: only tools it was actually
        # offered may run.
        offered_names = {tool["function"]["name"] for tool in offered_tools}
        result.tool_calls = [
            call for call in result.tool_calls if call.name in offered_names
        ]
        if mcp_rounds == 0:
            ensure_note_lookup(result, offered_tools, text)

        image_call = next(
            (call for call in result.tool_calls if call.name == "generate_image"),
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
        can_search = (
            bool(search_calls)
            and search_rounds < config.max_search_rounds
            and budget.search_left
        )
        can_mcp = (
            bool(mcp_calls)
            and mcp_rounds < config.mcp_max_rounds
            and budget.mcp_left
        )
        if (search_calls or mcp_calls) and image_call is None and (can_search or can_mcp):
            messages.append(
                build_assistant_tool_message(
                    result.content, [*search_calls, *mcp_calls]
                )
            )
            if search_calls:
                search_rounds += 1
            if mcp_calls:
                mcp_rounds += 1
            for call in search_calls:
                if not budget.take_search():
                    messages.append(build_tool_response(call.id, CALL_LIMIT_NOTICE))
                    continue
                query = str(call.arguments.get("query") or "").strip()
                max_results = call.arguments.get("max_results")
                if not isinstance(max_results, int) or isinstance(max_results, bool):
                    max_results = None
                if query:
                    metadata.api_search_used = True
                found = await search_web(query, max_results) if query else []
                failed = bool(getattr(found, "error", "") or "")
                messages.append(
                    build_tool_response(
                        call.id,
                        budget.clip(
                            format_search_results(query, found, failed=failed)
                        ),
                    )
                )
                if failed:
                    # Do not offer a broken provider again on the next round.
                    search_ready = False
                    tools = _without_search_tool(tools)
                    remove_search_hint(messages)
                    logger.warning("Guest web search failed; retrying without it")
            for call in mcp_calls:
                if not budget.take_mcp():
                    messages.append(build_tool_response(call.id, CALL_LIMIT_NOTICE))
                    continue
                output = await mcp_manager.call_tool(
                    call.name, call.arguments, max_chars=budget.context_chars, read_budget=budget,
                )
                messages.append(build_tool_response(call.id, budget.clip(output)))
            continue
        if (search_calls or mcp_calls) and not result.content and image_call is None:
            # Round/budget limit reached and the model only asked for tools:
            # one final tools-less request so the guest still gets an answer.
            logger.info("Guest tool round limit reached; forcing a final answer")
            result = await ai_client.chat_completion_result(
                messages=final_answer_messages(messages),
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                public=True,
            ) or ChatResult()
            metadata.add(result)
            result.tool_calls = []
            image_call = None
        if not result.content and not result.tool_calls and image_call is None:
            # No answer and no usable tool call (for example, a tool was
            # filtered out or the search provider failed): ask once more
            # without tools instead of returning an error.
            logger.info("Guest model returned no answer; forcing a final request")
            result = await ai_client.chat_completion_result(
                messages=final_answer_messages(messages),
                model=settings["model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                public=True,
            ) or ChatResult()
            metadata.add(result)
            result.tool_calls = []
        if is_retrieval_promise(result.content) and (
            search_rounds or mcp_rounds or route(text) in {"notes", "combined"}
        ):
            previous_elapsed = result.elapsed
            result = await ai_client.chat_completion_result(
                messages=final_answer_messages(messages),
                model=settings["model"], temperature=settings["temperature"],
                max_tokens=settings["max_tokens"], public=True,
            ) or ChatResult()
            metadata.add(result)
            result.elapsed += previous_elapsed
            result.tool_calls = []
            if not result.content or is_retrieval_promise(result.content):
                result.content = retrieval_fallback(messages)
        break
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
        metadata=metadata.summary(),
    )
    telegraph_reply = await _maybe_telegraph_reply(
        text, result.content, footer,
        allow=not external and not messages_have_untrusted_content(messages),
    )
    if telegraph_reply is not None:
        return telegraph_reply
    tail = "".join(f"\n\n{note}" for note in crawl_notes)
    return GuestReply(
        text=with_timing(result.content + tail + _guest_hint(access), footer)
    )


# --- Handler ---------------------------------------------------------------


@router.guest_message()
@tracked("Guest request")
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
