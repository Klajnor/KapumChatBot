from __future__ import annotations

import datetime
import html
import logging

from aiogram import Router, F
from aiogram.types import Message
from bot.config import config
from bot.database.session import async_session_factory
from bot.services.openrouter import openrouter_client
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

router = Router()

DAILY_LIMIT = 10  # Max questions per day for non-admin allowed users


def _is_admin(user_id: int) -> bool:
    """Check if a user is an admin by their Telegram ID."""
    return user_id in config.admin_ids


async def _can_use_bot(telegram_id: int) -> tuple[bool, str | None]:
    """
    Check if a user is allowed to use the bot, and increment daily usage for non-admins.
    Returns (is_allowed, error_message_or_None).
    """
    # Admins have unlimited access
    if _is_admin(telegram_id):
        return True, None

    async with async_session_factory() as session:
        service = UserService(session)
        # Check if user is in allowed list
        if not await service.is_allowed_user(telegram_id):
            return False, None  # Will get themed response

        # Check and increment daily limit
        today = datetime.date.today()
        from bot.database.repository import UserRepository
        repo = UserRepository(session)
        usage = await repo.get_daily_usage(telegram_id, today)
        used = usage.query_count if usage else 0

        if used >= DAILY_LIMIT:
            return False, (
                f"⚠️ <b>Daily limit reached.</b>\n\n"
                f"You've used all {DAILY_LIMIT} questions for today. "
                f"Come back tomorrow!"
            )

        # Increment usage for this question (atomic with the check)
        await repo.increment_daily_usage(telegram_id, today)
        await session.commit()
        return True, None


@router.message(F.text)
async def handle_text(message: Message) -> None:
    """Handle text messages — chat with AI via OpenRouter (admins + allowed users)."""
    user = message.from_user
    if user is None or message.text is None:
        return

    # Check access
    can_use, error_msg = await _can_use_bot(user.id)
    if not can_use:
        if error_msg:
            await message.answer(error_msg)
        else:
            await message.answer(get_random_themed_response())
        return

    user_text = message.text

    # Send typing indicator
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    async with async_session_factory() as session:
        service = UserService(session)

        # Get user settings and history
        settings = await service.get_user_settings(user.id)
        history = await service.get_history(user.id)

        # Build messages array for OpenRouter
        messages = [
            {"role": "system", "content": settings["system_prompt"]}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        # Save user message
        await service.add_message(
            telegram_id=user.id,
            role="user",
            content=user_text,
            model=settings["model"],
        )

        # Get AI response
        response = await openrouter_client.chat_completion(
            messages=messages,
            model=settings["model"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
        )

        if response is None:
            await message.answer(
                "❌ <b>Error communicating with AI.</b>\n\n"
                "Please try again later. If the problem persists, "
                "check your OpenRouter API key and balance."
            )
            return

        # Save assistant message
        await service.add_message(
            telegram_id=user.id,
            role="assistant",
            content=response,
            model=settings["model"],
        )

    # Send response (split if too long for Telegram)
    await send_long_message(message, response)


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    """Handle photo messages — analyze image with vision model (admins + allowed users)."""
    user = message.from_user
    if user is None:
        return

    # Check access
    can_use, error_msg = await _can_use_bot(user.id)
    if not can_use:
        if error_msg:
            await message.answer(error_msg)
        else:
            caption = message.caption or "📷 <i>(image omitted)</i>"
            await message.answer(get_random_themed_response())
        return

    # Get the largest photo
    photo = message.photo[-1]
    caption = message.caption or "What's in this image?"

    # Send typing indicator
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    async with async_session_factory() as session:
        service = UserService(session)
        settings = await service.get_user_settings(user.id)

        # Check if current model supports vision
        if not openrouter_client.is_vision_model(settings["model"]):
            await message.answer(
                f"⚠️ <b>Current model doesn't support images.</b>\n\n"
                f"Your model (<code>{html.escape(settings['model'])}</code>) cannot process images. "
                f"Please switch to a vision-capable model using /models.\n\n"
                f"<b>Vision models:</b> GPT-4o, GPT-4o Mini, Claude 3.5 Sonnet, "
                f"Claude 3 Haiku, Gemini 2.0 Flash"
            )
            return

        # Download the photo
        file = await message.bot.get_file(photo.file_id)
        image_bytes = await message.bot.download_file(file.file_path)
        image_data = image_bytes.read()

        # Get conversation history
        history = await service.get_history(user.id)

        # Build messages array
        messages = [
            {"role": "system", "content": settings["system_prompt"]}
        ]
        messages.extend(history)

        # Save user message with image info
        await service.add_message(
            telegram_id=user.id,
            role="user",
            content=f"[Image] {caption}",
            image_url=photo.file_id,
            model=settings["model"],
        )

        # Get AI response with vision
        response = await openrouter_client.chat_completion_with_image(
            text=caption,
            image_bytes=image_data,
            messages=messages,
            model=settings["model"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
        )

        if response is None:
            await message.answer(
                "❌ <b>Error analyzing the image.</b>\n\n"
                "Please try again later."
            )
            return

        # Save assistant message
        await service.add_message(
            telegram_id=user.id,
            role="assistant",
            content=response,
            model=settings["model"],
        )

    # Send response
    await send_long_message(message, response)


async def send_long_message(message: Message, text: str) -> None:
    """Send a message, splitting if it exceeds Telegram's limit."""
    MAX_LENGTH = 4000  # Leave room for formatting

    # Escape HTML special characters to prevent rendering issues
    escaped_text = html.escape(text)

    if len(escaped_text) <= MAX_LENGTH:
        await message.answer(escaped_text)
    else:
        # Split into chunks
        parts = []
        while len(escaped_text) > MAX_LENGTH:
            # Try to split at a natural boundary
            split_at = escaped_text.rfind("\n\n", 0, MAX_LENGTH)
            if split_at == -1:
                split_at = escaped_text.rfind(". ", 0, MAX_LENGTH)
            if split_at == -1:
                split_at = escaped_text.rfind(" ", 0, MAX_LENGTH)
            if split_at == -1:
                split_at = MAX_LENGTH
            else:
                split_at += 1  # Include the delimiter

            parts.append(escaped_text[:split_at].strip())
            escaped_text = escaped_text[split_at:].strip()

        if escaped_text:
            parts.append(escaped_text)

        for i, part in enumerate(parts):
            header = f"<i>(Part {i + 1}/{len(parts)})</i>\n\n" if len(parts) > 1 else ""
            await message.answer(header + part)
