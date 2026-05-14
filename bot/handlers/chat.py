from __future__ import annotations

import io
import logging
import random

from aiogram import Router, F
from aiogram.types import BufferedInputFile, Message
from bot.config import config
from bot.database.session import async_session_factory
from bot.services.openrouter import openrouter_client
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

router = Router()

# Non-offensive jokes for non-admin users (Jedi mind trick style)
_NON_ADMIN_JOKES = [
    "👋 Привет! Тут нет ИИ-дроидов, которых вы ищете. Попробуйте поискать в другом секторе галактики.",
    "🧘 Это не те ответы, которые вы ищете. Двигайтесь дальше, странник.",
    "🤖 ИИ-бот? Здесь? Нет-нет, это просто скриншот работающего кода. Ничего интересного.",
    "🌌 В этом чате нет разумной жизни. Только эхо и тишина. Идите с миром.",
    "🕵️ Ошибка 404: ИИ не найден. Ваш запрос улетел в открытый космос.",
    '⚡ "Эти дроиды не те, кто вам нужен". Попробуйте /start, но предупреждаю — тут скучно.',
    "🌀 Вы видите то, что хотите видеть. А хотите вы увидеть пустой чат. Так и есть.",
    "🚀 Корабль улетел без ИИ. Приходите через тысячу световых лет.",
    "🎭 *включает режим «просто декоративный бот»* — Извините, все настоящее ИИ закончилось вчера.",
    "🔮 Бот говорит: «Мне нет до вас дела». Но вежливо. Очень вежливо.",
    "📡 Сигнал потерян. Возможно, вас глушит атмосфера. А может, тут просто нет бота.",
    "🧠 Нейросеть ушла в запой. Вернётся через 42 дня. Заходите позже!",
]


@router.message(F.text)
async def handle_text(message: Message) -> None:
    """Handle text messages — chat with AI via OpenRouter."""
    user = message.from_user
    if user is None or message.text is None:
        return

    # Check if user is admin
    if user.id not in config.admin_ids:
        joke = random.choice(_NON_ADMIN_JOKES)
        await message.answer(joke)
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
    """Handle photo messages — analyze image with vision model."""
    user = message.from_user
    if user is None:
        return

    # Check if user is admin
    if user.id not in config.admin_ids:
        joke = random.choice(_NON_ADMIN_JOKES)
        await message.answer(joke)
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
                f"Your model (<code>{settings['model']}</code>) cannot process images. "
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

    if len(text) <= MAX_LENGTH:
        await message.answer(text)
    else:
        # Split into chunks
        parts = []
        while len(text) > MAX_LENGTH:
            # Try to split at a natural boundary
            split_at = text.rfind("\n\n", 0, MAX_LENGTH)
            if split_at == -1:
                split_at = text.rfind(". ", 0, MAX_LENGTH)
            if split_at == -1:
                split_at = text.rfind(" ", 0, MAX_LENGTH)
            if split_at == -1:
                split_at = MAX_LENGTH
            else:
                split_at += 1  # Include the delimiter

            parts.append(text[:split_at].strip())
            text = text[split_at:].strip()

        if text:
            parts.append(text)

        for i, part in enumerate(parts):
            header = f"<i>(Part {i + 1}/{len(parts)})</i>\n\n" if len(parts) > 1 else ""
            await message.answer(header + part)

