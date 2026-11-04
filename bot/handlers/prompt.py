from __future__ import annotations

import html

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.config import config, persona_matches
from bot.keyboards.inline import get_persona_keyboard
from bot.services.telegram_utils import (
    require_private_chat,
    safe_answer_callback,
    safe_edit_text,
)
from bot.services.user_service import user_service

router = Router()


@router.message(Command("prompt"))
async def cmd_prompt(message: Message) -> None:
    """Handle /prompt command — view or set system prompt."""
    user = message.from_user
    if user is None:
        return
    # A system prompt is personal (and may contain private context).
    if not await require_private_chat(message, what="/prompt"):
        return

    # Extract prompt text if provided
    text = message.text or ""
    parts = text.split(maxsplit=1)

    async with user_service() as service:

        if len(parts) > 1:
            # Set new prompt
            new_prompt = parts[1]
            if len(new_prompt) > config.max_system_prompt_length:
                await message.answer(
                    f"⚠️ <b>System prompt is too long.</b>\n\n"
                    f"Maximum length is {config.max_system_prompt_length} characters. "
                    f"Your prompt is {len(new_prompt)} characters.\n\n"
                    f"<i>Shorten it and try again.</i>"
                )
                return
            await service.set_system_prompt(user.id, new_prompt)
            await message.answer(
                f"✅ <b>System prompt updated!</b>\n\n"
                f"<b>New prompt ({len(new_prompt)} characters):</b>\n"
                f"<code>{html.escape(new_prompt)}</code>"
            )
        else:
            # Show current prompt
            settings = await service.get_user_settings(user.id)
            await message.answer(
                f"📝 <b>Current System Prompt</b> "
                f"({len(settings['system_prompt'])} / {config.max_system_prompt_length}):\n\n"
                f"<code>{html.escape(settings['system_prompt'])}</code>\n\n"
                f"<i>To change it, use:</i>\n"
                f"<code>/prompt Your new system prompt here</code>"
            )


# --- AI personas (preset system prompts) --------------------------------------


def _active_persona_name(current_prompt: str) -> str:
    """Display name of the persona whose prompt matches, else "Custom"."""
    for persona in config.personas.values():
        if persona_matches(current_prompt, persona):
            return persona.get("name", "Custom")
    return "Custom"


@router.message(Command("persona"))
async def cmd_persona(message: Message) -> None:
    """Handle /persona — pick a ready-made AI personality (a system prompt)."""
    user = message.from_user
    if user is None:
        return

    async with user_service() as service:
        settings = await service.get_user_settings(user.id)

    current = settings["system_prompt"]
    active = _active_persona_name(current)
    await message.answer(
        "🎭 <b>AI Personas</b>\n\n"
        f"Current: <b>{html.escape(active)}</b>\n\n"
        "Choose a persona — it replaces your system prompt. "
        "A persona is a ready-made prompt; you can still write a fully custom "
        "one with /prompt.",
        reply_markup=get_persona_keyboard(config.personas, current),
    )


@router.callback_query(F.data.startswith("persona:"))
async def cb_persona(callback: CallbackQuery) -> None:
    """Apply the selected persona as the user's system prompt."""
    if callback.data is None or callback.from_user is None:
        await safe_answer_callback(callback)
        return

    key = callback.data.split(":", 1)[1]
    persona = config.personas.get(key)
    if persona is None:
        await safe_answer_callback(callback, "⚠️ Unknown persona.", show_alert=True)
        return

    prompt_text = persona.get("prompt", "")
    if len(prompt_text) > config.max_system_prompt_length:
        await safe_answer_callback(
            callback,
            "⚠️ This persona's prompt exceeds the length limit.",
            show_alert=True,
        )
        return

    async with user_service() as service:
        await service.set_system_prompt(callback.from_user.id, prompt_text)

    await safe_answer_callback(callback, f"✅ Persona: {persona.get('name', key)}")

    if callback.message is not None:
        await safe_edit_text(
            callback.message,
            f"🎭 <b>Persona set:</b> {html.escape(persona.get('name', key))}\n\n"
            f"<i>{html.escape(prompt_text[:300])}</i>"
            f"{'…' if len(prompt_text) > 300 else ''}\n\n"
            "Change anytime with /persona, or write a custom prompt with /prompt.",
            reply_markup=get_persona_keyboard(config.personas, prompt_text),
        )


@router.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    """Handle /settings command — show current settings."""
    user = message.from_user
    if user is None:
        return
    # Prints the system prompt: private chat only, like /prompt.
    if not await require_private_chat(message, what="/settings"):
        return

    async with user_service() as service:
        settings = await service.get_user_settings(user.id)

    await message.answer(
        f"⚙️ <b>Your Settings</b>\n\n"
        f"🤖 <b>Model:</b> <code>{html.escape(settings['model'])}</code>\n"
        f"🎭 <b>Persona:</b> {html.escape(_active_persona_name(settings['system_prompt']))}\n"
        f"🌡️ <b>Temperature:</b> <code>{settings['temperature']}</code>\n"
        f"📏 <b>Max Tokens:</b> <code>{settings['max_tokens']}</code>\n\n"
        f"📝 <b>System Prompt:</b>\n"
        f"<code>{html.escape(settings['system_prompt'][:200])}</code>\n"
        f"{'<i>...(truncated)</i>' if len(settings['system_prompt']) > 200 else ''}"
    )