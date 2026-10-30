from __future__ import annotations

import secrets

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import select

from bot.database.models import GeneratedSticker, StickerPack
from bot.database.session import async_session_factory

PAGE_SIZE = 6


async def save_sticker(user_id: int, bot_id: int, file_id: str, unique_id: str) -> str:
    """Сохранить Telegram-ссылку, не сохраняя сами байты изображения в БД."""
    token = secrets.token_hex(8)
    async with async_session_factory() as session:
        session.add(GeneratedSticker(
            token=token, user_id=user_id, bot_id=bot_id,
            file_id=file_id, file_unique_id=unique_id,
        ))
        await session.commit()
    return token


async def get_sticker(token: str, user_id: int, bot_id: int) -> GeneratedSticker | None:
    """Проверить владельца кнопки и идентичность бота на стороне БД."""
    async with async_session_factory() as session:
        return await session.scalar(select(GeneratedSticker).where(
            GeneratedSticker.token == token,
            GeneratedSticker.user_id == user_id,
            GeneratedSticker.bot_id == bot_id,
        ))


async def list_packs(user_id: int, bot_id: int, page: int) -> list[StickerPack]:
    """Вернуть страницу наборов и одну дополнительную строку для навигации."""
    async with async_session_factory() as session:
        result = await session.scalars(select(StickerPack).where(
            StickerPack.user_id == user_id, StickerPack.bot_id == bot_id,
            StickerPack.created.is_(True),
        ).order_by(StickerPack.id).offset(page * PAGE_SIZE).limit(PAGE_SIZE + 1))
        return list(result)


async def get_pack(pack_id: int, user_id: int, bot_id: int) -> StickerPack | None:
    """Найти только собственный набор пользователя, созданный этим ботом."""
    async with async_session_factory() as session:
        return await session.scalar(select(StickerPack).where(
            StickerPack.id == pack_id, StickerPack.user_id == user_id,
            StickerPack.bot_id == bot_id, StickerPack.created.is_(True),
        ))


async def reserve_pack(sticker: GeneratedSticker, username: str, title: str) -> StickerPack:
    """Зарезервировать воспроизводимое имя набора до обращения к Telegram."""
    name = f"s{sticker.token}_by_{username.lower()}"
    if len(name) > 64:
        raise ValueError("Sticker set name is too long")
    async with async_session_factory() as session:
        pack = await session.scalar(select(StickerPack).where(StickerPack.name == name))
        if pack is None:
            pack = StickerPack(
                user_id=sticker.user_id, bot_id=sticker.bot_id,
                name=name, title=title[:64], created=False,
            )
            session.add(pack)
            await session.commit()
        return pack


async def mark_created(pack_id: int) -> None:
    """Подтвердить создание набора после успешного Telegram-запроса."""
    async with async_session_factory() as session:
        pack = await session.get(StickerPack, pack_id)
        if pack is not None:
            pack.created = True
            await session.commit()


def add_keyboard(token: str) -> InlineKeyboardMarkup:
    """Кнопка выбора набора для уже отправленного стикера."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="➕ Add to sticker set", callback_data=f"stk:menu:{token}:0"),
    ]])


def packs_keyboard(token: str, packs: list[StickerPack], page: int) -> InlineKeyboardMarkup:
    """Выбор существующего набора с пагинацией или создание нового."""
    rows = [[InlineKeyboardButton(
        text=pack.title, callback_data=f"stk:add:{token}:{pack.id}",
    )] for pack in packs[:PAGE_SIZE]]
    rows.append([InlineKeyboardButton(text="➕ Create new set", callback_data=f"stk:new:{token}")])
    nav = []
    if page:
        nav.append(InlineKeyboardButton(text="◀", callback_data=f"stk:menu:{token}:{page - 1}"))
    if len(packs) > PAGE_SIZE:
        nav.append(InlineKeyboardButton(text="▶", callback_data=f"stk:menu:{token}:{page + 1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(inline_keyboard=rows)
