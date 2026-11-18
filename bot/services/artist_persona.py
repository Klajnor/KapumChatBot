"""Личность художника для генерации картинок.

Выбранная пользователем персона (/persona) — это обычный системный промпт.
В чате она задаёт характер собеседника, а при генерации картинки становится
характером художника: настроением, палитрой и манерой исполнения. Здесь
хранится единственное место, где промпт дополняется этим указанием, чтобы
подпись, история и текст запроса пользователю не менялись.

Стандартный промпт «Ассистент» ничего не добавляет: он не описывает
художника, и без него поведение для пользователей без выбранной персоны
остаётся прежним.
"""

from __future__ import annotations

import re

from bot.config import config

# Персона — пользовательский текст: ограничиваем длину, чтобы не раздувать
# запрос к платной генерации и не терять из вида сам сюжет картинки.
MAX_ARTIST_PERSONA_CHARS = 600

_WHITESPACE = re.compile(r"\s+")


def artist_style(persona: str | None) -> str:
    """Сжать персону в однострочное описание стиля; пустая строка — без стиля."""
    text = _WHITESPACE.sub(" ", persona or "").strip()
    if not text:
        return ""
    default = _WHITESPACE.sub(" ", config.default_system_prompt or "").strip()
    if text == default:
        return ""
    return text[:MAX_ARTIST_PERSONA_CHARS]


def with_artist_persona(prompt: str, persona: str | None) -> str:
    """Дописать к промпту генерации выбранную личность как личность художника."""
    style = artist_style(persona)
    if not style:
        return prompt
    return (
        f"{prompt.strip()}\n\n"
        "Artist persona — paint this picture in the spirit of the artist below: "
        "keep their character, mood, palette and rendering style, while the "
        "requested subject, composition and any text stay exactly as described. "
        f"Artist: {style}"
    )