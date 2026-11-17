"""Личное обращение к пользователю в системном промпте.

Владелец задаёт в карточке ``/admin`` одно или несколько имён через точку с
запятой (например, «Александр; Саша»). Если личное имя не задано, бот может
использовать имя из профиля Telegram (или ник). В системный промпт такое
указание попадает не в каждом ответе: примерно в половине случаев и со
случайно выбранным вариантом, чтобы бот не начинал каждую реплику одинаково.
"""

from __future__ import annotations

import random
import re

# Границы хранения: колонка users.personal_address — VARCHAR(256).
MAX_ADDRESS_LENGTH = 256
MAX_VARIANTS = 8
MAX_VARIANT_LENGTH = 48
# Доля ответов, где к собеседнику обращаются по имени.
ADDRESS_HINT_CHANCE = 0.5

_HINT_TEMPLATES = (
    "В этом ответе обратись к собеседнику по имени: «{name}».",
    "Собеседника зовут «{name}» — обратись к нему по имени.",
    "Обратись к пользователю по имени «{name}», естественно, в начале ответа.",
)

_SPACES = re.compile(r"\s+")
# Символы, которые нельзя пропускать в имя из профиля: ими можно закрыть
# кавычки-обёртку или подмешать разметку в системный промпт.
_UNSAFE_PROFILE_CHARS = str.maketrans({char: " " for char in "«»<>`"})


def split_personal_address(value: str | None) -> list[str]:
    """Разбить сохранённое значение на варианты обращения.

    Разделитель — точка с запятой; пустые части и повторы (без учёта
    регистра) отбрасываются. Переводы строк схлопываются в пробел, чтобы
    значение не могло разорвать системный промпт.
    """
    if not value:
        return []
    seen: set[str] = set()
    variants: list[str] = []
    for part in str(value).split(";"):
        name = _SPACES.sub(" ", part).strip()[:MAX_VARIANT_LENGTH].strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        variants.append(name)
        if len(variants) >= MAX_VARIANTS:
            break
    return variants


def normalize_personal_address(value: str | None) -> str | None:
    """Каноническое значение для хранения; ``None`` — обращение не задано."""
    variants = split_personal_address(value)
    if not variants:
        return None
    return "; ".join(variants)[:MAX_ADDRESS_LENGTH]


def profile_name(first_name: str | None, username: str | None) -> str | None:
    """Имя из профиля Telegram: сначала имя, затем ник без ``@``.

    Профиль задаёт сам пользователь, поэтому значение очищается: переводы
    строк схлопываются, опасные для промпта символы заменяются пробелами,
    длина ограничена.
    """
    for value in (first_name, username):
        if not value:
            continue
        name = _SPACES.sub(
            " ", str(value).translate(_UNSAFE_PROFILE_CHARS)
        ).strip()[:MAX_VARIANT_LENGTH].strip()
        if name:
            return name
    return None


def personal_address_hint(
    value: str | None, *, rng: random.Random | None = None
) -> str:
    """Случайное указание обратиться по имени; пустая строка — без указания."""
    variants = split_personal_address(value)
    if not variants:
        return ""
    picker = rng or random
    if picker.random() >= ADDRESS_HINT_CHANCE:
        return ""
    name = picker.choice(variants)
    return picker.choice(_HINT_TEMPLATES).format(name=name)


def with_personal_address(
    prompt: str, value: str | None, *, rng: random.Random | None = None
) -> str:
    """Дописать к промпту случайное указание об обращении по имени."""
    hint = personal_address_hint(value, rng=rng)
    return f"{prompt}\n\n{hint}" if hint else prompt


def system_prompt_for(settings: dict, *, rng: random.Random | None = None) -> str:
    """Системный промпт пользователя с возможным обращением по имени.

    Приоритет: имя, заданное владельцем, затем имя/ник из профиля Telegram.
    """
    name = settings.get("address") or settings.get("name")
    return with_personal_address(settings["system_prompt"], name, rng=rng)