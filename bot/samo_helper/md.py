"""Помощники для Telegram MarkdownV2: экранирование и разбивка длинных текстов."""

from __future__ import annotations

# Символы, обязательные к экранированию в MarkdownV2 (вне code-блоков).
_MD2_SPECIALS = set(r"_*[]()~`>#+-=|{}.!") | {"\\"}

# Запас от лимита Telegram 4096 — на префикс (i/n).
PART_LIMIT = 4000


def md_escape(text: str) -> str:
    """Экранировать динамический текст под MarkdownV2."""
    return "".join("\\" + c if c in _MD2_SPECIALS else c for c in text)


def bold(text: str) -> str:
    """Полужирный: контент экранируется целиком."""
    return "*" + md_escape(text) + "*"


def link(label: str, url: str) -> str:
    """Ссылка [label](url): label экранируется, в url маскируем скобки."""
    dest = url.replace("\\", "\\\\").replace("]", "\\]").replace(")", "\\)")
    return f"[{md_escape(label)}]({dest})"


def sources_block(items: list[tuple[str, str]], heading: str = "Источники") -> str:
    """MarkdownV2-блок ссылок; items — (подпись, url), дубли URL отбрасываются."""
    seen: set[str] = set()
    lines: list[str] = []
    for label, url in items:
        if not url or url in seen:
            continue
        seen.add(url)
        lines.append("• " + link(label or url, url))
    if not lines:
        return ""
    return "\n\n📚 " + bold(heading) + ":\n" + "\n".join(lines)


def _safe_cut(text: str, cut: int) -> int:
    """Не резать между обратным слэшем и экранированным символом."""
    backslashes = 0
    j = cut - 1
    while j >= 0 and text[j] == "\\":
        backslashes += 1
        j -= 1
    return cut - 1 if backslashes % 2 == 1 else cut


def split_message(text: str, limit: int = PART_LIMIT) -> list[str]:
    """Разбить готовый (уже экранированный) текст на части <= limit.

    Инвариант: "".join(частей) == исходный текст (разделители остаются в конце
    частей). Сначала режем по переводам строк, затем по пробелам, в крайнем
    случае — жёстко по limit.
    """
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = window.rfind("\n") + 1  # 0, если переводов нет
        if cut < limit // 4:  # перевод слишком далеко — пробуем пробел
            sp = window.rfind(" ") + 1
            cut = sp if sp >= limit // 4 else limit
        cut = max(_safe_cut(text, cut), 1)
        parts.append(text[:cut])
        text = text[cut:]
    if text:
        parts.append(text)
    return parts


def with_part_headers(parts: list[str]) -> list[str]:
    """Для >1 части добавить MarkdownV2-безопасный префикс (i/n)."""
    n = len(parts)
    if n <= 1:
        return parts
    return [rf"\({i}/{n}\)" + "\n" + p for i, p in enumerate(parts, 1)]
