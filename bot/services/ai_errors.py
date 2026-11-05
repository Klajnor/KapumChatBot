from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from http import HTTPStatus


@dataclass(frozen=True, slots=True)
class AIRequestError:
    """Безопасная причина сбоя одной попытки обращения к модели."""

    model: str
    code: str
    message: str = ""


def _safe_text(value: str, secrets: Iterable[str], limit: int) -> str:
    # Маскируем до обрезки: иначе в чат может попасть начало длинного ключа.
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        value = value.replace(secret, "[redacted]")
    value = re.sub(r"(?i)\b(?:https?|wss?|socks5h?)://[^\s<>\"']+", "[URL hidden]", value)
    value = re.sub(r"(?i)\bBearer\s+[^\s,;<>\"']+", "Bearer [redacted]", value)
    value = re.sub(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|authorization|password|secret|token)"
        r"[\"']?\s*[:=]\s*[\"']?[^\s,;<>\"'{}]+",
        r"\1=[redacted]",
        value,
    )
    value = re.sub(r"\bsk-[\w-]+|\b\d{6,}:[\w-]{20,}", "[redacted]", value)
    # Управляющие символы/биди и внутренние ';' не должны подменять список.
    value = "".join(" " if unicodedata.category(char).startswith("C") else char for char in value)
    value = " ".join(value.replace(";", ",").split())
    return value if len(value) <= limit else value[:limit - 1] + "…"


def request_error(
    model: str, code: str, message: str = "", *, secrets: Iterable[str] = (),
) -> AIRequestError:
    """Ограничить поля ошибки и убрать ключи, токены и URL с credentials."""
    secrets = tuple(secrets)
    return AIRequestError(
        model=_safe_text(model, secrets, 96),
        code=_safe_text(code, secrets, 80),
        message=_safe_text(message, secrets, 180),
    )


def provider_error(
    model: str, status: int, body: str, *, secrets: Iterable[str] = (),
) -> AIRequestError:
    """Извлечь HTTP/provider-код и описание, не публикуя сырой JSON/HTML."""
    code = str(status)
    message = ""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        # Короткие plain-text ответы прокси полезны; страницы блокировок — нет.
        if not body.lstrip().startswith(("<", "{", "[")):
            message = body
    else:
        if isinstance(data, dict):
            error = data.get("error", data)
            if isinstance(error, str):
                message = error
            elif isinstance(error, dict):
                provider_code = error.get("code") or error.get("type")
                if isinstance(provider_code, (str, int)) and not isinstance(provider_code, bool):
                    code += f" / {provider_code}"
                for key in ("message", "detail", "error_description"):
                    if isinstance(error.get(key), str) and error[key].strip():
                        message = error[key]
                        break
    if not message:
        try:
            message = HTTPStatus(status).phrase
        except ValueError:
            pass
    return request_error(model, code, message, secrets=secrets)


def format_request_errors(errors: Iterable[AIRequestError]) -> str:
    """Собрать список через ';', объединяя только полностью одинаковые ошибки."""
    entries = []
    for error, count in Counter(errors).items():
        entry = f"{error.model} — {error.code}"
        if error.message:
            entry += f" ({error.message})"
        if count > 1:
            entry += f" ×{count}"
        entries.append(entry)
    return "; ".join(entries)
