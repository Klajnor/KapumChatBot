"""Диагностика намеренно отключённых HTTP-редиректов.

AI-клиент и Telegraph не повторяют 307/308 на другой host: вместе с телом
запроса ушли бы история, файлы, голос или access_token. Но «молчаливый» 3xx
выглядит для администратора как «провайдер не отвечает». Помогаем отличить
ограничение от сетевого сбоя, не печатая URL с токеном, заголовки и тело.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


def warn_redirect_blocked(client: str, status: int, url: str) -> None:
    """Записать в журнал только имя клиента, статус и хост настроенной точки."""
    host = urlsplit(url).hostname or "unknown"
    logger.warning(
        "%s: HTTP %s redirect is intentionally not followed (request secrets "
        "must stay on the configured host). Check the endpoint/proxy for %s",
        client,
        status,
        host,
    )
