"""Клиент REST API Redmine (один конкретный Redmine из .env)."""

from __future__ import annotations

from typing import Any

import aiohttp

PAGE_SIZE = 100


class RedmineClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"X-Redmine-API-Key": api_key}

    def issue_url(self, issue_id: int) -> str:
        return f"{self._base}/issues/{issue_id}"

    async def _get(
        self, session: aiohttp.ClientSession, path: str, **params: Any
    ) -> dict[str, Any]:
        async with session.get(
            f"{self._base}{path}", headers=self._headers, params=params
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def current_user_id(self, session: aiohttp.ClientSession) -> int:
        """ID пользователя-владельца в Redmine (по API-ключу)."""
        return (await self._get(session, "/users/current.json"))["user"]["id"]

    async def assigned_issues(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Открытые задачи, назначенные на владельца (постранично)."""
        issues: list[dict[str, Any]] = []
        offset = 0
        while True:
            data = await self._get(
                session,
                "/issues.json",
                assigned_to_id="me",
                status_id="open",
                sort="id:asc",
                limit=PAGE_SIZE,
                offset=offset,
            )
            batch = data.get("issues", [])
            issues.extend(batch)
            offset += PAGE_SIZE
            total = data.get("total_count", len(issues))
            if not batch or offset >= total:
                return issues

    async def issue_journals(
        self, session: aiohttp.ClientSession, issue_id: int
    ) -> list[dict[str, Any]]:
        """Журнал изменений задачи (комментарии-«ответы» и правки полей)."""
        data = await self._get(session, f"/issues/{issue_id}.json", include="journals")
        return data["issue"].get("journals", [])

    async def get_issue(
        self, session: aiohttp.ClientSession, issue_id: int
    ) -> dict[str, Any]:
        """Одна задача по ID (subject, description, ...)."""
        data = await self._get(session, f"/issues/{issue_id}.json")
        return data["issue"]

    async def issue_with_journals(
        self, session: aiohttp.ClientSession, issue_id: int
    ) -> dict[str, Any]:
        """Задача + journals одним запросом (для резюме и baseline)."""
        data = await self._get(session, f"/issues/{issue_id}.json", include="journals")
        return data["issue"]

    @staticmethod
    def customer_name(issue: dict[str, Any]) -> str:
        """Название заказчика из кастомного поля задачи с именем, содержащим «заказчик»."""
        for cf in issue.get("custom_fields") or []:
            if "заказчик" in (cf.get("name") or "").lower():
                value = cf.get("value")
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                value = str(value or "").strip()
                if value:
                    return value
        return ""

    @staticmethod
    def summary_source(issue: dict[str, Any]) -> str:
        """Единый источник текста для LLM-резюме по задаче.

        Тема + описание + первые 4 и последние 4 непустых комментария.
        """
        lines = [issue.get("subject") or "", issue.get("description") or ""]
        notes = [
            j for j in issue.get("journals") or []
            if (j.get("notes") or "").strip()
        ]
        picked = notes[:4]
        seen = {j.get("id") for j in picked}
        picked += [j for j in notes[-4:] if j.get("id") not in seen]
        for j in picked:
            author = j.get("user", {}).get("name", "?")
            lines.append(f"Комментарий ({author}): {j['notes'].strip()}")
        return "\n\n".join(x for x in lines if x.strip())
