"""Personal address: parsing, random prompt hint and /admin editing."""

from __future__ import annotations

import datetime
import random
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from aiogram.types import Chat, Message, User

from bot.config import config
from bot.handlers import admin_panel as panel
from bot.services import digest  # noqa: F401 - imported for the session patch
from bot.services.personal_address import (
    ADDRESS_HINT_CHANCE,
    MAX_VARIANT_LENGTH,
    MAX_VARIANTS,
    normalize_personal_address,
    personal_address_hint,
    profile_name,
    split_personal_address,
    system_prompt_for,
    with_personal_address,
)
from bot.services.user_service import user_service
from tests.test_database import DBTest

OWNER, ADMIN, USER = 1, 2, 3
NOW = datetime.datetime(2026, 11, 17, 12, 0, tzinfo=datetime.UTC)


class _FixedPicker:
    """Deterministic stand-in for ``random.Random``."""

    def __init__(self, value: float, index: int = 0) -> None:
        self.value = value
        self.index = index

    def random(self) -> float:
        return self.value

    def choice(self, sequence):
        return sequence[self.index % len(sequence)]


class TestPersonalAddressFormat(TestCase):
    def test_split_normalizes_parts(self):
        self.assertEqual(
            split_personal_address("  Александр ;; ша ; александр\nМладший "),
            ["Александр", "ша", "александр Младший"],
        )
        self.assertEqual(split_personal_address(None), [])
        self.assertEqual(split_personal_address("  ; ; "), [])

    def test_storage_limits(self):
        many = "; ".join(f"Имя{i}" for i in range(MAX_VARIANTS + 4))
        self.assertEqual(len(split_personal_address(many)), MAX_VARIANTS)

        long_name = "Я" * (MAX_VARIANT_LENGTH + 20)
        self.assertEqual(
            split_personal_address(long_name), ["Я" * MAX_VARIANT_LENGTH]
        )

        normalized = normalize_personal_address("  Александр ; Саша ")
        self.assertEqual(normalized, "Александр; Саша")
        self.assertIsNone(normalize_personal_address("   "))

    def test_hint_is_random_and_uses_one_variant(self):
        included = personal_address_hint(
            "Александр; Саша", rng=_FixedPicker(ADDRESS_HINT_CHANCE / 2, index=1)
        )
        self.assertIn("«Саша»", included)
        self.assertNotIn("«Александр»", included)

        skipped = personal_address_hint(
            "Александр; Саша", rng=_FixedPicker(ADDRESS_HINT_CHANCE)
        )
        self.assertEqual(skipped, "")

    def test_both_variants_and_skips_appear(self):
        rng = random.Random(1234)
        hints = [personal_address_hint("Аня; Анна", rng=rng) for _ in range(200)]
        self.assertTrue(any("«Аня»" in hint for hint in hints))
        self.assertTrue(any("«Анна»" in hint for hint in hints))
        self.assertTrue(any(hint == "" for hint in hints))

    def test_prompt_is_unchanged_without_address(self):
        self.assertEqual(with_personal_address("Базовый промпт", None), "Базовый промпт")
        self.assertEqual(with_personal_address("Базовый промпт", " ; "), "Базовый промпт")
        settings = {"system_prompt": "Базовый промпт"}
        self.assertEqual(system_prompt_for(settings), "Базовый промпт")

    def test_prompt_gets_the_hint(self):
        prompt = with_personal_address(
            "Базовый промпт", "Саша", rng=_FixedPicker(0.0)
        )
        self.assertTrue(prompt.startswith("Базовый промпт\n\n"))
        self.assertIn("«Саша»", prompt)

    def test_profile_name_is_cleaned_and_falls_back_to_username(self):
        self.assertEqual(profile_name("  Иван\nМладший ", "ivan"), "Иван Младший")
        self.assertEqual(profile_name("«Иван»", None), "Иван")
        self.assertEqual(profile_name("", "ivan"), "ivan")
        self.assertIsNone(profile_name(None, None))

    def test_system_prompt_prefers_the_owner_name(self):
        profile = system_prompt_for(
            {"system_prompt": "Промпт", "name": "Иван"}, rng=_FixedPicker(0.0)
        )
        self.assertIn("«Иван»", profile)

        personal = system_prompt_for(
            {"system_prompt": "Промпт", "address": "Саша", "name": "Иван"},
            rng=_FixedPicker(0.0),
        )
        self.assertIn("«Саша»", personal)
        self.assertNotIn("«Иван»", personal)


def _dm(text: str, user_id: int = OWNER) -> Message:
    return Message(
        message_id=1,
        date=NOW,
        chat=Chat(id=user_id, type="private"),
        from_user=User(id=user_id, is_bot=False, first_name="Ann"),
        text=text,
    )


def _callback(data: str, user_id: int = OWNER) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=user_id),
        message=SimpleNamespace(chat=SimpleNamespace(id=user_id, type="private")),
    )


class TestPersonalAddressStorage(DBTest):
    def test_roundtrip_and_settings_key(self):
        async def go() -> None:
            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    await service.register_user(USER)
                    await service.set_personal_address(USER, "Александр; Саша")
                    self.assertEqual(
                        await service.get_personal_address(USER),
                        "Александр; Саша",
                    )
                    self.assertEqual(
                        (await service.get_user_settings(USER))["address"],
                        "Александр; Саша",
                    )

                    await service.set_personal_address(USER, None)
                    self.assertIsNone(await service.get_personal_address(USER))
                    self.assertIsNone(
                        (await service.get_user_settings(USER))["address"]
                    )

        self._loop.run_until_complete(go())

    def test_address_works_for_user_without_profile(self):
        async def go() -> None:
            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    await service.set_personal_address(999, "Иван")
                    self.assertEqual(
                        (await service.get_user_settings(999))["address"], "Иван"
                    )

        self._loop.run_until_complete(go())

    def test_settings_expose_profile_name_only_without_personal_name(self):
        async def go() -> None:
            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    await service.register_user(
                        USER, first_name="Иван", username="ivan"
                    )

                    settings = await service.get_user_settings(USER)
                    self.assertEqual(settings["name"], "Иван")
                    self.assertIsNone(settings["address"])

                    await service.set_personal_address(USER, "Саша")
                    settings = await service.get_user_settings(USER)
                    self.assertIsNone(settings["name"])
                    self.assertEqual(settings["address"], "Саша")

                    await service.set_personal_address(USER, None)
                    settings = await service.get_user_settings(USER)
                    self.assertEqual(settings["name"], "Иван")

        self._loop.run_until_complete(go())


class TestPersonalAddressAdmin(DBTest):
    def _run(self, coro_fn):
        self._answer = AsyncMock()
        self._alert = AsyncMock()
        patches = [
            patch("bot.database.session.async_session_factory", self._factory),
            patch("bot.services.digest.async_session_factory", self._factory),
            patch.object(config, "owner_ids", [OWNER]),
            patch.object(config, "extra_admin_ids", set()),
            patch.object(config, "daily_limit", 3),
            patch.object(panel, "_show", AsyncMock()),
            patch.object(panel, "safe_answer_callback", self._alert),
            patch.object(Message, "answer", self._answer),
        ]
        for p in patches:
            p.start()
        panel._awaiting.clear()
        try:
            return self._loop.run_until_complete(coro_fn())
        finally:
            panel._awaiting.clear()
            for p in reversed(patches):
                p.stop()

    async def _register(self) -> None:
        with patch("bot.database.session.async_session_factory", self._factory):
            async with user_service() as service:
                await service.register_user(USER)
                await service.add_allowed_user(USER)

    def test_owner_sets_and_resets_address(self):
        async def go() -> None:
            await self._register()

            await panel.cb_admin(_callback("adm:nm:3"))
            self.assertEqual(panel._awaiting[OWNER], ("address", USER))

            message = _dm("Александр; Саша; александр")
            self.assertTrue(panel._expecting_input(message))
            await panel.on_admin_input(message)

            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    self.assertEqual(
                        await service.get_personal_address(USER),
                        "Александр; Саша",
                    )

            await panel.cb_admin(_callback("adm:nmr:3"))
            self.assertEqual(panel._awaiting[OWNER], ("reset:nm", USER))
            message = _dm("не подтверждаю")
            self.assertTrue(panel._expecting_input(message))
            await panel.on_admin_input(message)

            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    self.assertEqual(
                        await service.get_personal_address(USER),
                        "Александр; Саша",
                    )

            await panel.cb_admin(_callback("adm:nmr:3"))
            message = _dm("CONFIRM")
            self.assertTrue(panel._expecting_input(message))
            await panel.on_admin_input(message)

            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    self.assertIsNone(await service.get_personal_address(USER))

        self._run(go)

    def test_only_the_owner_can_edit_address(self):
        async def go() -> None:
            await self._register()
            with patch.object(config, "extra_admin_ids", {ADMIN}):
                await panel.cb_admin(_callback("adm:nm:3", user_id=ADMIN))
                self.assertNotIn(ADMIN, panel._awaiting)
                self.assertEqual(
                    self._alert.await_args.args[1], "Owners only"
                )

                panel._awaiting[ADMIN] = ("address", USER)
                message = _dm("Иван", user_id=ADMIN)
                self.assertTrue(panel._expecting_input(message))
                await panel.on_admin_input(message)

            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    self.assertIsNone(await service.get_personal_address(USER))

        self._run(go)

    def test_typed_dash_removes_address(self):
        async def go() -> None:
            await self._register()
            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    await service.set_personal_address(USER, "Иван")

            await panel.cb_admin(_callback("adm:nm:3"))
            message = _dm("-")
            self.assertTrue(panel._expecting_input(message))
            await panel.on_admin_input(message)

            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    self.assertIsNone(await service.get_personal_address(USER))

        self._run(go)