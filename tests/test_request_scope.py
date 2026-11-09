"""Общая классификация новых запросов для изоляции истории."""

from __future__ import annotations

import unittest

from bot.services.request_scope import (
    is_fresh_request,
    route,
)


class FreshRequestTests(unittest.TestCase):
    def test_new_questions_are_fresh_even_without_an_action_verb(self):
        for text in (
            "Какая сейчас погода в Москве?",
            "Актуальный курс доллара на сегодня",
            "Проверь в интернете последние новости",
            "Что там по проекту X в заметках?",
            "What is the current exchange rate?",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_fresh_request(text))

    def test_ordinary_followups_are_not_fresh(self):
        for text in (
            "объясни второй пункт",
            "Объясни второй пункт",
            "Найди ещё по этой теме",
            "расскажи об этом подробнее",
            "tell me more",
            "what about this",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_fresh_request(text))

    def test_short_replies_are_not_fresh(self):
        for text in ("", "hi", "привет", "спасибо", "@bot hello", None):
            with self.subTest(text=text):
                self.assertFalse(is_fresh_request(text))

    def test_current_untrusted_text_is_handled_by_callers_not_here(self):
        # Маркер проверяют функции изоляции; классификатор смотрит только на
        # самостоятельность сообщения.
        self.assertFalse(is_fresh_request("объясни второй пункт"))


class RouteTests(unittest.TestCase):
    def test_web_and_note_requests(self):
        self.assertEqual(route("найди в интернете плагины для Obsidian"), "web")
        self.assertEqual(route("search my notes for Rust"), "notes")
        self.assertEqual(route("Какая погода в Москве?"), "general")

    def test_earlier_scope_wins_when_both_markers_are_present(self):
        self.assertEqual(route("Найди в интернете плагины в Obsidian"), "web")
        self.assertEqual(
            route("Найди во втором мозге заметки про настройку поиска в интернете"),
            "notes",
        )
        self.assertEqual(route("Найди в моих заметках про интернет"), "notes")

    def test_followups_have_no_route(self):
        self.assertIsNone(route("Найди ещё по этой теме"))

    def test_explicit_comparison_or_both_sources_use_combined_route(self):
        for text in (
            "Найди в моих заметках и в интернете сведения о Rust",
            "Сравни в интернете актуальные данные с тем, что в моих заметках о Rust",
            "Сопоставь во втором мозге план с документацией в интернете",
            "Search my notes and the web for Rust and combine the results",
            "Compare the web documentation with my notes about Rust",
            "Сравни мои заметки про Rust с актуальными данными в интернете",
            "Найди в MCP и в интернете сведения о Rust",
        ):
            with self.subTest(text=text):
                self.assertEqual(route(text), "combined")


if __name__ == "__main__":
    unittest.main()
