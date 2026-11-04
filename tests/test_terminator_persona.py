from __future__ import annotations

import unittest

from bot.config import DEFAULT_PERSONAS, _parse_personas, config
from bot.keyboards.inline import get_persona_keyboard


class TerminatorPersonaTests(unittest.TestCase):
    """Проверки встроенной личности Терминатора и её кнопки выбора."""

    def test_terminator_is_available_and_selectable(self) -> None:
        personas = _parse_personas("")
        persona = personas["terminator"]
        self.assertEqual(persona, DEFAULT_PERSONAS["terminator"])
        self.assertEqual(persona["name"], "🦾 Терминатор")
        self.assertEqual(persona["self_name"], "Арни")
        self.assertIn("Твоё имя — Арни", persona["prompt"])
        self.assertIn("третьем лице", persona["prompt"])
        self.assertLessEqual(len(persona["prompt"]), config.max_system_prompt_length)
        markup = get_persona_keyboard(personas, persona["prompt"])
        button = next(
            button
            for row in markup.inline_keyboard
            for button in row
            if button.callback_data == "persona:terminator"
        )
        self.assertEqual(button.text, f"{persona['name']} ✓")
