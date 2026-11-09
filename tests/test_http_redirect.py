"""Диагностика отключённых редиректов не должна раскрывать секреты."""

from __future__ import annotations

import logging
import unittest

from bot.services.http_redirect import warn_redirect_blocked


class RedirectWarningTests(unittest.TestCase):
    def test_warning_keeps_only_the_host(self):
        url = "https://api.example.org/bot12345:VERY_SECRET_TOKEN/sendMessage"
        with self.assertLogs("bot.services.http_redirect", level=logging.WARNING) as logs:
            warn_redirect_blocked("Test client", 307, url)
        message = "\n".join(logs.output)
        self.assertIn("Test client", message)
        self.assertIn("307", message)
        self.assertIn("api.example.org", message)
        self.assertNotIn("VERY_SECRET_TOKEN", message)
        self.assertNotIn("sendMessage", message)

    def test_empty_url_does_not_raise(self):
        with self.assertLogs("bot.services.http_redirect", level=logging.WARNING):
            warn_redirect_blocked("Test client", 308, "")


if __name__ == "__main__":
    unittest.main()
