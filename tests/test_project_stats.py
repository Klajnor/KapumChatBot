"""PROJECT_STATS.md is generated and must match the current tree."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectStatsTests(unittest.TestCase):
    def test_generated_stats_are_fresh(self):
        result = subprocess.run(
            [sys.executable, "scripts/project_stats.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        rendered = result.stdout.decode("utf-8")
        stored = (ROOT / "PROJECT_STATS.md").read_text(encoding="utf-8")
        self.assertEqual(stored, rendered)
        self.assertIn("| Весь проект |", stored)
        self.assertIn("Итого:", stored)


if __name__ == "__main__":
    unittest.main()
