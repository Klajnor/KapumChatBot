from __future__ import annotations

import unittest

from bot.services.pillow_limits import MAX_IMAGE_PIXELS, configure_pillow_limits


class PillowLimitsTests(unittest.TestCase):
    def test_configure_sets_explicit_pixel_budget(self) -> None:
        from PIL import Image

        old = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            configure_pillow_limits()
            self.assertEqual(Image.MAX_IMAGE_PIXELS, MAX_IMAGE_PIXELS)
        finally:
            Image.MAX_IMAGE_PIXELS = old


if __name__ == "__main__":
    unittest.main()
