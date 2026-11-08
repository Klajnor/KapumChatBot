from __future__ import annotations

# Pillow's default decompression-bomb threshold is far too high for user input:
# a small file can declare hundreds of millions of pixels and force a huge
# native allocation before any post-decode check. Keep this aligned with the
# explicit pixel budgets of image_tools.py and stickers.py.
MAX_IMAGE_PIXELS = 60_000_000


def configure_pillow_limits() -> None:
    """Apply the process-wide Pillow pixel budget (call after importing PIL)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
