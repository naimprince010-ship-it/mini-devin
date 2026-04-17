"""Browser-related heuristics and vision message shaping (no Playwright import)."""

from __future__ import annotations

import base64
from typing import Any


def map_llm_click_to_pixels(
    x: float,
    y: float,
    viewport_width: int,
    viewport_height: int,
) -> tuple[int, int]:
    """
    Map model-provided coordinates to viewport pixels.

    - ``0..1`` on both axes → fractional viewport.
    - ``0..1000`` with max > 1 → millinorm grid scaled to viewport.
    - Otherwise pixel coordinates clamped to the viewport.
    """
    w, h = max(1, viewport_width), max(1, viewport_height)
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return int(x * w), int(y * h)
    if 0.0 < x <= 1000.0 and 0.0 < y <= 1000.0 and max(x, y) > 1.0:
        return int((x / 1000.0) * w), int((y / 1000.0) * h)
    px, py = int(round(x)), int(round(y))
    return max(0, min(px, w - 1)), max(0, min(py, h - 1))


def looks_like_frontend_test_failure(stderr: str, stdout: str) -> bool:
    """Heuristic: failed frontend / e2e / UI test output worth a visual screenshot."""
    blob = f"{stderr}\n{stdout}".lower()
    keys = (
        "vitest",
        "jest",
        ".test.",
        ".spec.",
        "playwright",
        "cypress",
        "failed snapshot",
        "screenshot",
        "css error",
        "syntaxerror",
        "referenceerror",
        "typeerror",
        "e2e",
        "@vite",
        "vitepress",
        "testing library",
        "rtl",
        "enzyme",
        "webpack",
        "rollup",
        "eslint",
    )
    return any(k in blob for k in keys)


def build_vision_user_content(text_block: str, png_bytes: bytes) -> list[dict[str, Any]]:
    """OpenAI-style multimodal ``content`` list (text + image)."""
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    return [
        {"type": "text", "text": text_block},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
    ]
