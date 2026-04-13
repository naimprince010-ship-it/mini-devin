"""Plodder tool adapters (browser, etc.)."""

from plodder.tools.browser_heuristics import (
    build_vision_user_content,
    map_llm_click_to_pixels,
    looks_like_frontend_test_failure,
)
from plodder.tools.browser_manager import BrowserManager, capture_url_screenshot_base64

__all__ = [
    "BrowserManager",
    "build_vision_user_content",
    "capture_url_screenshot_base64",
    "map_llm_click_to_pixels",
    "looks_like_frontend_test_failure",
]
