"""Unit tests for browser grounding helpers (no Playwright required)."""

from plodder.tools.browser_grounding import _normalize_element_id


def test_normalize_element_id() -> None:
    assert _normalize_element_id("p12") == "p12"
    assert _normalize_element_id("P3") == "p3"
    assert _normalize_element_id("12") == "p12"
    assert _normalize_element_id("  7 ") == "p7"
    assert _normalize_element_id("") is None
    assert _normalize_element_id("  ") is None
    assert _normalize_element_id("not-an-id") is None
