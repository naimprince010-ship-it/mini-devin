from plodder.tools.browser_heuristics import (
    build_vision_user_content,
    map_llm_click_to_pixels,
    looks_like_frontend_test_failure,
)


def test_map_normalized() -> None:
    assert map_llm_click_to_pixels(0.5, 0.5, 1000, 800) == (500, 400)


def test_map_millinorm() -> None:
    assert map_llm_click_to_pixels(500, 250, 1000, 1000) == (500, 250)


def test_looks_like_frontend() -> None:
    assert looks_like_frontend_test_failure("FAIL src/App.test.tsx", "")
    assert looks_like_frontend_test_failure("", "Error: playwright expectation failed")
    assert not looks_like_frontend_test_failure("ZeroDivisionError", "")


def test_build_vision_user_content() -> None:
    parts = build_vision_user_content("fix ui", b"\x89PNG\r\n\x1a\n")
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
