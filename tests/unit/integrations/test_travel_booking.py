from mini_devin.integrations.travel_booking import render_travel_booking_prompt


def test_render_travel_booking_prompt_includes_live_preview_and_safety_rules():
    prompt = render_travel_booking_prompt(
        destination="Dubai",
        check_in="2026-05-10",
        check_out="2026-05-13",
        guests="2 adults",
    )

    assert "Dubai" in prompt
    assert "live preview" in prompt.lower()
    assert "Do not submit payment details." in prompt
    assert "final booking confirmation" in prompt


def test_render_travel_booking_prompt_can_disable_sample_data():
    prompt = render_travel_booking_prompt(use_sample_data=False, live_preview=False)

    assert "Do not invent trip details" in prompt
    assert "live preview is optional" in prompt.lower()
