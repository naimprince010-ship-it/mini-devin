"""
Reusable prompt builders for travel / hotel booking tasks.

These helpers keep booking-flow prompts consistent across CLI, API, and tests.
"""

from __future__ import annotations

from typing import Iterable


DEFAULT_SENSITIVE_STOPS: tuple[str, ...] = (
    "login",
    "captcha",
    "OTP",
    "identity verification",
    "payment entry",
    "final booking confirmation",
)


def render_travel_booking_prompt(
    *,
    site_name: str = "Expedia",
    site_url: str = "https://www.expedia.com/",
    destination: str | None = None,
    check_in: str | None = None,
    check_out: str | None = None,
    guests: str | None = None,
    live_preview: bool = True,
    use_sample_data: bool = True,
    stop_events: Iterable[str] = DEFAULT_SENSITIVE_STOPS,
) -> str:
    """Build a safe browser-agent prompt for hotel booking exploration."""
    details: list[str] = []
    if destination:
        details.append(f"destination `{destination}`")
    if check_in:
        details.append(f"check-in `{check_in}`")
    if check_out:
        details.append(f"check-out `{check_out}`")
    if guests:
        details.append(f"guests `{guests}`")

    detail_text = ", ".join(details) if details else "sample search inputs"
    preview_line = (
        "Keep live preview active so I can follow the browser state."
        if live_preview
        else "Short status updates are enough; live preview is optional."
    )
    sample_line = (
        "If exact trip details are missing, use safe sample data and clearly label it as sample data."
        if use_sample_data
        else "Do not invent trip details; only use information I explicitly provided."
    )
    stop_line = ", ".join(str(item).strip() for item in stop_events if str(item).strip())

    return (
        f"Go to {site_name} at {site_url} and explore the hotel booking flow using {detail_text}.\n\n"
        "Goals:\n"
        "1. Open the hotel search flow.\n"
        "2. Identify the destination, date, and guest controls.\n"
        "3. Run a hotel search.\n"
        "4. Compare the cheapest, best rated, and best value options.\n"
        "5. Open one promising property and continue until the checkout page.\n"
        "6. Give short progress updates after each major step.\n\n"
        "Rules:\n"
        f"- {preview_line}\n"
        f"- {sample_line}\n"
        f"- Stop immediately if you hit any of: {stop_line}.\n"
        "- Do not submit payment details.\n"
        "- Do not click the final booking confirmation button without explicit approval.\n"
        "- End with a short summary of what was automated and what still needs manual input.\n"
    )
