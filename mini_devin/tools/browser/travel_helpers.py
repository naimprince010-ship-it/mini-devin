"""
Helpers for extracting and ranking hotel-like search results from browser pages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence


_PRICE_RE = re.compile(r"([$€£])\s?(\d[\d,]*(?:\.\d{1,2})?)")
_REVIEWS_RE = re.compile(r"\b(\d[\d,]*)\s*(?:reviews?|ratings?)\b", re.I)
_RATING_PATTERNS = (
    re.compile(r"\b([0-5](?:\.\d)?)\s*/\s*5\b"),
    re.compile(r"\b([0-5](?:\.\d)?)\s+(?:out of 5|stars?)\b", re.I),
)


@dataclass
class HotelOption:
    """Normalized hotel option extracted from a page."""

    name: str
    price_text: str | None = None
    price_value: float | None = None
    rating: float | None = None
    review_count: int | None = None
    free_cancellation: bool = False
    breakfast_included: bool = False
    url: str | None = None
    raw_text: str = ""
    best_value_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_hotel_options(raw_candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Normalize raw DOM candidates and derive cheapest / best-rated / best-value picks."""
    options = [opt for opt in (_normalize_candidate(item) for item in raw_candidates) if opt]
    if len(options) < 2:
        return None

    prices = [opt.price_value for opt in options if opt.price_value is not None]
    ratings = [opt.rating for opt in options if opt.rating is not None]
    reviews = [opt.review_count for opt in options if opt.review_count is not None]

    for option in options:
        option.best_value_score = _score_best_value(option, prices, ratings, reviews)

    cheapest = min(
        (opt for opt in options if opt.price_value is not None),
        key=lambda opt: opt.price_value,
        default=None,
    )
    best_rated = max(
        (opt for opt in options if opt.rating is not None),
        key=lambda opt: (opt.rating or 0.0, opt.review_count or 0),
        default=None,
    )
    best_value = max(
        options,
        key=lambda opt: (opt.best_value_score or 0.0, opt.rating or 0.0, -(opt.price_value or 10**9)),
        default=None,
    )

    summary = {
        "count": len(options),
        "cheapest": cheapest.to_dict() if cheapest else None,
        "best_rated": best_rated.to_dict() if best_rated else None,
        "best_value": best_value.to_dict() if best_value else None,
        "top_options": [opt.to_dict() for opt in sorted(options, key=_sort_key)[:5]],
        "notes": _build_notes(options),
    }
    return summary


def format_hotel_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact plain-text summary for the LLM / logs."""
    lines = [f"Detected {summary.get('count', 0)} hotel-like options."]
    cheapest = summary.get("cheapest")
    best_rated = summary.get("best_rated")
    best_value = summary.get("best_value")
    if cheapest:
        lines.append("Cheapest: " + _format_option_line(cheapest))
    if best_rated:
        lines.append("Best rated: " + _format_option_line(best_rated))
    if best_value:
        lines.append("Best value: " + _format_option_line(best_value))
    notes = summary.get("notes") or []
    for note in notes[:3]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _normalize_candidate(item: Mapping[str, Any]) -> HotelOption | None:
    text = _clean_text(item.get("text"))
    name = _clean_text(item.get("name")) or _guess_name_from_text(text)
    if not name:
        return None

    price_text = _clean_text(item.get("price_text")) or _find_price_text(text)
    price_value = _parse_price_value(price_text)
    rating = _parse_rating(item.get("rating"), text)
    review_count = _parse_review_count(item.get("review_count"), text)

    option = HotelOption(
        name=name,
        price_text=price_text or None,
        price_value=price_value,
        rating=rating,
        review_count=review_count,
        free_cancellation=_truthy(item.get("free_cancellation")) or ("free cancellation" in text.lower()),
        breakfast_included=_truthy(item.get("breakfast_included"))
        or ("breakfast included" in text.lower())
        or ("includes breakfast" in text.lower()),
        url=_clean_text(item.get("href")) or None,
        raw_text=text,
    )
    if option.price_value is None and option.rating is None and option.review_count is None:
        return None
    return option


def _score_best_value(
    option: HotelOption,
    prices: Sequence[float | None],
    ratings: Sequence[float | None],
    reviews: Sequence[int | None],
) -> float:
    score = 0.0
    if option.price_value is not None:
        low = min(p for p in prices if p is not None)
        high = max(p for p in prices if p is not None)
        score += 0.45 * _invert_normalize(option.price_value, low, high)
    if option.rating is not None:
        low = min(r for r in ratings if r is not None)
        high = max(r for r in ratings if r is not None)
        score += 0.35 * _normalize(option.rating, low, high)
    if option.review_count is not None:
        low = min(r for r in reviews if r is not None)
        high = max(r for r in reviews if r is not None)
        score += 0.12 * _normalize(math.log1p(option.review_count), math.log1p(low), math.log1p(high))
    if option.free_cancellation:
        score += 0.05
    if option.breakfast_included:
        score += 0.03
    return round(score, 4)


def _build_notes(options: Sequence[HotelOption]) -> list[str]:
    notes: list[str] = []
    if any(opt.free_cancellation for opt in options):
        notes.append("Some options mention free cancellation.")
    if any(opt.breakfast_included for opt in options):
        notes.append("Breakfast-inclusive options are available.")
    priced = [opt.price_value for opt in options if opt.price_value is not None]
    if len(priced) >= 2:
        notes.append(f"Visible prices range from {min(priced):.0f} to {max(priced):.0f}.")
    return notes


def _sort_key(option: HotelOption) -> tuple[float, float, float]:
    return (
        -(option.best_value_score or 0.0),
        -(option.rating or 0.0),
        option.price_value if option.price_value is not None else 10**9,
    )


def _format_option_line(option: Mapping[str, Any]) -> str:
    bits = [str(option.get("name", "Unknown option"))]
    if option.get("price_text"):
        bits.append(f"price {option['price_text']}")
    if option.get("rating") is not None:
        bits.append(f"rating {option['rating']}")
    if option.get("review_count") is not None:
        bits.append(f"{option['review_count']} reviews")
    if option.get("free_cancellation"):
        bits.append("free cancellation")
    if option.get("breakfast_included"):
        bits.append("breakfast included")
    return " | ".join(bits)


def _parse_price_value(value: str | None) -> float | None:
    if not value:
        return None
    match = _PRICE_RE.search(value)
    if not match:
        return None
    try:
        return float(match.group(2).replace(",", ""))
    except ValueError:
        return None


def _find_price_text(text: str) -> str:
    match = _PRICE_RE.search(text)
    return match.group(0) if match else ""


def _parse_rating(raw_rating: Any, text: str) -> float | None:
    if raw_rating not in (None, ""):
        try:
            value = float(str(raw_rating).strip())
            if 0.0 <= value <= 5.0:
                return value
        except ValueError:
            pass
    for pattern in _RATING_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _parse_review_count(raw_reviews: Any, text: str) -> int | None:
    if raw_reviews not in (None, ""):
        try:
            return int(str(raw_reviews).replace(",", "").strip())
        except ValueError:
            pass
    match = _REVIEWS_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _guess_name_from_text(text: str) -> str:
    for line in text.splitlines():
        cleaned = _clean_text(line)
        if cleaned and len(cleaned) >= 4:
            return cleaned[:140]
    return ""


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _invert_normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ((value - low) / (high - low))))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
