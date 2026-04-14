"""
Head+tail truncation for huge sandbox streams (token-friendly, OpenHands-style).

Long compiler / test output keeps the beginning (errors, file paths) and the end
(final assertion / summary) while omitting a counted middle section.
"""

from __future__ import annotations

import os


def truncate_stream(
    text: str,
    *,
    max_chars: int | None = None,
    head_chars: int | None = None,
    tail_chars: int | None = None,
) -> tuple[str, bool]:
    """
    Return ``(possibly_truncated, was_truncated)``.

    Env overrides (optional):
    - ``PLODDER_SANDBOX_STREAM_MAX`` — total cap (default 8000)
    - ``PLODDER_SANDBOX_STREAM_HEAD`` — head bytes (default 60% of max minus overhead)
    - ``PLODDER_SANDBOX_STREAM_TAIL`` — tail (default remainder)
    """
    if not text:
        return text, False
    raw_max = max_chars if max_chars is not None else int(os.environ.get("PLODDER_SANDBOX_STREAM_MAX", "8000") or "8000")
    max_c = max(2_000, min(raw_max, 120_000))
    if len(text) <= max_c:
        return text, False

    h_env = os.environ.get("PLODDER_SANDBOX_STREAM_HEAD", "").strip()
    t_env = os.environ.get("PLODDER_SANDBOX_STREAM_TAIL", "").strip()
    if head_chars is not None and tail_chars is not None:
        h, t = head_chars, tail_chars
    elif h_env.isdigit() and t_env.isdigit():
        h, t = int(h_env), int(t_env)
    else:
        h = max(800, int(max_c * 0.62))
        t = max(400, max_c - h - 120)
    overhead = 80
    if h + t + overhead >= len(text):
        return text, False
    h = min(h, max_c // 2)
    t = min(t, max_c - h - overhead)
    omitted = len(text) - h - t
    mid = f"\n… ({omitted} characters omitted) …\n"
    out = text[:h] + mid + text[-t:]
    return out, True
