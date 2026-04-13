"""Shell output cleanup — ANSI stripping and safe decoding (OpenHands-style hygiene)."""

from __future__ import annotations

import re

# CSI / OSC / common 7-bit escape sequences (good enough for terminal tool output)
_ANSI_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))"
)


def strip_ansi(data: str | bytes) -> str:
    """Remove ANSI color / cursor / OSC sequences for stable logs and LLM input."""
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="replace")
    else:
        text = data
    return _ANSI_RE.sub("", text)


def strip_ansi_bytes(data: bytes) -> bytes:
    return strip_ansi(data).encode("utf-8", errors="replace")
