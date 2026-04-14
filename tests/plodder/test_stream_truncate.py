"""Tests for sandbox stream head+tail truncation."""

from plodder.sandbox.stream_truncate import truncate_stream


def test_truncate_stream_short_unchanged() -> None:
    s = "hello\n" * 10
    out, trunc = truncate_stream(s, max_chars=10_000)
    assert out == s
    assert trunc is False


def test_truncate_stream_long_inserts_omission_marker() -> None:
    s = "A" * 5000 + "MID" + "B" * 5000
    out, trunc = truncate_stream(s, max_chars=3000)
    assert trunc is True
    assert "omitted" in out
    assert out.startswith("AAA")
    assert out.rstrip().endswith("BBB") or "B" * 100 in out
