"""ANSI stripping for terminal output."""

from mini_devin.sandbox.shell_text import strip_ansi


def test_strip_ansi_removes_sgr():
    raw = "\x1b[31merror\x1b[0m plain"
    assert strip_ansi(raw) == "error plain"


def test_strip_ansi_bytes():
    assert strip_ansi(b"\x1b[1mok\x1b[0m") == "ok"
