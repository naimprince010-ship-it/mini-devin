"""LSP-style diagnostics helpers."""

from mini_devin.lsp.diagnostics import collect_diagnostics


def test_python_syntax_error_in_temp(tmp_path):
    ws = str(tmp_path)
    bad = "def x(\n"
    items, src = collect_diagnostics(ws, "dummy.py", content=bad)
    assert src in ("syntax", "pyright")
    assert items
    assert any("line" in d for d in items)


def test_empty_python_ok(tmp_path):
    ws = str(tmp_path)
    ok = "x = 1\n"
    items, src = collect_diagnostics(ws, "ok.py", content=ok)
    assert isinstance(items, list)
