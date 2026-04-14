"""Incremental snapshot cache for Docker sandbox uploads."""

from pathlib import Path

from plodder.workspace.session_workspace import SessionWorkspace


def test_snapshot_cache_incremental_after_write(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    ws = SessionWorkspace(root=root)
    (root / "a.txt").write_text("one", encoding="utf-8")
    (root / "b.txt").write_text("two", encoding="utf-8")
    s1 = ws.snapshot_text_files()
    assert s1["a.txt"] == "one" and s1["b.txt"] == "two"
    ws.write_file("a.txt", "ONE")
    s2 = ws.snapshot_text_files()
    assert s2["a.txt"] == "ONE" and s2["b.txt"] == "two"


def test_snapshot_cache_delete_file(tmp_path: Path) -> None:
    root = tmp_path / "ws2"
    ws = SessionWorkspace(root=root)
    (root / "x.txt").write_text("x", encoding="utf-8")
    ws.snapshot_text_files()
    ws.delete_path("x.txt")
    s = ws.snapshot_text_files()
    assert "x.txt" not in s
