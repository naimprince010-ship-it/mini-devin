from pathlib import Path

from plodder.workspace.atomic_editor import atomic_edit
from plodder.workspace.session_workspace import SessionWorkspace


def test_atomic_str_replace(tmp_path: Path) -> None:
    ws = SessionWorkspace(tmp_path)
    ws.write_file("a.txt", "hello OLD world")
    r = atomic_edit(ws, "a.txt", mode="str_replace", old_string="OLD", new_string="NEW")
    assert r["ok"]
    assert "NEW" in ws.read_file("a.txt")
