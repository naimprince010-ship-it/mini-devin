import json
from pathlib import Path

from plodder.sandbox.stateful_shell_tracker import StatefulShellTracker


def test_session_state_json_schema_v1(tmp_path: Path) -> None:
    t = StatefulShellTracker(tmp_path)
    t.ingest_user_command("cd sub && export X=1")
    p = tmp_path / ".plodder" / "shell" / "session_state.json"
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("version") == 1
    assert "session_id" in data and len(str(data["session_id"])) > 8
    assert data.get("cwd") == "sub"
    assert data.get("exports", {}).get("X") == "1"
    assert isinstance(data.get("history"), list)
    assert len(data["history"]) >= 1


def test_wrap_preserves_then_ingests_cwd(tmp_path: Path) -> None:
    t = StatefulShellTracker(tmp_path)
    argv = ["sh", "-c", "cd sub && echo hi"]
    wrapped = t.wrap_argv(list(argv))
    assert "cd" in wrapped[2]
    assert "/workspace" in wrapped[2]
    t.ingest_user_command("cd sub && echo hi")
    assert "sub" in t.cwd.replace("\\", "/")


def test_export_persisted(tmp_path: Path) -> None:
    t = StatefulShellTracker(tmp_path)
    t.ingest_user_command("export FOO=bar")
    w = t.wrap_argv(["sh", "-c", "echo ok"])
    assert "FOO" in w[2] or "export" in w[2]
