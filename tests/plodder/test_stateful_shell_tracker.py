from pathlib import Path

from plodder.sandbox.stateful_shell_tracker import StatefulShellTracker


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
