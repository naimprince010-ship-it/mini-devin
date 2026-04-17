"""SessionRecoveryTracker + post-mortem formatting."""

from mini_devin.reliability.post_mortem import PostMortemPayload, format_post_mortem
from plodder.agent.orchestrator import SessionRecoveryTracker


def test_tracker_triggers_diagnostic_after_three_identical_failures():
    tr = SessionRecoveryTracker(streak_threshold=3)
    r = {"tool": "sandbox_shell", "ok": False, "exit_code": 1, "stderr": "boom", "command": "npm test"}
    assert tr.record_tool_observation("sandbox_shell", r) is None
    assert tr.record_tool_observation("sandbox_shell", dict(r)) is None
    b = tr.record_tool_observation("sandbox_shell", dict(r))
    assert b is not None
    assert "fingerprint" in b.system_block.lower() or "error_fingerprint" in b.system_block.lower()
    assert len(tr.diagnostic_triggers) == 1
    assert tr.diagnostic_triggers[0].error_fingerprint


def test_recovery_path_recorded_after_success_following_diagnostic():
    tr = SessionRecoveryTracker(streak_threshold=2)
    bad = {"tool": "fs_read", "ok": False, "error": "missing"}
    tr.record_tool_observation("fs_read", bad)
    tr.record_tool_observation("fs_read", dict(bad))
    tr.record_tool_observation("fs_list", {"tool": "fs_list", "ok": True, "path": ".", "entries": []})
    assert len(tr.recovery_paths) == 1


def test_format_post_mortem_includes_fingerprints():
    md = format_post_mortem(
        PostMortemPayload(
            goal_or_task_id="t1",
            diagnostic_triggers=[],
            recovery_paths=[],
            failure_streaks=[],
        )
    )
    assert "t1" in md
    assert "Diagnostic triggers" in md
