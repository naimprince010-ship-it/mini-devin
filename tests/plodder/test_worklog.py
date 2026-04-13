"""PlodderWorklog action–observation stream and export."""

import json
from pathlib import Path

from plodder.agent.orchestrator import (
    OrchestratorHintBundle,
    PlodderWorklog,
    export_worklog_json,
    recovery_hint_for_failed_tool,
)


def test_worklog_records_failure_hint():
    wl = PlodderWorklog(goal="test goal", workspace_root="/tmp")
    wl.record_action_observation(
        round_idx=0,
        thought="I will list files",
        tool_name="fs_list",
        args={"path": "."},
        result={"tool": "fs_list", "ok": False, "error": "not found"},
    )
    assert len(wl.events) == 1
    assert wl.events[0]["state"] == "failure"
    assert wl.events[0]["incremental_recovery_hint"]
    assert "FAILED" in wl.summary_of_progress or "failed" in wl.summary_of_progress.lower()


def test_diagnostic_trigger_emits_stream_event():
    wl = PlodderWorklog(goal="g", workspace_root=".")
    bundle = OrchestratorHintBundle(
        system_block="diag",
        incremental_hint="hint",
        error_fingerprint="sandbox_shell|1|boom",
        streak_length=3,
    )
    wl.record_action_observation(
        round_idx=1,
        thought="retry shell",
        tool_name="sandbox_shell",
        args={"argv": ["false"]},
        result={"tool": "sandbox_shell", "ok": False, "stderr": "boom", "exit_code": 1},
        diagnostic_bundle=bundle,
    )
    types = [e["event_type"] for e in wl.events]
    assert "action_observation" in types
    assert "diagnostic_trigger" in types
    diag = next(e for e in wl.events if e["event_type"] == "diagnostic_trigger")
    assert diag["error_fingerprint"] == "sandbox_shell|1|boom"
    assert diag.get("ref_event_id")


def test_export_worklog_json(tmp_path: Path):
    wl = PlodderWorklog(goal="x", workspace_root=str(tmp_path))
    wl.record_action_observation(
        round_idx=0,
        thought="t",
        tool_name="fs_list",
        args={},
        result={"tool": "fs_list", "ok": True, "entries": []},
    )
    p = export_worklog_json(wl, tmp_path)
    assert p.name == "worklog.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == "plodder_worklog/1.0"
    assert data["goal"] == "x"
    assert len(data["events"]) >= 1


def test_recovery_hint_for_failed_tool_unknown():
    h = recovery_hint_for_failed_tool(
        "fs_read",
        {"ok": False, "error": "file not found", "tool": "fs_read"},
    )
    assert "incremental" in h.lower() or "recovery" in h.lower() or "path" in h.lower()
