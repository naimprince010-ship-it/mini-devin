"""Workspace sidecar, worklog, event timeline, terminal hints."""

from __future__ import annotations

import json
from pathlib import Path

from mini_devin.orchestrator.event_stream import EventStream, TimelineEventKind
from mini_devin.orchestrator.session_worklog import SessionWorklog, load_worklog, save_worklog
from mini_devin.orchestrator.terminal_recovery import terminal_recovery_hint
from mini_devin.orchestrator.workspace_sidecar import WorkspaceSidecar


def test_worklog_roundtrip(tmp_path: Path) -> None:
    log = SessionWorklog(
        session_id="s1",
        last_task_id="t1",
        current_plan=["a", "b"],
        finished_steps=["a"],
        current_step_idx=1,
    )
    save_worklog(tmp_path, log)
    loaded = load_worklog(tmp_path)
    assert loaded is not None
    assert loaded.current_plan == ["a", "b"]
    assert loaded.finished_steps == ["a"]
    assert loaded.current_step_idx == 1


def test_event_stream_normalize(tmp_path: Path) -> None:
    pl = tmp_path / ".plodder"
    pl.mkdir(parents=True)
    rows = [
        {"ts": "2026-01-01T00:00:00Z", "kind": "tool_call", "type": "act", "tool_name": "terminal", "tool_args": {"command": "ls"}},
        {"ts": "2026-01-01T00:00:01Z", "kind": "observation", "type": "observe", "tool_name": "terminal", "exit_code": 1, "output": "npm err"},
    ]
    (pl / "session_events.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    es = EventStream(tmp_path)
    evs = es.to_export_list(max_lines=10)
    assert evs[0]["kind"] == TimelineEventKind.ACTION.value
    assert evs[1]["kind"] == TimelineEventKind.OBSERVATION.value


def test_terminal_recovery_hint() -> None:
    h = terminal_recovery_hint(1, "npm ERR! code ELIFECYCLE", command="npm test")
    assert h and "npm" in h.lower()


def test_sidecar_snapshot(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    sc = WorkspaceSidecar(tmp_path)
    sc.start()
    try:
        text = sc.get_snapshot_text(max_lines=100, max_chars=5000)
        assert "a.txt" in text
    finally:
        sc.stop()
