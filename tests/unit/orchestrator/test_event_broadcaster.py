"""AgentEventBroadcaster + append_standard_event fan-out."""

import json
from pathlib import Path

import pytest

from mini_devin.orchestrator.event_broadcaster import AgentEventBroadcaster
from mini_devin.orchestrator.standard_events import AgentEventKind, AgentStreamEvent, append_standard_event


def test_broadcaster_publish_delivers_json():
    b = AgentEventBroadcaster(queue_maxsize=64)
    q = b.subscribe("sid-1")
    row = {"ts": "2026-01-01T00:00:00Z", "kind": "status", "text": "hi"}
    b.publish("sid-1", row)
    line = q.get_nowait()
    assert json.loads(line) == row
    b.unsubscribe("sid-1", q)


def test_append_standard_event_broadcasts_with_session_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    b = AgentEventBroadcaster(queue_maxsize=64)
    q = b.subscribe("sess-x")

    monkeypatch.setattr(
        "mini_devin.orchestrator.event_broadcaster.get_agent_event_broadcaster",
        lambda: b,
    )

    wd = tmp_path / "ws"
    wd.mkdir()
    ev = AgentStreamEvent(kind=AgentEventKind.STATUS, role="agent", text="ping", legacy_type="think")
    out = append_standard_event(wd, ev, session_id="sess-x")
    assert out is not None
    assert out["kind"] == "status"
    assert "ts" in out

    line = q.get_nowait()
    assert json.loads(line)["text"] == "ping"
