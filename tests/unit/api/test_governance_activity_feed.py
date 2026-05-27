from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

app_module = importlib.import_module("mini_devin.api.app")
from mini_devin.api.app import app


def test_activity_feed_preserves_governance_metadata(monkeypatch, tmp_path: Path) -> None:
    log_dir = tmp_path / ".plodder"
    log_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": "2026-01-01T00:00:00+00:00",
        "kind": "status",
        "type": "test_status",
        "governance_schema": "governance.telemetry.v1",
        "governance_observe_only": True,
        "governance_signals": [
            {
                "schema": "governance.telemetry.v1",
                "observe_only": True,
                "signal_type": "retry",
                "status": "active",
                "counters": {"consecutive_failures": 1},
            }
        ],
    }
    (log_dir / "session_events.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    mock_session = SimpleNamespace(working_directory=str(tmp_path))
    monkeypatch.setattr(app_module.session_manager, "get_session", AsyncMock(return_value=mock_session))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/sessions/sess-governance/activity-feed", params={"limit": 20})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    event = payload["events"][0]
    assert event["governance_schema"] == "governance.telemetry.v1"
    assert event["governance_observe_only"] is True
    assert event["governance_signals"][0]["signal_type"] == "retry"
    assert event["governance_signals"][0]["schema"] == "governance.telemetry.v1"
