from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from mini_devin.reliability.ops_actions import (
    FileOperatorActionIntake,
    OperatorActionPolicyConfig,
    OperatorActionStoreConfig,
)

os.environ.setdefault("PLODDER_SKIP_FRONTEND_STATIC", "1")

from mini_devin.api.app import app


def _intake(tmp_path: Path) -> FileOperatorActionIntake:
    intake = FileOperatorActionIntake(
        events_file=tmp_path / "ops_actions.jsonl",
        state_file=tmp_path / "ops_actions_state.json",
        policy=OperatorActionPolicyConfig(
            enabled=True,
            enforce_dry_run=True,
            min_reason_chars=8,
            required_confirmation_token="APPROVE",
            allowed_actions={
                "acknowledge_incident",
                "mark_investigation_started",
                "pause_runtime",
                "resume_runtime",
                "retry_task",
                "replay_session",
                "quarantine_runtime",
                "jump_to_diagnostics",
            },
        ),
        store=OperatorActionStoreConfig(retention_hours=24, max_events=10000),
    )
    app.state.ops_action_intake = intake
    return intake


def test_ops_action_intake_accepts_dry_run(tmp_path: Path) -> None:
    _intake(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/ops/actions/intake",
            json={
                "action_type": "acknowledge_incident",
                "target": "incident.current",
                "operator_id": "operator.api",
                "reason": "Incident acknowledged for human-led review",
                "confirmation_token": "APPROVE",
                "dry_run": True,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.action.v1"
    assert payload["accepted"] is True
    assert payload["decision"] == "accepted"
    assert payload["status"] == "dry_run_completed"


def test_ops_action_intake_rejects_non_dry_run(tmp_path: Path) -> None:
    _intake(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/ops/actions/intake",
            json={
                "action_type": "pause_runtime",
                "target": "runtime.main",
                "operator_id": "operator.api",
                "reason": "Pause requested pending investigation notes",
                "confirmation_token": "APPROVE",
                "dry_run": False,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["decision"] == "rejected"
    assert payload["status"] == "policy_rejected"
    assert any(item["code"] == "policy.dry_run_required" for item in payload["policy"]["issues"])


def test_ops_action_timeline_and_lookup(tmp_path: Path) -> None:
    _intake(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/ops/actions/intake",
            json={
                "action_type": "mark_investigation_started",
                "target": "incident.current",
                "operator_id": "operator.api",
                "reason": "Investigation started by on-call operator",
                "confirmation_token": "APPROVE",
                "dry_run": True,
            },
        )
        action_id = created.json()["action_id"]
        timeline = client.get("/api/ops/actions/timeline", params={"hours": 2, "limit": 20})
        lookup = client.get(f"/api/ops/actions/{action_id}")

    assert timeline.status_code == 200
    timeline_payload = timeline.json()
    assert timeline_payload["schema"] == "ops.action.timeline.v1"
    assert timeline_payload["count"] >= 1
    assert timeline_payload["items"][0]["action_id"] == action_id

    assert lookup.status_code == 200
    assert lookup.json()["action_id"] == action_id


def test_dashboard_operator_action_timeline_endpoint(tmp_path: Path) -> None:
    _intake(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        _ = client.post(
            "/api/ops/actions/intake",
            json={
                "action_type": "retry_task",
                "target": "task.pending",
                "operator_id": "operator.api",
                "reason": "Queueing retry scaffold action for failed task",
                "confirmation_token": "APPROVE",
                "dry_run": True,
            },
        )
        response = client.get(
            "/api/ops/dashboard/timeline/operator-actions",
            params={"hours": 2, "page": 1, "page_size": 10},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.dashboard.v1"
    assert len(payload["items"]) >= 1
    assert payload["items"][0]["kind"] == "operator_action"
