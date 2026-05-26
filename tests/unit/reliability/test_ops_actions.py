from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mini_devin.reliability.ops_actions import (
    FileOperatorActionIntake,
    OperatorActionPolicyConfig,
    OperatorActionRequest,
    OperatorActionStoreConfig,
    OperatorActionStatus,
    generate_action_id,
)


def _intake(tmp_path: Path) -> FileOperatorActionIntake:
    return FileOperatorActionIntake(
        events_file=tmp_path / "actions.jsonl",
        state_file=tmp_path / "state.json",
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
        store=OperatorActionStoreConfig(retention_hours=24, max_events=10_000),
    )


def test_generate_action_id_unique() -> None:
    now = datetime.now(timezone.utc)
    a = generate_action_id(now)
    b = generate_action_id(now)
    assert a != b
    assert a.startswith("act-")


def test_policy_rejects_non_dry_run(tmp_path: Path) -> None:
    intake = _intake(tmp_path)
    payload = OperatorActionRequest(
        action_type="pause_runtime",
        target="runtime.main",
        operator_id="operator.test",
        reason="Need to pause runtime for investigation",
        confirmation_token="APPROVE",
        dry_run=False,
    )
    response = intake.intake(payload)
    assert response.accepted is False
    assert response.status == OperatorActionStatus.POLICY_REJECTED
    assert any(item.get("code") == "policy.dry_run_required" for item in response.policy.get("issues", []))


def test_intake_accepts_and_persists_action(tmp_path: Path) -> None:
    intake = _intake(tmp_path)
    payload = OperatorActionRequest(
        action_type="acknowledge_incident",
        target="incident.current",
        operator_id="operator.test",
        reason="Incident acknowledged and under active review",
        confirmation_token="APPROVE",
        dry_run=True,
    )
    response = intake.intake(payload)
    assert response.accepted is True
    assert response.status == OperatorActionStatus.DRY_RUN_COMPLETED
    assert response.runtime_hook.get("ready") is False

    rows = intake.list_actions(hours=2, limit=10)
    assert len(rows) == 1
    assert rows[0]["action_id"] == response.action_id

    found = intake.get_action(response.action_id)
    assert found is not None
    assert found["decision"] == "accepted"
