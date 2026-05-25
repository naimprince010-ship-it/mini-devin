from __future__ import annotations

import pytest

from mini_devin.contracts import EventSchemaValidationError, load_event_schema_registry, validate_event_payload


def test_event_schema_registry_loads_all_contracts() -> None:
    registry = load_event_schema_registry()

    assert registry.event_types() == (
        "sandbox.lifecycle",
        "step.completed",
        "step.started",
        "task.completed",
        "task.created",
        "task.dead_letter",
        "task.heartbeat",
        "task.leased",
        "task.queued",
        "task.requeued",
        "tool.called",
    )

    document = registry.get("task.created")
    assert document.path.name == "task.created.json"
    assert document.schema["properties"]["event_type"]["const"] == "task.created"


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        (
            "task.created",
            {
                "event_type": "task.created",
                "session_id": "sess-1",
                "task_id": "task-1",
                "created_at": "2026-01-01T00:00:00Z",
                "goal": "add a test",
                "priority": "normal",
                "metadata": {},
            },
        ),
        (
            "task.queued",
            {
                "event_type": "task.queued",
                "session_id": "sess-1",
                "task_id": "task-1",
                "unit_id": "step-1",
                "queued_at": "2026-01-01T00:00:00Z",
                "goal": "add a test",
                "queue_name": "plodder.task_queue",
                "attempt": 1,
                "metadata": {},
            },
        ),
        (
            "step.completed",
            {
                "event_type": "step.completed",
                "session_id": "sess-1",
                "task_id": "task-1",
                "step_id": "step-1",
                "completed_at": "2026-01-01T00:00:01+00:00",
                "status": "passed",
                "summary": "ok",
            },
        ),
    ],
)
def test_validate_event_payload_accepts_valid_contracts(event_type: str, payload: dict[str, object]) -> None:
    validated = validate_event_payload(payload)
    assert validated["event_type"] == event_type


def test_validate_event_payload_rejects_invalid_contracts() -> None:
    with pytest.raises(EventSchemaValidationError):
        validate_event_payload(
            {
                "event_type": "task.created",
                "session_id": "sess-1",
                "task_id": "task-1",
                "created_at": "not-a-date",
                "goal": "add a test",
            }
        )

    with pytest.raises(EventSchemaValidationError):
        validate_event_payload(
            {
                "event_type": "step.started",
                "session_id": "sess-1",
                "task_id": "task-1",
                "step_id": "step-1",
                "started_at": "2026-01-01T00:00:00Z",
                "description": "run validation",
                "extra": "not allowed",
            }
        )
