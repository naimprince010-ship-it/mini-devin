from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mini_devin.orchestration.checkpoint_store import JsonlCheckpointStore
from mini_devin.orchestration.task_queue import InMemoryQueueTransport, TaskQueueCoordinator


def _make_coordinator(tmp_path: Path, *, lease_seconds: int = 2, max_attempts: int = 2) -> TaskQueueCoordinator:
    return TaskQueueCoordinator(
        transport=InMemoryQueueTransport(),
        checkpoint_store=JsonlCheckpointStore(tmp_path),
        queue_name="plodder.task_queue",
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        emit_events=False,
        emit_typed_events=False,
        enable_validation=True,
    )


def test_queue_enqueue_and_dequeue(tmp_path: Path) -> None:
    coordinator = _make_coordinator(tmp_path)

    record = coordinator.enqueue_unit(
        session_id="sess-1",
        task_id="task-1",
        unit_id="step-1",
        goal="Run queued execution",
        acceptance_criteria=["criteria"],
        depends_on=[],
    )

    assert record.status == "queued"

    lease = coordinator.lease_next("worker-1", now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert lease is not None
    assert lease.message_id == record.message_id
    assert lease.unit_id == "step-1"

    checkpoint_ids = [item.checkpoint_id for item in JsonlCheckpointStore(tmp_path).list()]
    assert checkpoint_ids == [
        f"queue:{record.message_id}:queued",
        f"queue:{record.message_id}:leased",
    ]


def test_queue_heartbeat_refreshes_lease(tmp_path: Path) -> None:
    coordinator = _make_coordinator(tmp_path, lease_seconds=2)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    record = coordinator.enqueue_unit(
        session_id="sess-1",
        task_id="task-1",
        unit_id="step-1",
        goal="Run queued execution",
        acceptance_criteria=["criteria"],
        depends_on=[],
    )
    lease = coordinator.lease_next("worker-1", now=base)
    assert lease is not None

    refreshed = coordinator.heartbeat(lease.lease_id, now=base + timedelta(seconds=1))
    assert refreshed is not None
    assert refreshed.lease_expires_at is not None
    assert refreshed.lease_expires_at > base + timedelta(seconds=2)

    store = JsonlCheckpointStore(tmp_path)
    assert any(item.checkpoint_id == f"queue:{record.message_id}:heartbeat" for item in store.list())


def test_queue_recovery_requeues_expired_message(tmp_path: Path) -> None:
    coordinator = _make_coordinator(tmp_path, lease_seconds=1, max_attempts=2)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    coordinator.enqueue_unit(
        session_id="sess-1",
        task_id="task-1",
        unit_id="step-1",
        goal="Recover expired work",
        acceptance_criteria=["criteria"],
        depends_on=[],
    )
    lease = coordinator.lease_next("worker-1", now=base)
    assert lease is not None

    recovered = coordinator.recover_expired(now=base + timedelta(seconds=5))
    assert len(recovered) == 1
    assert recovered[0].status == "queued"
    assert recovered[0].attempt == 2

    next_lease = coordinator.lease_next("worker-2", now=base + timedelta(seconds=6))
    assert next_lease is not None
    assert next_lease.attempt == 2


def test_queue_recovery_dead_letters_after_max_attempts(tmp_path: Path) -> None:
    coordinator = _make_coordinator(tmp_path, lease_seconds=1, max_attempts=1)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    coordinator.enqueue_unit(
        session_id="sess-1",
        task_id="task-1",
        unit_id="step-1",
        goal="Dead letter work",
        acceptance_criteria=["criteria"],
        depends_on=[],
    )
    lease = coordinator.lease_next("worker-1", now=base)
    assert lease is not None

    recovered = coordinator.recover_expired(now=base + timedelta(seconds=5))
    assert len(recovered) == 1
    assert recovered[0].status == "dead_letter"
    assert recovered[0].dead_letter_reason is not None
