from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from mini_devin.contracts import DurableCheckpoint
from mini_devin.orchestration.checkpoint_store import JsonlCheckpointStore
from mini_devin.orchestration.runtime_contracts import emit_step_completed, emit_step_started


class DummyTypedEmitter:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, event):  # type: ignore[no-untyped-def]
        self.events.append(dict(event))


def test_step_lifecycle_dual_emission(tmp_path: Path, monkeypatch) -> None:
    for name in (
        "PLODDER_TYPED_EVENTS",
        "PLODDER_EVENT_SCHEMA_VALIDATION",
        "PLODDER_STEP_CHECKPOINTS",
        "PLODDER_TRACE_EVENT_IDS",
    ):
        monkeypatch.setenv(name, "1")

    calls: list[str] = []
    emitter = DummyTypedEmitter()
    step = SimpleNamespace(step_id="step-1", description="Apply the safe integration slice")

    async def legacy_started() -> None:
        calls.append("started")

    async def legacy_completed() -> None:
        calls.append("completed")

    started = asyncio.run(
        emit_step_started(
            workspace=tmp_path,
            session_id="sess-1",
            task_id="task-1",
            step_index=0,
            step=step,
            legacy_callback=legacy_started,
            typed_emitter=emitter,
        )
    )
    completed = asyncio.run(
        emit_step_completed(
            workspace=tmp_path,
            session_id="sess-1",
            task_id="task-1",
            step_index=0,
            step=step,
            legacy_callback=legacy_completed,
            typed_emitter=emitter,
        )
    )

    assert calls == ["started", "completed"]
    assert started is not None and started["event_type"] == "step.started"
    assert completed is not None and completed["event_type"] == "step.completed"
    assert started["trace_id"]
    assert completed["trace_id"]
    assert len(emitter.events) == 2
    assert emitter.events[0]["event_type"] == "step.started"
    assert emitter.events[1]["event_type"] == "step.completed"
    assert emitter.events[1]["checkpoint_id"] == "sess-1:task-1:step-1:checkpoint"

    typed_events_path = tmp_path / ".plodder" / "typed_events.jsonl"
    assert not typed_events_path.exists()

    checkpoint_store = JsonlCheckpointStore(tmp_path)
    checkpoints = checkpoint_store.list()
    assert len(checkpoints) == 1
    assert checkpoints[0].checkpoint_id == "sess-1:task-1:step-1:checkpoint"
    assert checkpoints[0].scope_id == "task-1"


def test_checkpoint_store_round_trip(tmp_path: Path) -> None:
    store = JsonlCheckpointStore(tmp_path)
    checkpoint = DurableCheckpoint(
        checkpoint_id="cp-1",
        scope_id="task-1",
        state={"step": "step-1", "status": "passed"},
        metadata={"trace_id": "abc123"},
    )

    store.save(checkpoint)

    loaded = store.load("cp-1")
    assert loaded is not None
    assert loaded.checkpoint_id == "cp-1"
    assert loaded.scope_id == "task-1"
    assert loaded.state["status"] == "passed"
    assert loaded.metadata["trace_id"] == "abc123"

    scoped = store.list("task-1")
    assert len(scoped) == 1

    store.delete("cp-1")
    assert store.load("cp-1") is None
