from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mini_devin.orchestration.observability import (
    build_correlation_context,
    build_queue_timeline_record,
    debug_replay_task_timeline,
    emit_worker_metric,
    lease_timeout_seconds,
    load_worker_metrics,
    queue_lag_seconds,
    record_timeline_event,
    reconstruct_task_timeline,
    replay_timeline_records,
    summarize_worker_metrics,
)


def test_trace_propagation_builds_consistent_traceparent() -> None:
    context = build_correlation_context(
        session_id="sess-1",
        task_id="task-1",
        unit_id="step-1",
        event_type="task.queued",
    )

    assert context["correlation_id"]
    assert len(context["trace_id"]) == 32
    assert len(context["span_id"]) == 16
    assert context["traceparent"] == f"00-{context['trace_id']}-{context['span_id']}-01"


def test_replay_orders_timeline_by_timestamp(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_OBSERVABILITY", "1")
    monkeypatch.setenv("PLODDER_TIMELINE_RECORDING", "1")

    later = build_queue_timeline_record(
        event_type="task.completed",
        source="queue",
        session_id="sess-1",
        task_id="task-1",
        unit_id="step-1",
        payload={"summary": "done"},
        status="completed",
        ts=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        sequence=2,
    )
    earlier = build_queue_timeline_record(
        event_type="task.queued",
        source="queue",
        session_id="sess-1",
        task_id="task-1",
        unit_id="step-1",
        payload={"summary": "queued"},
        status="queued",
        ts=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        sequence=1,
    )

    record_timeline_event(tmp_path, later)
    record_timeline_event(tmp_path, earlier)

    replayed = replay_timeline_records(tmp_path)
    assert [entry.record.event_type for entry in replayed] == ["task.queued", "task.completed"]


def test_reconstruct_task_timeline_and_debug_view(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_OBSERVABILITY", "1")
    monkeypatch.setenv("PLODDER_TIMELINE_RECORDING", "1")

    record_timeline_event(
        tmp_path,
        build_queue_timeline_record(
            event_type="task.queued",
            source="queue",
            session_id="sess-1",
            task_id="task-1",
            unit_id="step-1",
            payload={"goal": "run work"},
            status="queued",
            trace_id="trace-1",
            span_id="span-1",
            correlation_id="corr-1",
        ),
    )
    record_timeline_event(
        tmp_path,
        build_queue_timeline_record(
            event_type="task.completed",
            source="queue",
            session_id="sess-1",
            task_id="task-1",
            unit_id="step-1",
            payload={"summary": "finished"},
            status="completed",
            trace_id="trace-1",
            span_id="span-2",
            correlation_id="corr-1",
        ),
    )

    snapshot = reconstruct_task_timeline(tmp_path, "task-1")
    assert snapshot.session_id == "sess-1"
    assert snapshot.event_types == ("task.queued", "task.completed")
    assert snapshot.points[0].summary == "run work"
    assert snapshot.points[1].summary == "finished"

    debug_view = debug_replay_task_timeline(tmp_path, "task-1")
    assert "task_id: task-1" in debug_view
    assert "task.completed" in debug_view


def test_worker_metric_emission_and_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_OBSERVABILITY", "1")
    monkeypatch.setenv("PLODDER_WORKER_METRICS", "1")

    emit_worker_metric(
        tmp_path,
        "worker.task.completed",
        1.0,
        labels={"task_id": "task-1", "unit_id": "step-1"},
        correlation_id="corr-1",
        trace_id="trace-1",
        span_id="span-1",
    )
    emit_worker_metric(
        tmp_path,
        "worker.task.completed",
        2.0,
        labels={"task_id": "task-2", "unit_id": "step-2"},
        correlation_id="corr-2",
        trace_id="trace-2",
        span_id="span-2",
    )

    metrics = load_worker_metrics(tmp_path)
    summary = summarize_worker_metrics(tmp_path)

    assert len(metrics) == 2
    assert metrics[0].metric_name == "worker.task.completed"
    assert summary["count"] == 2
    assert summary["metric_counts"]["worker.task.completed"] == 2
    assert summary["metric_totals"]["worker.task.completed"] == 3.0


def test_queue_lag_and_lease_timeout_calculations() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    queued_at = now - timedelta(seconds=12)
    lease_expires_at = now - timedelta(seconds=4)

    assert queue_lag_seconds(queued_at, now=now) == 12.0
    assert lease_timeout_seconds(lease_expires_at, now=now) == 4.0
    assert lease_timeout_seconds(None, now=now) == 0.0
