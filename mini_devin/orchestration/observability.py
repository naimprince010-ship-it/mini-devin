"""Local observability, replay, and metrics scaffolding for Plodder.

This module keeps the rollout additive and file-backed. It provides a centralized
timeline record, replay helpers, lightweight metrics emission, and OpenTelemetry-
compatible trace fields without introducing heavy infra dependencies.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def observability_enabled() -> bool:
    return _flag_enabled("PLODDER_OBSERVABILITY", True)


def timeline_recording_enabled() -> bool:
    return _flag_enabled("PLODDER_TIMELINE_RECORDING")


def replay_debug_enabled() -> bool:
    return _flag_enabled("PLODDER_REPLAY_DEBUG")


def worker_metrics_enabled() -> bool:
    return _flag_enabled("PLODDER_WORKER_METRICS")


def queue_metrics_enabled() -> bool:
    return _flag_enabled("PLODDER_QUEUE_METRICS")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _repo_timeline_dir(workspace: str | Path) -> Path:
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    timeline_dir = root / ".plodder"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    return timeline_dir


def _timeline_path(workspace: str | Path) -> Path:
    return _repo_timeline_dir(workspace) / "timeline.jsonl"


def _metrics_path(workspace: str | Path) -> Path:
    return _repo_timeline_dir(workspace) / "worker_metrics.jsonl"


def _trace_hex(seed: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    return uuid.uuid5(namespace, seed).hex


def _trace_span_hex(seed: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_OID, seed).hex[:16]


def build_correlation_context(
    *,
    session_id: str,
    task_id: str,
    unit_id: str,
    event_type: str,
    parent_trace_id: str | None = None,
) -> dict[str, str]:
    seed = f"plodder:{session_id}:{task_id}:{unit_id}:{event_type}"
    trace_id = _trace_hex(seed)
    span_id = _trace_span_hex(seed)
    correlation_id = _trace_hex(f"correlation:{seed}", uuid.NAMESPACE_DNS)
    out = {
        "correlation_id": correlation_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "traceparent": format_traceparent(trace_id, span_id),
    }
    if parent_trace_id:
        out["parent_trace_id"] = parent_trace_id
    return out


def format_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    flags = "01" if sampled else "00"
    return f"00-{trace_id}-{span_id}-{flags}"


@dataclass(frozen=True, slots=True)
class TimelineRecord:
    event_type: str
    source: str
    ts: datetime = field(default_factory=_utcnow)
    session_id: str | None = None
    task_id: str | None = None
    unit_id: str | None = None
    status: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    traceparent: str | None = None
    parent_trace_id: str | None = None
    sequence: int = 0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ts": self.ts.isoformat(),
            "event_type": self.event_type,
            "source": self.source,
            "sequence": self.sequence,
            "payload": dict(self.payload),
        }
        for key in (
            "session_id",
            "task_id",
            "unit_id",
            "status",
            "correlation_id",
            "trace_id",
            "span_id",
            "traceparent",
            "parent_trace_id",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True, slots=True)
class WorkerMetricRecord:
    metric_name: str
    value: float
    ts: datetime = field(default_factory=_utcnow)
    labels: dict[str, str] = field(default_factory=dict)
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    traceparent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ts": self.ts.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "labels": dict(self.labels),
        }
        if self.correlation_id:
            data["correlation_id"] = self.correlation_id
        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id
        if self.traceparent:
            data["traceparent"] = self.traceparent
        return data


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    index: int
    record: TimelineRecord


@dataclass(frozen=True, slots=True)
class TaskTimelinePoint:
    event_type: str
    ts: datetime
    summary: str
    status: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class TaskTimelineSnapshot:
    task_id: str
    session_id: str | None
    points: tuple[TaskTimelinePoint, ...]

    @property
    def event_types(self) -> tuple[str, ...]:
        return tuple(point.event_type for point in self.points)


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return _utcnow()
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        return []
    try:
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append((index, row))
    return rows


def record_timeline_event(workspace: str | Path, record: TimelineRecord) -> TimelineRecord:
    if not observability_enabled() or not timeline_recording_enabled():
        return record
    path = _timeline_path(workspace)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), default=str, ensure_ascii=False) + "\n")
    return record


def load_timeline_records(workspace: str | Path, *, max_lines: int = 4000) -> list[TimelineRecord]:
    rows = _load_jsonl(_timeline_path(workspace))[-max_lines:]
    records: list[TimelineRecord] = []
    for index, row in rows:
        records.append(
            TimelineRecord(
                event_type=str(row.get("event_type", "unknown")),
                source=str(row.get("source", "unknown")),
                ts=_parse_datetime(row.get("ts")),
                session_id=row.get("session_id"),
                task_id=row.get("task_id"),
                unit_id=row.get("unit_id"),
                status=row.get("status"),
                correlation_id=row.get("correlation_id"),
                trace_id=row.get("trace_id"),
                span_id=row.get("span_id"),
                traceparent=row.get("traceparent"),
                parent_trace_id=row.get("parent_trace_id"),
                sequence=int(row.get("sequence", index)),
                payload=dict(row.get("payload") or {}),
            )
        )
    return records


def replay_timeline_records(workspace: str | Path, *, max_lines: int = 4000) -> list[ReplayEntry]:
    ordered = sorted(
        enumerate(load_timeline_records(workspace, max_lines=max_lines)),
        key=lambda item: (item[1].ts, item[1].sequence, item[0]),
    )
    return [ReplayEntry(index=index, record=record) for index, record in ordered]


def reconstruct_task_timeline(workspace: str | Path, task_id: str, *, max_lines: int = 4000) -> TaskTimelineSnapshot:
    records = [entry.record for entry in replay_timeline_records(workspace, max_lines=max_lines) if entry.record.task_id == task_id]
    points = tuple(
        TaskTimelinePoint(
            event_type=record.event_type,
            ts=record.ts,
            summary=str(record.payload.get("summary") or record.payload.get("goal") or record.payload.get("reason") or record.event_type),
            status=record.status,
            trace_id=record.trace_id,
            span_id=record.span_id,
            correlation_id=record.correlation_id,
        )
        for record in records
    )
    session_id = records[0].session_id if records else None
    return TaskTimelineSnapshot(task_id=task_id, session_id=session_id, points=points)


def debug_replay_task_timeline(workspace: str | Path, task_id: str, *, max_lines: int = 4000) -> str:
    snapshot = reconstruct_task_timeline(workspace, task_id, max_lines=max_lines)
    lines = [f"task_id: {snapshot.task_id}"]
    if snapshot.session_id:
        lines.append(f"session_id: {snapshot.session_id}")
    for point in snapshot.points:
        lines.append(f"- {point.ts.isoformat()} {point.event_type} [{point.status or 'n/a'}] {point.summary}")
    return "\n".join(lines)


def emit_worker_metric(
    workspace: str | Path,
    metric_name: str,
    value: float,
    *,
    labels: Mapping[str, str] | None = None,
    correlation_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    traceparent: str | None = None,
) -> WorkerMetricRecord:
    record = WorkerMetricRecord(
        metric_name=metric_name,
        value=float(value),
        labels=dict(labels or {}),
        correlation_id=correlation_id,
        trace_id=trace_id,
        span_id=span_id,
        traceparent=traceparent,
    )
    if not observability_enabled() or not worker_metrics_enabled():
        return record
    path = _metrics_path(workspace)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), default=str, ensure_ascii=False) + "\n")
    return record


def load_worker_metrics(workspace: str | Path, *, max_lines: int = 4000) -> list[WorkerMetricRecord]:
    rows = _load_jsonl(_metrics_path(workspace))[-max_lines:]
    records: list[WorkerMetricRecord] = []
    for _, row in rows:
        records.append(
            WorkerMetricRecord(
                metric_name=str(row.get("metric_name", "")),
                value=float(row.get("value", 0.0)),
                ts=_parse_datetime(row.get("ts")),
                labels={str(k): str(v) for k, v in dict(row.get("labels") or {}).items()},
                correlation_id=row.get("correlation_id"),
                trace_id=row.get("trace_id"),
                span_id=row.get("span_id"),
                traceparent=row.get("traceparent"),
            )
        )
    return records


def summarize_worker_metrics(workspace: str | Path, *, max_lines: int = 4000) -> dict[str, Any]:
    records = load_worker_metrics(workspace, max_lines=max_lines)
    counts: dict[str, int] = {}
    totals: dict[str, float] = {}
    for record in records:
        counts[record.metric_name] = counts.get(record.metric_name, 0) + 1
        totals[record.metric_name] = totals.get(record.metric_name, 0.0) + record.value
    return {
        "count": len(records),
        "metric_counts": counts,
        "metric_totals": totals,
    }


def queue_lag_seconds(queued_at: datetime, *, now: datetime | None = None) -> float:
    current = now or _utcnow()
    return max(0.0, (current - queued_at).total_seconds())


def lease_timeout_seconds(lease_expires_at: datetime | None, *, now: datetime | None = None) -> float:
    if lease_expires_at is None:
        return 0.0
    current = now or _utcnow()
    return max(0.0, (current - lease_expires_at).total_seconds())


def build_queue_timeline_record(
    *,
    event_type: str,
    source: str,
    session_id: str,
    task_id: str,
    unit_id: str,
    payload: dict[str, Any],
    status: str | None = None,
    correlation_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    traceparent: str | None = None,
    parent_trace_id: str | None = None,
    sequence: int = 0,
    ts: datetime | None = None,
) -> TimelineRecord:
    return TimelineRecord(
        event_type=event_type,
        source=source,
        ts=ts or _utcnow(),
        session_id=session_id,
        task_id=task_id,
        unit_id=unit_id,
        status=status,
        correlation_id=correlation_id,
        trace_id=trace_id,
        span_id=span_id,
        traceparent=traceparent,
        parent_trace_id=parent_trace_id,
        sequence=sequence,
        payload=dict(payload),
    )
