"""Feature-flagged task queue coordinator for dual-mode execution.

This is an additive queue-backed execution slice. The default path still uses the
existing in-process orchestration loop.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from mini_devin.contracts import validate_event_payload
from mini_devin.contracts.protocols import DurableCheckpoint, DurableCheckpointStore, TypedEventEmitter

from .runtime_contracts import FileTypedEventEmitter, build_trace_context
from .observability import build_queue_timeline_record, emit_worker_metric, lease_timeout_seconds, queue_lag_seconds, record_timeline_event


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def queue_execution_enabled() -> bool:
    return _flag_enabled("PLODDER_QUEUE_EXECUTION")


def queue_events_enabled() -> bool:
    return _flag_enabled("PLODDER_QUEUE_TASK_EVENTS", True)


def queue_validation_enabled() -> bool:
    return _flag_enabled("PLODDER_QUEUE_SCHEMA_VALIDATION", True)


def queue_typed_events_enabled() -> bool:
    return _flag_enabled("PLODDER_QUEUE_TYPED_EVENTS")


def queue_leases_enabled() -> bool:
    return _flag_enabled("PLODDER_QUEUE_LEASES", True)


def queue_dead_letter_enabled() -> bool:
    return _flag_enabled("PLODDER_QUEUE_DEAD_LETTER", True)


def _parse_seconds(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def queue_lease_seconds() -> int:
    return _parse_seconds("PLODDER_QUEUE_LEASE_SECONDS", 45)


def queue_heartbeat_seconds() -> int:
    return _parse_seconds("PLODDER_QUEUE_HEARTBEAT_SECONDS", 10)


def queue_max_attempts() -> int:
    return _parse_seconds("PLODDER_QUEUE_MAX_ATTEMPTS", 3)


def queue_stream_name() -> str:
    return (os.environ.get("PLODDER_QUEUE_STREAM_NAME") or "plodder.task_queue").strip()


def queue_consumer_group() -> str:
    return (os.environ.get("PLODDER_QUEUE_CONSUMER_GROUP") or "plodder-workers").strip()


def queue_consumer_name(default: str = "worker") -> str:
    suffix = uuid.uuid4().hex[:8]
    return f"{default}-{suffix}"


def queue_backend_name() -> str:
    return (os.environ.get("PLODDER_QUEUE_BACKEND") or "memory").strip().lower()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class QueueTaskRecord:
    message_id: str
    event_type: str
    session_id: str
    task_id: str
    unit_id: str
    queued_at: datetime
    goal: str
    queue_name: str
    attempt: int = 1
    status: str = "queued"
    payload: dict[str, Any] = field(default_factory=dict)
    lease_id: str | None = None
    consumer_name: str | None = None
    lease_expires_at: datetime | None = None
    leased_at: datetime | None = None
    heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    dead_lettered_at: datetime | None = None
    dead_letter_reason: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    traceparent: str | None = None
    parent_trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueueLease:
    lease_id: str
    message_id: str
    unit_id: str
    consumer_name: str
    lease_expires_at: datetime
    attempt: int


@runtime_checkable
class TaskQueueTransport(Protocol):
    def enqueue(self, record: QueueTaskRecord) -> QueueTaskRecord:
        ...

    def lease_next(self, consumer_name: str, *, now: datetime, lease_seconds: int) -> QueueTaskRecord | None:
        ...

    def heartbeat(self, lease_id: str, *, now: datetime, lease_seconds: int) -> QueueTaskRecord | None:
        ...

    def ack(self, lease_id: str, *, now: datetime) -> QueueTaskRecord | None:
        ...

    def dead_letter(self, lease_id: str, *, now: datetime, reason: str) -> QueueTaskRecord | None:
        ...

    def recover_expired(self, *, now: datetime, max_attempts: int) -> list[QueueTaskRecord]:
        ...

    def get(self, message_id: str) -> QueueTaskRecord | None:
        ...


class InMemoryQueueTransport:
    """Deterministic queue transport used by the first queue-backed execution slice."""

    def __init__(self) -> None:
        self._records: dict[str, QueueTaskRecord] = {}
        self._queue: list[str] = []
        self._leased_by_lease_id: dict[str, str] = {}

    def enqueue(self, record: QueueTaskRecord) -> QueueTaskRecord:
        self._records[record.message_id] = record
        self._queue.append(record.message_id)
        return record

    def lease_next(self, consumer_name: str, *, now: datetime, lease_seconds: int) -> QueueTaskRecord | None:
        while self._queue:
            message_id = self._queue.pop(0)
            record = self._records.get(message_id)
            if not record or record.status != "queued":
                continue
            lease_id = f"lease-{uuid.uuid4().hex[:12]}"
            leased = replace(
                record,
                status="leased",
                consumer_name=consumer_name,
                lease_id=lease_id,
                leased_at=now,
                heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            self._records[message_id] = leased
            self._leased_by_lease_id[lease_id] = message_id
            return leased
        return None

    def heartbeat(self, lease_id: str, *, now: datetime, lease_seconds: int) -> QueueTaskRecord | None:
        message_id = self._leased_by_lease_id.get(lease_id)
        if not message_id:
            return None
        record = self._records.get(message_id)
        if not record or record.status != "leased":
            return None
        refreshed = replace(
            record,
            heartbeat_at=now,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
        )
        self._records[message_id] = refreshed
        return refreshed

    def ack(self, lease_id: str, *, now: datetime) -> QueueTaskRecord | None:
        message_id = self._leased_by_lease_id.pop(lease_id, None)
        if not message_id:
            return None
        record = self._records.get(message_id)
        if not record:
            return None
        completed = replace(record, status="completed", completed_at=now)
        self._records[message_id] = completed
        return completed

    def dead_letter(self, lease_id: str, *, now: datetime, reason: str) -> QueueTaskRecord | None:
        message_id = self._leased_by_lease_id.pop(lease_id, None)
        if not message_id:
            return None
        record = self._records.get(message_id)
        if not record:
            return None
        dead = replace(
            record,
            status="dead_letter",
            dead_lettered_at=now,
            dead_letter_reason=reason,
        )
        self._records[message_id] = dead
        return dead

    def recover_expired(self, *, now: datetime, max_attempts: int) -> list[QueueTaskRecord]:
        recovered: list[QueueTaskRecord] = []
        for message_id, record in list(self._records.items()):
            if record.status != "leased" or not record.lease_expires_at or record.lease_expires_at > now:
                continue
            lease_id = record.lease_id
            if lease_id:
                self._leased_by_lease_id.pop(lease_id, None)
            if record.attempt >= max_attempts:
                dead = replace(
                    record,
                    status="dead_letter",
                    dead_lettered_at=now,
                    dead_letter_reason="lease expired after max attempts",
                )
                self._records[message_id] = dead
                recovered.append(dead)
                continue
            requeued = replace(
                record,
                status="queued",
                attempt=record.attempt + 1,
                lease_id=None,
                consumer_name=None,
                lease_expires_at=None,
                leased_at=None,
                heartbeat_at=None,
                dead_lettered_at=None,
                dead_letter_reason=None,
            )
            self._records[message_id] = requeued
            self._queue.append(message_id)
            recovered.append(requeued)
        return recovered

    def get(self, message_id: str) -> QueueTaskRecord | None:
        return self._records.get(message_id)


class RedisStreamsQueueTransportSkeleton(InMemoryQueueTransport):
    """Placeholder Redis Streams backend for the queue slice.

    The safe behavior today is still the in-memory scaffold. This class makes the
    Redis Streams boundary explicit so it can be swapped in behind a flag later.
    """

    def __init__(self, *, stream_name: str, consumer_group: str) -> None:
        super().__init__()
        self.stream_name = stream_name
        self.consumer_group = consumer_group


def create_queue_transport() -> TaskQueueTransport:
    backend = queue_backend_name()
    if backend == "redis_streams":
        return RedisStreamsQueueTransportSkeleton(
            stream_name=queue_stream_name(),
            consumer_group=queue_consumer_group(),
        )
    return InMemoryQueueTransport()


def _build_queue_event_payload(
    *,
    event_type: str,
    record: QueueTaskRecord,
    now: datetime,
    consumer_name: str | None = None,
    lease_id: str | None = None,
    lease_expires_at: datetime | None = None,
    reason: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_type": event_type,
        "session_id": record.session_id,
        "task_id": record.task_id,
        "unit_id": record.unit_id,
        "attempt": record.attempt,
        "trace_id": record.trace_id,
        "span_id": record.span_id,
        "traceparent": record.traceparent,
        "metadata": dict(record.metadata),
    }
    if record.parent_trace_id:
        payload["parent_trace_id"] = record.parent_trace_id
    if event_type == "task.queued":
        payload.update(
            {
                "queued_at": now.isoformat(),
                "goal": record.goal,
                "queue_name": record.queue_name,
            }
        )
    elif event_type == "task.leased":
        payload.update(
            {
                "leased_at": now.isoformat(),
                "lease_id": lease_id or record.lease_id,
                "lease_expires_at": (lease_expires_at or record.lease_expires_at or now).isoformat(),
                "consumer_name": consumer_name or record.consumer_name or "",
            }
        )
    elif event_type == "task.heartbeat":
        payload.update(
            {
                "heartbeat_at": now.isoformat(),
                "lease_id": lease_id or record.lease_id,
                "lease_expires_at": (lease_expires_at or record.lease_expires_at or now).isoformat(),
            }
        )
    elif event_type == "task.requeued":
        payload.update(
            {
                "requeued_at": now.isoformat(),
                "reason": reason or "lease expired",
            }
        )
    elif event_type == "task.completed":
        payload.update(
            {
                "completed_at": now.isoformat(),
                "status": "passed",
                "summary": summary or record.goal,
            }
        )
    elif event_type == "task.dead_letter":
        payload.update(
            {
                "dead_lettered_at": now.isoformat(),
                "reason": reason or "dead-lettered",
            }
        )
    else:
        raise ValueError(f"Unsupported task lifecycle event: {event_type}")

    payload.update(build_trace_context(session_id=record.session_id, task_id=record.task_id, step_id=record.unit_id, event_type=event_type))
    return payload


@dataclass(slots=True)
class TaskQueueCoordinator:
    transport: TaskQueueTransport
    checkpoint_store: DurableCheckpointStore | None = None
    typed_emitter: TypedEventEmitter | None = None
    workspace: str | Path | None = None
    queue_name: str = field(default_factory=queue_stream_name)
    lease_seconds: int = field(default_factory=queue_lease_seconds)
    heartbeat_seconds: int = field(default_factory=queue_heartbeat_seconds)
    max_attempts: int = field(default_factory=queue_max_attempts)
    enable_validation: bool = field(default_factory=queue_validation_enabled)
    emit_events: bool = field(default_factory=queue_events_enabled)
    emit_typed_events: bool = field(default_factory=queue_typed_events_enabled)
    enable_leases: bool = field(default_factory=queue_leases_enabled)
    enable_dead_letter: bool = field(default_factory=queue_dead_letter_enabled)

    def __post_init__(self) -> None:
        if self.typed_emitter is None and self.emit_typed_events:
            self.typed_emitter = None

    def _maybe_emit(self, payload: dict[str, Any]) -> None:
        if self.enable_validation:
            validate_event_payload(payload)
        if self.emit_typed_events:
            emitter = self.typed_emitter
            if emitter is None:
                return
            emitter.emit(payload)

    def _save_checkpoint(self, record: QueueTaskRecord, phase: str, *, extra: dict[str, Any] | None = None) -> None:
        if not self.checkpoint_store:
            return
        state = {
            "phase": phase,
            "event_type": record.event_type,
            "session_id": record.session_id,
            "task_id": record.task_id,
            "unit_id": record.unit_id,
            "attempt": record.attempt,
            "status": record.status,
        }
        if record.lease_id:
            state["lease_id"] = record.lease_id
        if record.lease_expires_at:
            state["lease_expires_at"] = record.lease_expires_at.isoformat()
        if extra:
            state.update(extra)
        self.checkpoint_store.save(
            DurableCheckpoint(
                checkpoint_id=f"queue:{record.message_id}:{phase}",
                scope_id=record.task_id,
                state=state,
                metadata={
                    "queue_name": record.queue_name,
                    "trace_id": record.trace_id,
                    "span_id": record.span_id,
                    "parent_trace_id": record.parent_trace_id,
                },
            )
        )

    def enqueue_unit(
        self,
        *,
        session_id: str,
        task_id: str,
        unit_id: str,
        goal: str,
        acceptance_criteria: list[str],
        depends_on: list[str],
        metadata: dict[str, Any] | None = None,
        parent_trace_id: str | None = None,
    ) -> QueueTaskRecord:
        now = _utcnow()
        trace_ids = build_trace_context(session_id=session_id, task_id=task_id, step_id=unit_id, event_type="task.queued")
        record = QueueTaskRecord(
            message_id=f"msg-{uuid.uuid4().hex[:12]}",
            event_type="task.queued",
            session_id=session_id,
            task_id=task_id,
            unit_id=unit_id,
            queued_at=now,
            goal=goal,
            queue_name=self.queue_name,
            attempt=1,
            payload={
                "acceptance_criteria": list(acceptance_criteria),
                "depends_on": list(depends_on),
                "metadata": dict(metadata or {}),
            },
            correlation_id=trace_ids["correlation_id"],
            trace_id=trace_ids["trace_id"],
            span_id=trace_ids["span_id"],
            traceparent=trace_ids["traceparent"],
            parent_trace_id=parent_trace_id,
            metadata=dict(metadata or {}),
        )
        self.transport.enqueue(record)
        payload = _build_queue_event_payload(event_type="task.queued", record=record, now=now)
        if self.emit_events:
            self._maybe_emit(payload)
        if self.workspace:
            record_timeline_event(
                self.workspace,
                build_queue_timeline_record(
                event_type="task.queued",
                source="queue",
                session_id=record.session_id,
                task_id=record.task_id,
                unit_id=record.unit_id,
                payload=dict(payload),
                status=record.status,
                correlation_id=record.correlation_id,
                trace_id=record.trace_id,
                span_id=record.span_id,
                traceparent=record.traceparent,
                parent_trace_id=record.parent_trace_id,
                ),
            )
            emit_worker_metric(
                self.workspace,
                "worker.task.queued",
                1.0,
                labels={"task_id": record.task_id, "unit_id": record.unit_id},
                correlation_id=record.correlation_id,
                trace_id=record.trace_id,
                span_id=record.span_id,
                traceparent=record.traceparent,
            )
        self._save_checkpoint(record, "queued")
        return record

    def lease_next(self, consumer_name: str, *, now: datetime | None = None) -> QueueLease | None:
        if not self.enable_leases:
            return None
        leased = self.transport.lease_next(consumer_name, now=now or _utcnow(), lease_seconds=self.lease_seconds)
        if not leased:
            return None
        payload = _build_queue_event_payload(
            event_type="task.leased",
            record=leased,
            now=leased.leased_at or _utcnow(),
            consumer_name=consumer_name,
            lease_id=leased.lease_id,
            lease_expires_at=leased.lease_expires_at,
        )
        if self.emit_events:
            self._maybe_emit(payload)
        if self.workspace:
            record_timeline_event(
                self.workspace,
                build_queue_timeline_record(
                event_type="task.leased",
                source="queue",
                session_id=leased.session_id,
                task_id=leased.task_id,
                unit_id=leased.unit_id,
                payload=dict(payload),
                status=leased.status,
                correlation_id=leased.correlation_id,
                trace_id=leased.trace_id,
                span_id=leased.span_id,
                traceparent=leased.traceparent,
                parent_trace_id=leased.parent_trace_id,
                ),
            )
            emit_worker_metric(
                self.workspace,
                "worker.queue.lag_seconds",
                queue_lag_seconds(leased.queued_at, now=leased.leased_at or _utcnow()),
                labels={"task_id": leased.task_id, "unit_id": leased.unit_id},
                correlation_id=leased.correlation_id,
                trace_id=leased.trace_id,
                span_id=leased.span_id,
                traceparent=leased.traceparent,
            )
        self._save_checkpoint(leased, "leased")
        return QueueLease(
            lease_id=leased.lease_id or "",
            message_id=leased.message_id,
            unit_id=leased.unit_id,
            consumer_name=consumer_name,
            lease_expires_at=leased.lease_expires_at or _utcnow(),
            attempt=leased.attempt,
        )

    def heartbeat(self, lease_id: str, *, now: datetime | None = None) -> QueueTaskRecord | None:
        renewed = self.transport.heartbeat(lease_id, now=now or _utcnow(), lease_seconds=self.lease_seconds)
        if not renewed:
            return None
        payload = _build_queue_event_payload(
            event_type="task.heartbeat",
            record=renewed,
            now=renewed.heartbeat_at or _utcnow(),
            lease_id=lease_id,
            lease_expires_at=renewed.lease_expires_at,
        )
        if self.emit_events:
            self._maybe_emit(payload)
        if self.workspace:
            record_timeline_event(
                self.workspace,
                build_queue_timeline_record(
                event_type="task.heartbeat",
                source="queue",
                session_id=renewed.session_id,
                task_id=renewed.task_id,
                unit_id=renewed.unit_id,
                payload=dict(payload),
                status=renewed.status,
                correlation_id=renewed.correlation_id,
                trace_id=renewed.trace_id,
                span_id=renewed.span_id,
                traceparent=renewed.traceparent,
                parent_trace_id=renewed.parent_trace_id,
                ),
            )
            emit_worker_metric(
                self.workspace,
                "worker.lease.timeout_seconds",
                lease_timeout_seconds(renewed.lease_expires_at, now=renewed.heartbeat_at or _utcnow()),
                labels={"task_id": renewed.task_id, "unit_id": renewed.unit_id},
                correlation_id=renewed.correlation_id,
                trace_id=renewed.trace_id,
                span_id=renewed.span_id,
                traceparent=renewed.traceparent,
            )
        self._save_checkpoint(renewed, "heartbeat")
        return renewed

    def complete(self, lease_id: str, *, summary: str = "", now: datetime | None = None) -> QueueTaskRecord | None:
        completed = self.transport.ack(lease_id, now=now or _utcnow())
        if not completed:
            return None
        payload = _build_queue_event_payload(
            event_type="task.completed",
            record=completed,
            now=completed.completed_at or _utcnow(),
            summary=summary or completed.goal,
        )
        if self.emit_events:
            self._maybe_emit(payload)
        if self.workspace:
            record_timeline_event(
                self.workspace,
                build_queue_timeline_record(
                event_type="task.completed",
                source="queue",
                session_id=completed.session_id,
                task_id=completed.task_id,
                unit_id=completed.unit_id,
                payload=dict(payload),
                status=completed.status,
                correlation_id=completed.correlation_id,
                trace_id=completed.trace_id,
                span_id=completed.span_id,
                traceparent=completed.traceparent,
                parent_trace_id=completed.parent_trace_id,
                ),
            )
            emit_worker_metric(
                self.workspace,
                "worker.task.completed",
                1.0,
                labels={"task_id": completed.task_id, "unit_id": completed.unit_id, "status": completed.status},
                correlation_id=completed.correlation_id,
                trace_id=completed.trace_id,
                span_id=completed.span_id,
                traceparent=completed.traceparent,
            )
        self._save_checkpoint(completed, "completed", extra={"summary": summary})
        return completed

    def dead_letter(self, lease_id: str, *, reason: str, now: datetime | None = None) -> QueueTaskRecord | None:
        if not self.enable_dead_letter:
            return None
        dead = self.transport.dead_letter(lease_id, now=now or _utcnow(), reason=reason)
        if not dead:
            return None
        payload = _build_queue_event_payload(
            event_type="task.dead_letter",
            record=dead,
            now=dead.dead_lettered_at or _utcnow(),
            reason=reason,
        )
        if self.emit_events:
            self._maybe_emit(payload)
        if self.workspace:
            record_timeline_event(
                self.workspace,
                build_queue_timeline_record(
                event_type="task.dead_letter",
                source="queue",
                session_id=dead.session_id,
                task_id=dead.task_id,
                unit_id=dead.unit_id,
                payload=dict(payload),
                status=dead.status,
                correlation_id=dead.correlation_id,
                trace_id=dead.trace_id,
                span_id=dead.span_id,
                traceparent=dead.traceparent,
                parent_trace_id=dead.parent_trace_id,
                ),
            )
            emit_worker_metric(
                self.workspace,
                "worker.task.dead_letter",
                1.0,
                labels={"task_id": dead.task_id, "unit_id": dead.unit_id, "reason": reason},
                correlation_id=dead.correlation_id,
                trace_id=dead.trace_id,
                span_id=dead.span_id,
                traceparent=dead.traceparent,
            )
        self._save_checkpoint(dead, "dead_letter", extra={"reason": reason})
        return dead

    def recover_expired(self, *, now: datetime | None = None) -> list[QueueTaskRecord]:
        recovered = self.transport.recover_expired(now=now or _utcnow(), max_attempts=self.max_attempts)
        for record in recovered:
            if record.status == "queued":
                payload = _build_queue_event_payload(
                    event_type="task.requeued",
                    record=record,
                    now=record.queued_at,
                    reason="lease expired",
                )
                if self.emit_events:
                    self._maybe_emit(payload)
                if self.workspace:
                    record_timeline_event(
                        self.workspace,
                        build_queue_timeline_record(
                        event_type="task.requeued",
                        source="queue",
                        session_id=record.session_id,
                        task_id=record.task_id,
                        unit_id=record.unit_id,
                        payload=dict(payload),
                        status=record.status,
                        correlation_id=record.correlation_id,
                        trace_id=record.trace_id,
                        span_id=record.span_id,
                        traceparent=record.traceparent,
                        parent_trace_id=record.parent_trace_id,
                        ),
                    )
                    emit_worker_metric(
                        self.workspace,
                        "worker.queue.recovered",
                        1.0,
                        labels={"task_id": record.task_id, "unit_id": record.unit_id},
                        correlation_id=record.correlation_id,
                        trace_id=record.trace_id,
                        span_id=record.span_id,
                        traceparent=record.traceparent,
                    )
                self._save_checkpoint(record, "requeued")
            elif record.status == "dead_letter":
                payload = _build_queue_event_payload(
                    event_type="task.dead_letter",
                    record=record,
                    now=record.dead_lettered_at or _utcnow(),
                    reason=record.dead_letter_reason or "lease expired after max attempts",
                )
                if self.emit_events:
                    self._maybe_emit(payload)
                if self.workspace:
                    record_timeline_event(
                        self.workspace,
                        build_queue_timeline_record(
                        event_type="task.dead_letter",
                        source="queue",
                        session_id=record.session_id,
                        task_id=record.task_id,
                        unit_id=record.unit_id,
                        payload=dict(payload),
                        status=record.status,
                        correlation_id=record.correlation_id,
                        trace_id=record.trace_id,
                        span_id=record.span_id,
                        traceparent=record.traceparent,
                        parent_trace_id=record.parent_trace_id,
                        ),
                    )
                    emit_worker_metric(
                        self.workspace,
                        "worker.lease.expired",
                        1.0,
                        labels={"task_id": record.task_id, "unit_id": record.unit_id},
                        correlation_id=record.correlation_id,
                        trace_id=record.trace_id,
                        span_id=record.span_id,
                        traceparent=record.traceparent,
                    )
                self._save_checkpoint(record, "dead_letter", extra={"reason": record.dead_letter_reason})
        return recovered
