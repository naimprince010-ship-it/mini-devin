"""Additive runtime-contract helpers for step lifecycle integration."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from mini_devin.contracts import validate_event_payload
from mini_devin.contracts.protocols import DurableCheckpoint, TypedEventEmitter

from .checkpoint_store import JsonlCheckpointStore, load_checkpoint_store
from .observability import build_correlation_context, build_queue_timeline_record, emit_worker_metric, record_timeline_event


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def typed_events_enabled() -> bool:
    return _flag_enabled("PLODDER_TYPED_EVENTS")


def schema_validation_enabled() -> bool:
    return _flag_enabled("PLODDER_EVENT_SCHEMA_VALIDATION")


def checkpointing_enabled() -> bool:
    return _flag_enabled("PLODDER_STEP_CHECKPOINTS")


def trace_ids_enabled() -> bool:
    return _flag_enabled("PLODDER_TRACE_EVENT_IDS")


def runtime_contracts_enabled() -> bool:
    return any((typed_events_enabled(), schema_validation_enabled(), checkpointing_enabled(), trace_ids_enabled()))


def governance_telemetry_enabled() -> bool:
    """Global feature flag for observe-only governance telemetry emission."""
    return _flag_enabled("PLODDER_GOVERNANCE_TELEMETRY")


def governance_budget_signals_enabled() -> bool:
    """Feature flag for budget-related governance signals."""
    return _flag_enabled("PLODDER_GOVERNANCE_EMIT_BUDGET_SIGNALS", True)


def governance_retry_signals_enabled() -> bool:
    """Feature flag for retry-related governance signals."""
    return _flag_enabled("PLODDER_GOVERNANCE_EMIT_RETRY_SIGNALS", True)


def governance_loop_signals_enabled() -> bool:
    """Feature flag for loop-proximity governance signals."""
    return _flag_enabled("PLODDER_GOVERNANCE_EMIT_LOOP_SIGNALS", True)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _step_id_for(step_index: int, step: Any) -> str:
    explicit = getattr(step, "step_id", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return f"step-{step_index + 1}"


def _step_description(step: Any) -> str:
    if isinstance(step, str):
        return step.strip()
    description = getattr(step, "description", None)
    if isinstance(description, str) and description.strip():
        return description.strip()
    return str(step)


def build_trace_context(*, session_id: str, task_id: str, step_id: str, event_type: str) -> dict[str, str]:
    return build_correlation_context(
        session_id=session_id,
        task_id=task_id,
        unit_id=step_id,
        event_type=event_type,
    )


def _build_step_payload(
    *,
    event_type: str,
    session_id: str,
    task_id: str,
    step_index: int,
    step: Any,
    status: str | None = None,
    summary: str | None = None,
    error: str | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    step_id = _step_id_for(step_index, step)
    payload: dict[str, Any] = {
        "event_type": event_type,
        "session_id": session_id,
        "task_id": task_id,
        "step_id": step_id,
        "metadata": {
            "step_index": step_index,
            "step_description": _step_description(step),
        },
    }
    if event_type == "step.started":
        payload["started_at"] = _utcnow().isoformat()
        payload["description"] = _step_description(step)
        attempt = getattr(step, "attempt", None)
        if isinstance(attempt, int) and attempt > 0:
            payload["attempt"] = attempt
        tool = getattr(step, "tool", None)
        if isinstance(tool, str) and tool.strip():
            payload["tool"] = tool.strip()
    elif event_type == "step.completed":
        payload["completed_at"] = _utcnow().isoformat()
        payload["status"] = status or "passed"
        payload["summary"] = summary or _step_description(step)
        if error:
            payload["error"] = error
        if checkpoint_id:
            payload["checkpoint_id"] = checkpoint_id
    else:
        raise ValueError(f"Unsupported step event type: {event_type}")

    if trace_ids_enabled():
        payload.update(build_trace_context(session_id=session_id, task_id=task_id, step_id=step_id, event_type=event_type))
    return payload


class FileTypedEventEmitter(TypedEventEmitter):
    """Append typed runtime events to ``.plodder/typed_events.jsonl``."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)

    def _path(self) -> Path:
        root = Path(self.workspace).resolve()
        root.mkdir(parents=True, exist_ok=True)
        log_dir = root / ".plodder"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "typed_events.jsonl"

    def emit(self, event: Mapping[str, Any]) -> None:
        path = self._path()
        row = {"ts": _utcnow().isoformat(), **dict(event)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


async def emit_step_started(
    *,
    workspace: str | Path,
    session_id: str,
    task_id: str,
    step_index: int,
    step: Any,
    legacy_callback: Callable[[], Awaitable[None]] | None = None,
    typed_emitter: TypedEventEmitter | None = None,
) -> dict[str, Any] | None:
    if legacy_callback is not None:
        await legacy_callback()

    if not runtime_contracts_enabled():
        return None

    payload = _build_step_payload(
        event_type="step.started",
        session_id=session_id,
        task_id=task_id,
        step_index=step_index,
        step=step,
    )

    if schema_validation_enabled():
        validate_event_payload(payload)

    if typed_events_enabled():
        emitter = typed_emitter or FileTypedEventEmitter(Path(workspace))
        emitter.emit(payload)

    if runtime_contracts_enabled():
        traceparent = payload.get("traceparent") if isinstance(payload.get("traceparent"), str) else None
        record_timeline_event(
            workspace,
            build_queue_timeline_record(
                event_type="step.started",
                source="orchestrator",
                session_id=session_id,
                task_id=task_id,
                unit_id=_step_id_for(step_index, step),
                payload=dict(payload),
                status="started",
                correlation_id=payload.get("correlation_id"),
                trace_id=payload.get("trace_id"),
                span_id=payload.get("span_id"),
                traceparent=traceparent,
                parent_trace_id=payload.get("parent_trace_id"),
            ),
        )
        emit_worker_metric(
            workspace,
            "worker.step.started",
            1.0,
            labels={"task_id": task_id, "step_id": _step_id_for(step_index, step)},
            correlation_id=payload.get("correlation_id"),
            trace_id=payload.get("trace_id"),
            span_id=payload.get("span_id"),
            traceparent=traceparent,
        )

    return payload


async def emit_step_completed(
    *,
    workspace: str | Path,
    session_id: str,
    task_id: str,
    step_index: int,
    step: Any,
    legacy_callback: Callable[[], Awaitable[None]] | None = None,
    typed_emitter: TypedEventEmitter | None = None,
    checkpoint_store: JsonlCheckpointStore | None = None,
) -> dict[str, Any] | None:
    if legacy_callback is not None:
        await legacy_callback()

    if not runtime_contracts_enabled():
        return None

    step_id = _step_id_for(step_index, step)
    checkpoint_id = f"{session_id}:{task_id}:{step_id}:checkpoint"
    payload = _build_step_payload(
        event_type="step.completed",
        session_id=session_id,
        task_id=task_id,
        step_index=step_index,
        step=step,
        status="passed",
        summary=_step_description(step),
        checkpoint_id=checkpoint_id if checkpointing_enabled() else None,
    )

    if schema_validation_enabled():
        validate_event_payload(payload)

    if typed_events_enabled():
        emitter = typed_emitter or FileTypedEventEmitter(Path(workspace))
        emitter.emit(payload)

    if checkpointing_enabled():
        store = checkpoint_store or load_checkpoint_store(workspace)
        checkpoint = DurableCheckpoint(
            checkpoint_id=checkpoint_id,
            scope_id=task_id,
            state={
                "session_id": session_id,
                "task_id": task_id,
                "step_id": step_id,
                "step_index": step_index,
                "step_description": _step_description(step),
                "status": "passed",
            },
            metadata={
                "event_type": "step.completed",
                **({"trace_id": payload["trace_id"], "span_id": payload["span_id"]} if trace_ids_enabled() else {}),
            },
        )
        store.save(checkpoint)

    if runtime_contracts_enabled():
        traceparent = payload.get("traceparent") if isinstance(payload.get("traceparent"), str) else None
        record_timeline_event(
            workspace,
            build_queue_timeline_record(
                event_type="step.completed",
                source="orchestrator",
                session_id=session_id,
                task_id=task_id,
                unit_id=step_id,
                payload=dict(payload),
                status=str(payload.get("status") or "passed"),
                correlation_id=payload.get("correlation_id"),
                trace_id=payload.get("trace_id"),
                span_id=payload.get("span_id"),
                traceparent=traceparent,
                parent_trace_id=payload.get("parent_trace_id"),
            ),
        )
        emit_worker_metric(
            workspace,
            "worker.step.completed",
            1.0,
            labels={"task_id": task_id, "step_id": step_id, "status": str(payload.get("status") or "passed")},
            correlation_id=payload.get("correlation_id"),
            trace_id=payload.get("trace_id"),
            span_id=payload.get("span_id"),
            traceparent=traceparent,
        )

    return payload
