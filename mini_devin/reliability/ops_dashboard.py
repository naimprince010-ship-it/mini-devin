from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from .ops_telemetry import FileOpsTelemetryCollector, aggregate_telemetry, calculate_operational_score


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    candidate = str(raw).strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(candidate)
    except Exception:
        return None


def clamp_hours(hours: int) -> int:
    return max(1, min(int(hours), 24 * 30))


def clamp_page(page: int) -> int:
    return max(1, int(page))


def clamp_page_size(page_size: int) -> int:
    return max(1, min(int(page_size), 500))


@dataclass(frozen=True, slots=True)
class TimeWindow:
    hours: int
    start: datetime
    end: datetime


class DashboardWindowModel(BaseModel):
    hours: int = Field(ge=1, le=24 * 30)
    start_time: str
    end_time: str


class DashboardPaginationModel(BaseModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=1)


class RuntimeHealthTimelineItem(BaseModel):
    time: str
    status: str
    readiness: bool
    degraded: bool
    queue_requested: str | None = None
    queue_active: str | None = None


class DeploymentTimelineItem(BaseModel):
    time: str
    phase: str
    tags: dict[str, Any] = Field(default_factory=dict)


class QueueDegradationTimelineItem(BaseModel):
    start_time: str
    end_time: str | None = None
    duration_seconds: float = Field(ge=0)
    queue_requested: str | None = None
    queue_active: str | None = None


class IncidentLifecycleTimelineItem(BaseModel):
    time: str
    source: str
    event: str
    incident_open_count: int | None = None
    crash_loop_failures: int | None = None


class ScoreHistoryItem(BaseModel):
    time: str
    value: float = Field(ge=0, le=100)
    band: str
    components: dict[str, float] = Field(default_factory=dict)


class WarningFrequencyTrendItem(BaseModel):
    bucket_start: str
    warning_count: int = Field(ge=0)
    warning_frequency_per_min: float = Field(ge=0)


class RestartLoopTrendItem(BaseModel):
    bucket_start: str
    crash_loop_active: bool
    crash_loop_failures: int = Field(ge=0)


class DashboardTimelineEvent(BaseModel):
    kind: str
    time: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DashboardSummaryResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    kpis: dict[str, Any] = Field(default_factory=dict)
    score: dict[str, Any] = Field(default_factory=dict)
    timeline_counts: dict[str, int] = Field(default_factory=dict)


class DashboardTimelineResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    pagination: DashboardPaginationModel
    items: list[DashboardTimelineEvent] = Field(default_factory=list)


class DashboardRuntimeTimelineResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    pagination: DashboardPaginationModel
    items: list[RuntimeHealthTimelineItem] = Field(default_factory=list)


class DashboardDeploymentTimelineResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    pagination: DashboardPaginationModel
    items: list[DeploymentTimelineItem] = Field(default_factory=list)


class DashboardQueueTimelineResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    pagination: DashboardPaginationModel
    items: list[QueueDegradationTimelineItem] = Field(default_factory=list)


class DashboardIncidentTimelineResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    pagination: DashboardPaginationModel
    items: list[IncidentLifecycleTimelineItem] = Field(default_factory=list)


class DashboardScoreHistoryResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    pagination: DashboardPaginationModel
    items: list[ScoreHistoryItem] = Field(default_factory=list)


class DashboardWarningTrendResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    pagination: DashboardPaginationModel
    items: list[WarningFrequencyTrendItem] = Field(default_factory=list)


class DashboardRestartTrendResponse(BaseModel):
    schema_name: str = Field(alias="schema")
    generated_at: str
    window: DashboardWindowModel
    pagination: DashboardPaginationModel
    items: list[RestartLoopTrendItem] = Field(default_factory=list)


def resolve_time_window(*, hours: int, start_time: str | None, end_time: str | None, now: datetime | None = None) -> TimeWindow:
    current = now or _utcnow()
    bounded_hours = clamp_hours(hours)
    start = _parse_iso(start_time) if start_time else None
    end = _parse_iso(end_time) if end_time else None
    if end is None:
        end = current
    if start is None:
        start = end - timedelta(hours=bounded_hours)
    if start > end:
        start, end = end, start
    duration_hours = max(1, int(math.ceil((end - start).total_seconds() / 3600.0)))
    return TimeWindow(hours=max(bounded_hours, duration_hours), start=start, end=end)


def read_window_rows(
    collector: FileOpsTelemetryCollector,
    *,
    hours: int,
    start_time: str | None = None,
    end_time: str | None = None,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], TimeWindow]:
    window = resolve_time_window(hours=hours, start_time=start_time, end_time=end_time, now=now)
    rows = collector._read_recent_events(hours=window.hours, now=window.end)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = _parse_iso(row.get("time"))
        if ts is None:
            continue
        if ts < window.start or ts > window.end:
            continue
        filtered.append(row)
    filtered.sort(key=lambda item: str(item.get("time") or ""))
    return filtered, window


def paginate_items(items: list[Any], *, page: int, page_size: int) -> tuple[list[Any], DashboardPaginationModel]:
    bounded_page = clamp_page(page)
    bounded_page_size = clamp_page_size(page_size)
    total = len(items)
    total_pages = max(1, int(math.ceil(total / float(bounded_page_size))))
    start = (bounded_page - 1) * bounded_page_size
    end = start + bounded_page_size
    sliced = items[start:end] if start < total else []
    return sliced, DashboardPaginationModel(
        page=bounded_page,
        page_size=bounded_page_size,
        total_items=total,
        total_pages=total_pages,
    )


def serialize_runtime_health_timeline(rows: list[dict[str, Any]]) -> list[RuntimeHealthTimelineItem]:
    points: list[RuntimeHealthTimelineItem] = []
    for row in rows:
        if str(row.get("event_type") or "") != "runtime.snapshot":
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        points.append(
            RuntimeHealthTimelineItem(
                time=str(row.get("time") or ""),
                status=str(metrics.get("status") or "unknown"),
                readiness=bool(metrics.get("readiness", False)),
                degraded=bool(metrics.get("degraded", False)),
                queue_requested=metrics.get("queue_requested"),
                queue_active=metrics.get("queue_active"),
            )
        )
    return points


def serialize_deployment_event_timeline(rows: list[dict[str, Any]]) -> list[DeploymentTimelineItem]:
    points: list[DeploymentTimelineItem] = []
    for row in rows:
        if str(row.get("event_type") or "") != "deployment.event":
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        tags = row.get("tags") if isinstance(row.get("tags"), dict) else {}
        points.append(
            DeploymentTimelineItem(
                time=str(row.get("time") or ""),
                phase=str(metrics.get("phase") or "unknown"),
                tags=dict(tags),
            )
        )
    return points


def serialize_queue_degradation_timeline(rows: list[dict[str, Any]]) -> list[QueueDegradationTimelineItem]:
    points: list[QueueDegradationTimelineItem] = []
    active_start: datetime | None = None
    active_requested: str | None = None
    active_backend: str | None = None
    last_ts: datetime | None = None

    for row in rows:
        if str(row.get("event_type") or "") != "runtime.snapshot":
            continue
        ts = _parse_iso(str(row.get("time") or ""))
        if ts is None:
            continue
        last_ts = ts
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        queue_degraded = bool(metrics.get("queue_degraded", False))
        requested = metrics.get("queue_requested")
        backend = metrics.get("queue_active")

        if queue_degraded and active_start is None:
            active_start = ts
            active_requested = requested
            active_backend = backend
            continue

        if not queue_degraded and active_start is not None:
            points.append(
                QueueDegradationTimelineItem(
                    start_time=active_start.isoformat(),
                    end_time=ts.isoformat(),
                    duration_seconds=max(0.0, (ts - active_start).total_seconds()),
                    queue_requested=active_requested,
                    queue_active=active_backend,
                )
            )
            active_start = None
            active_requested = None
            active_backend = None

    if active_start is not None and last_ts is not None:
        points.append(
            QueueDegradationTimelineItem(
                start_time=active_start.isoformat(),
                end_time=None,
                duration_seconds=max(0.0, (last_ts - active_start).total_seconds()),
                queue_requested=active_requested,
                queue_active=active_backend,
            )
        )

    return points


def serialize_incident_lifecycle_timeline(rows: list[dict[str, Any]]) -> list[IncidentLifecycleTimelineItem]:
    points: list[IncidentLifecycleTimelineItem] = []
    previous_open_count = 0
    previous_crash_loop = False

    for row in rows:
        if str(row.get("event_type") or "") != "runtime.snapshot":
            continue
        ts = str(row.get("time") or "")
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        open_count = int(metrics.get("incident_open_count") or 0)
        crash_active = bool(metrics.get("crash_loop_active", False))
        crash_failures = int(metrics.get("crash_loop_failures") or 0)

        if open_count > previous_open_count:
            points.append(
                IncidentLifecycleTimelineItem(
                    time=ts,
                    source="incidents",
                    event="opened",
                    incident_open_count=open_count,
                )
            )
        elif open_count < previous_open_count:
            points.append(
                IncidentLifecycleTimelineItem(
                    time=ts,
                    source="incidents",
                    event="resolved",
                    incident_open_count=open_count,
                )
            )

        if crash_active and not previous_crash_loop:
            points.append(
                IncidentLifecycleTimelineItem(
                    time=ts,
                    source="crash_loop",
                    event="opened",
                    crash_loop_failures=crash_failures,
                )
            )
        elif not crash_active and previous_crash_loop:
            points.append(
                IncidentLifecycleTimelineItem(
                    time=ts,
                    source="crash_loop",
                    event="resolved",
                    crash_loop_failures=crash_failures,
                )
            )

        previous_open_count = open_count
        previous_crash_loop = crash_active

    return points


def serialize_score_history(rows: list[dict[str, Any]]) -> list[ScoreHistoryItem]:
    history: list[ScoreHistoryItem] = []
    running_rows: list[dict[str, Any]] = []

    for row in rows:
        running_rows.append(row)
        if str(row.get("event_type") or "") != "runtime.snapshot":
            continue

        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        aggregate = aggregate_telemetry(running_rows)
        score = calculate_operational_score(
            {
                "readiness_success_ratio": aggregate.get("readiness_success_ratio", 1.0),
                "warning_frequency_per_min": aggregate.get("warning_frequency_per_min", 0.0),
                "queue_degraded_dwell_seconds": float(metrics.get("queue_degraded_dwell_seconds") or 0.0),
                "readiness_converged_seconds": metrics.get("readiness_converged_seconds"),
            }
        )
        history.append(
            ScoreHistoryItem(
                time=str(row.get("time") or ""),
                value=float(score.get("value") or 0.0),
                band=str(score.get("band") or "risk"),
                components={
                    str(key): float(value)
                    for key, value in (score.get("components") or {}).items()
                },
            )
        )

    return history


def serialize_warning_frequency_trend(rows: list[dict[str, Any]], *, resolution_seconds: int = 300) -> list[WarningFrequencyTrendItem]:
    resolution = max(60, int(resolution_seconds))
    buckets: dict[int, int] = {}

    for row in rows:
        event_type = str(row.get("event_type") or "")
        level = str(row.get("level") or "").lower()
        if event_type != "runtime.warning" and level not in {"warning", "error"}:
            continue
        ts = _parse_iso(str(row.get("time") or ""))
        if ts is None:
            continue
        bucket = int(ts.timestamp() // resolution)
        buckets[bucket] = buckets.get(bucket, 0) + 1

    trend: list[WarningFrequencyTrendItem] = []
    for bucket, count in sorted(buckets.items()):
        start = datetime.fromtimestamp(bucket * resolution, tz=timezone.utc)
        freq = float(count) / max(1.0, float(resolution) / 60.0)
        trend.append(
            WarningFrequencyTrendItem(
                bucket_start=start.isoformat(),
                warning_count=count,
                warning_frequency_per_min=freq,
            )
        )

    return trend


def serialize_restart_loop_trend(rows: list[dict[str, Any]], *, resolution_seconds: int = 300) -> list[RestartLoopTrendItem]:
    resolution = max(60, int(resolution_seconds))
    buckets: dict[int, dict[str, Any]] = {}

    for row in rows:
        if str(row.get("event_type") or "") != "runtime.snapshot":
            continue
        ts = _parse_iso(str(row.get("time") or ""))
        if ts is None:
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        bucket = int(ts.timestamp() // resolution)
        entry = buckets.setdefault(bucket, {"active": False, "failures": 0})
        entry["active"] = bool(entry["active"] or bool(metrics.get("crash_loop_active", False)))
        entry["failures"] = max(int(entry["failures"]), int(metrics.get("crash_loop_failures") or 0))

    trend: list[RestartLoopTrendItem] = []
    for bucket, payload in sorted(buckets.items()):
        start = datetime.fromtimestamp(bucket * resolution, tz=timezone.utc)
        trend.append(
            RestartLoopTrendItem(
                bucket_start=start.isoformat(),
                crash_loop_active=bool(payload.get("active", False)),
                crash_loop_failures=int(payload.get("failures") or 0),
            )
        )
    return trend


def build_dashboard_summary(*, export_payload: dict[str, Any], rows: list[dict[str, Any]], window: TimeWindow, generated_at: datetime) -> DashboardSummaryResponse:
    runtime_points = serialize_runtime_health_timeline(rows)
    deploy_points = serialize_deployment_event_timeline(rows)
    queue_points = serialize_queue_degradation_timeline(rows)
    incident_points = serialize_incident_lifecycle_timeline(rows)
    return DashboardSummaryResponse(
        schema="ops.dashboard.v1",
        generated_at=generated_at.isoformat(),
        window=DashboardWindowModel(
            hours=window.hours,
            start_time=window.start.isoformat(),
            end_time=window.end.isoformat(),
        ),
        kpis=dict(export_payload.get("kpis") or {}),
        score=dict(export_payload.get("score") or {}),
        timeline_counts={
            "runtime_health": len(runtime_points),
            "deployment_events": len(deploy_points),
            "queue_degradation": len(queue_points),
            "incident_lifecycle": len(incident_points),
        },
    )
