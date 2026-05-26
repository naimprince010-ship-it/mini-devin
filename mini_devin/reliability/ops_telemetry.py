from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _bounded_int(raw: str | None, default: int, *, low: int, high: int) -> int:
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class OpsTelemetryConfig:
    enabled: bool = True
    snapshot_interval_seconds: int = 30
    retention_hours: int = 168
    max_events: int = 200_000


@dataclass(slots=True)
class OpsTelemetryState:
    deployment_started_at: str | None = None
    readiness_converged_seconds: float | None = None
    queue_degraded_since: str | None = None
    queue_degraded_total_seconds: float = 0.0
    baseline_rss_mb: float | None = None
    last_snapshot_at: str | None = None
    incident_started_at: dict[str, str] = field(default_factory=dict)
    incident_mttr_seconds: list[float] = field(default_factory=list)
    append_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_started_at": self.deployment_started_at,
            "readiness_converged_seconds": self.readiness_converged_seconds,
            "queue_degraded_since": self.queue_degraded_since,
            "queue_degraded_total_seconds": self.queue_degraded_total_seconds,
            "baseline_rss_mb": self.baseline_rss_mb,
            "last_snapshot_at": self.last_snapshot_at,
            "incident_started_at": dict(self.incident_started_at),
            "incident_mttr_seconds": list(self.incident_mttr_seconds),
            "append_count": self.append_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OpsTelemetryState:
        return cls(
            deployment_started_at=payload.get("deployment_started_at"),
            readiness_converged_seconds=payload.get("readiness_converged_seconds"),
            queue_degraded_since=payload.get("queue_degraded_since"),
            queue_degraded_total_seconds=float(payload.get("queue_degraded_total_seconds") or 0.0),
            baseline_rss_mb=payload.get("baseline_rss_mb"),
            last_snapshot_at=payload.get("last_snapshot_at"),
            incident_started_at=dict(payload.get("incident_started_at") or {}),
            incident_mttr_seconds=[float(v) for v in (payload.get("incident_mttr_seconds") or [])],
            append_count=int(payload.get("append_count") or 0),
        )


def telemetry_config_from_env() -> OpsTelemetryConfig:
    enabled = (os.getenv("PLODDER_OPS_TELEMETRY", "true").strip().lower() in {"1", "true", "yes", "on"})
    snapshot_interval = _bounded_int(
        os.getenv("PLODDER_OPS_TELEMETRY_SNAPSHOT_SEC"),
        default=30,
        low=5,
        high=3_600,
    )
    retention_hours = _bounded_int(
        os.getenv("PLODDER_OPS_TELEMETRY_RETENTION_HOURS"),
        default=168,
        low=1,
        high=24 * 90,
    )
    max_events = _bounded_int(
        os.getenv("PLODDER_OPS_TELEMETRY_MAX_EVENTS"),
        default=200_000,
        low=1_000,
        high=2_000_000,
    )
    return OpsTelemetryConfig(
        enabled=enabled,
        snapshot_interval_seconds=snapshot_interval,
        retention_hours=retention_hours,
        max_events=max_events,
    )


class FileOpsTelemetryCollector:
    """Local lightweight telemetry collector using JSONL events and a small state file."""

    schema_version = "ops.telemetry.v1"

    def __init__(
        self,
        *,
        events_file: Path,
        state_file: Path,
        config: OpsTelemetryConfig,
    ) -> None:
        self.events_file = events_file
        self.state_file = state_file
        self.config = config
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> OpsTelemetryState:
        if not self.state_file.exists():
            return OpsTelemetryState()
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return OpsTelemetryState.from_dict(payload)
        except Exception:
            pass
        return OpsTelemetryState()

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self.state.to_dict(), ensure_ascii=True), encoding="utf-8")

    def should_capture_snapshot(self, *, now: datetime | None = None) -> bool:
        if not self.config.enabled:
            return False
        current = now or _utcnow()
        last = _parse_iso(self.state.last_snapshot_at)
        if last is None:
            return True
        return (current - last).total_seconds() >= float(self.config.snapshot_interval_seconds)

    def _append_event(
        self,
        *,
        event_type: str,
        level: str = "info",
        metrics: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        ts = (now or _utcnow()).isoformat()
        row = {
            "schema": self.schema_version,
            "time": ts,
            "event_type": event_type,
            "level": level,
            "metrics": metrics or {},
            "tags": tags or {},
        }
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        self.state.append_count += 1
        if self.state.append_count % 25 == 0:
            self._enforce_retention(now=_parse_iso(ts) or _utcnow())
        self._save_state()

    def _read_recent_events(self, *, hours: int, now: datetime | None = None) -> list[dict[str, Any]]:
        if not self.events_file.exists():
            return []
        current = now or _utcnow()
        cutoff = current - timedelta(hours=max(1, hours))
        rows: list[dict[str, Any]] = []
        with self.events_file.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ts = _parse_iso(row.get("time"))
                if ts is None or ts < cutoff:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def _enforce_retention(self, *, now: datetime | None = None) -> None:
        if not self.events_file.exists():
            return
        current = now or _utcnow()
        cutoff = current - timedelta(hours=self.config.retention_hours)
        kept: list[str] = []
        with self.events_file.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ts = _parse_iso(row.get("time"))
                if ts is None or ts < cutoff:
                    continue
                kept.append(json.dumps(row, ensure_ascii=True))
        if len(kept) > self.config.max_events:
            kept = kept[-self.config.max_events :]
        self.events_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    def record_deployment_event(self, phase: str, *, tags: dict[str, Any] | None = None, now: datetime | None = None) -> None:
        current = now or _utcnow()
        if phase == "deploy.start":
            self.state.deployment_started_at = current.isoformat()
            self.state.readiness_converged_seconds = None
        self._append_event(
            event_type="deployment.event",
            metrics={"phase": phase},
            tags=tags,
            now=current,
        )

    def record_startup_stage(self, stage: str, *, now: datetime | None = None) -> None:
        self._append_event(
            event_type="startup.stage",
            metrics={"stage": stage},
            now=now,
        )

    def record_warning(self, code: str, message: str, *, severity: str = "warning", now: datetime | None = None) -> None:
        self._append_event(
            event_type="runtime.warning",
            level=severity,
            metrics={"code": code, "message": message},
            now=now,
        )

    def record_governance_signal(
        self,
        signal_type: str,
        status: str,
        *,
        counters: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Append observe-only governance telemetry signal for dashboard/export consumers."""
        self._append_event(
            event_type="governance.signal",
            metrics={
                "signal_type": str(signal_type),
                "status": str(status),
                "counters": dict(counters or {}),
                "observe_only": True,
                "schema": "governance.telemetry.v1",
            },
            tags=tags,
            now=now,
        )

    def record_runtime_snapshot(
        self,
        payload: dict[str, Any],
        *,
        resources: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        current = now or _utcnow()
        checks = payload.get("checks") if isinstance(payload, dict) else {}
        checks = checks if isinstance(checks, dict) else {}
        readiness = bool(checks.get("readiness", False))
        degraded = bool(checks.get("degraded", False))
        queue = checks.get("queue_backend") if isinstance(checks.get("queue_backend"), dict) else {}
        incidents = checks.get("incidents") if isinstance(checks.get("incidents"), dict) else {}
        crash_loop = incidents.get("crash_loop") if isinstance(incidents.get("crash_loop"), dict) else {}
        incident_rows = incidents.get("incidents") if isinstance(incidents.get("incidents"), list) else []

        queue_degraded = bool(queue.get("degraded", False))
        queue_since = _parse_iso(self.state.queue_degraded_since)
        if queue_degraded and queue_since is None:
            self.state.queue_degraded_since = current.isoformat()
        elif not queue_degraded and queue_since is not None:
            self.state.queue_degraded_total_seconds += max(0.0, (current - queue_since).total_seconds())
            self.state.queue_degraded_since = None

        if readiness and self.state.deployment_started_at and self.state.readiness_converged_seconds is None:
            started = _parse_iso(self.state.deployment_started_at)
            if started is not None:
                self.state.readiness_converged_seconds = max(0.0, (current - started).total_seconds())

        rss_mb = None
        if isinstance(resources, dict):
            raw = resources.get("rss_mb")
            if raw is not None:
                try:
                    rss_mb = float(raw)
                except (TypeError, ValueError):
                    rss_mb = None
        if rss_mb is not None and self.state.baseline_rss_mb is None:
            self.state.baseline_rss_mb = rss_mb

        for row in incident_rows:
            if not isinstance(row, dict):
                continue
            incident_id = str(row.get("incident_id") or "").strip()
            state = str(row.get("state") or "").strip().lower()
            updated_at = _parse_iso(str(row.get("updated_at") or "")) or current
            if not incident_id:
                continue
            if state == "open" and incident_id not in self.state.incident_started_at:
                self.state.incident_started_at[incident_id] = updated_at.isoformat()
            if state == "resolved":
                started_raw = self.state.incident_started_at.pop(incident_id, None)
                started = _parse_iso(started_raw)
                if started is not None:
                    self.state.incident_mttr_seconds.append(max(0.0, (updated_at - started).total_seconds()))

        queue_dwell = float(self.state.queue_degraded_total_seconds)
        active_since = _parse_iso(self.state.queue_degraded_since)
        if active_since is not None:
            queue_dwell += max(0.0, (current - active_since).total_seconds())

        metrics = {
            "status": payload.get("status"),
            "readiness": readiness,
            "degraded": degraded,
            "queue_requested": queue.get("requested"),
            "queue_active": queue.get("active"),
            "queue_degraded": queue_degraded,
            "queue_degraded_dwell_seconds": queue_dwell,
            "crash_loop_active": bool(crash_loop.get("active", False)),
            "crash_loop_failures": int(crash_loop.get("failure_count") or 0),
            "incident_open_count": len([r for r in incident_rows if isinstance(r, dict) and str(r.get("state")).lower() == "open"]),
            "readiness_converged_seconds": self.state.readiness_converged_seconds,
        }
        if isinstance(resources, dict):
            metrics["resources"] = dict(resources)
            if rss_mb is not None and self.state.baseline_rss_mb is not None:
                metrics["rss_delta_mb"] = rss_mb - self.state.baseline_rss_mb

        self.state.last_snapshot_at = current.isoformat()
        self._append_event(event_type="runtime.snapshot", metrics=metrics, now=current)

    def export(self, *, hours: int = 24, now: datetime | None = None) -> dict[str, Any]:
        current = now or _utcnow()
        rows = self._read_recent_events(hours=hours, now=current)
        aggregates = aggregate_telemetry(rows)

        queue_dwell = float(self.state.queue_degraded_total_seconds)
        active_since = _parse_iso(self.state.queue_degraded_since)
        if active_since is not None:
            queue_dwell += max(0.0, (current - active_since).total_seconds())

        mttr = None
        if self.state.incident_mttr_seconds:
            mttr = sum(self.state.incident_mttr_seconds) / len(self.state.incident_mttr_seconds)

        kpis = {
            "window_hours": int(max(1, hours)),
            "deployment_events": aggregates["deployment_events"],
            "runtime_snapshots": aggregates["runtime_snapshots"],
            "governance_signals": aggregates["governance_signals"],
            "governance_high_risk_signals": aggregates["governance_high_risk_signals"],
            "warning_count": aggregates["warning_count"],
            "warning_frequency_per_min": aggregates["warning_frequency_per_min"],
            "readiness_success_ratio": aggregates["readiness_success_ratio"],
            "avg_crash_loop_failures": aggregates["avg_crash_loop_failures"],
            "queue_degraded_dwell_seconds": queue_dwell,
            "readiness_converged_seconds": self.state.readiness_converged_seconds,
            "incident_mttr_seconds": mttr,
            "resource_baseline_rss_mb": self.state.baseline_rss_mb,
        }
        score = calculate_operational_score(kpis)
        return {
            "schema": self.schema_version,
            "generated_at": current.isoformat(),
            "retention_policy": {
                "retention_hours": self.config.retention_hours,
                "max_events": self.config.max_events,
            },
            "config": {
                "snapshot_interval_seconds": self.config.snapshot_interval_seconds,
                "enabled": self.config.enabled,
            },
            "kpis": kpis,
            "score": score,
            "sample_events": rows[-20:],
        }


def aggregate_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deployment_events = 0
    runtime_snapshots = 0
    governance_signals = 0
    governance_high_risk_signals = 0
    warning_count = 0
    readiness_values: list[float] = []
    crash_loop_values: list[float] = []
    first: datetime | None = None
    last: datetime | None = None

    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = _parse_iso(row.get("time"))
        if ts is not None:
            first = ts if first is None or ts < first else first
            last = ts if last is None or ts > last else last

        event_type = str(row.get("event_type") or "").strip()
        level = str(row.get("level") or "").strip().lower()
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        if event_type == "deployment.event":
            deployment_events += 1
        elif event_type == "runtime.snapshot":
            runtime_snapshots += 1
            readiness_values.append(1.0 if bool(metrics.get("readiness")) else 0.0)
            crash_loop_values.append(float(metrics.get("crash_loop_failures") or 0.0))
        elif event_type == "governance.signal":
            governance_signals += 1
            status = str(metrics.get("status") or "").strip().lower()
            if status in {"near_limit", "limit_exceeded", "elevated", "risk", "near_ceiling"}:
                governance_high_risk_signals += 1
        if event_type == "runtime.warning" or level in {"warning", "error"}:
            warning_count += 1

    minutes = 1.0
    if first is not None and last is not None:
        minutes = max(1.0, (last - first).total_seconds() / 60.0)

    readiness_ratio = sum(readiness_values) / len(readiness_values) if readiness_values else 1.0
    crash_loop_avg = sum(crash_loop_values) / len(crash_loop_values) if crash_loop_values else 0.0

    return {
        "deployment_events": deployment_events,
        "runtime_snapshots": runtime_snapshots,
        "governance_signals": governance_signals,
        "governance_high_risk_signals": governance_high_risk_signals,
        "warning_count": warning_count,
        "warning_frequency_per_min": warning_count / minutes,
        "readiness_success_ratio": readiness_ratio,
        "avg_crash_loop_failures": crash_loop_avg,
    }


def calculate_operational_score(kpis: dict[str, Any]) -> dict[str, Any]:
    readiness = float(kpis.get("readiness_success_ratio") or 0.0)
    warning_rate = float(kpis.get("warning_frequency_per_min") or 0.0)
    queue_dwell = float(kpis.get("queue_degraded_dwell_seconds") or 0.0)
    convergence = kpis.get("readiness_converged_seconds")
    convergence = float(convergence) if convergence is not None else 0.0

    readiness_component = max(0.0, min(1.0, readiness)) * 40.0
    warning_component = max(0.0, 20.0 - min(20.0, warning_rate * 10.0))
    queue_component = max(0.0, 20.0 - min(20.0, queue_dwell / 30.0))
    convergence_component = max(0.0, 20.0 - min(20.0, convergence / 30.0))

    score = readiness_component + warning_component + queue_component + convergence_component
    score = max(0.0, min(100.0, score))
    band = "high" if score >= 90.0 else ("conditional" if score >= 75.0 else "risk")
    return {
        "value": round(score, 2),
        "band": band,
        "components": {
            "readiness": round(readiness_component, 2),
            "warnings": round(warning_component, 2),
            "queue_degraded": round(queue_component, 2),
            "convergence": round(convergence_component, 2),
        },
    }
