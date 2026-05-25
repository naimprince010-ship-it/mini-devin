from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class IncidentRecord:
    incident_id: str
    code: str
    severity: str
    message: str
    state: str
    started_at: str
    updated_at: str
    recovered_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "state": self.state,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "recovered_at": self.recovered_at,
        }


@dataclass(slots=True)
class CrashLoopSnapshot:
    active: bool
    failure_count: int
    window_seconds: int
    threshold: int
    last_failure_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "failure_count": self.failure_count,
            "window_seconds": self.window_seconds,
            "threshold": self.threshold,
            "last_failure_at": self.last_failure_at,
        }


@dataclass(slots=True)
class RuntimeIncidentTracker:
    crash_loop_file: Path
    threshold: int = 3
    window_seconds: int = 300
    _incidents: dict[str, IncidentRecord] = field(default_factory=dict)
    _failures: list[datetime] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.crash_loop_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_crash_loop_state()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _trim_window(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.window_seconds)
        self._failures = [ts for ts in self._failures if ts >= cutoff]

    def _save_crash_loop_state(self) -> None:
        payload = {
            "failure_timestamps": [ts.isoformat() for ts in self._failures],
            "threshold": self.threshold,
            "window_seconds": self.window_seconds,
        }
        self.crash_loop_file.write_text(json.dumps(payload), encoding="utf-8")

    def _load_crash_loop_state(self) -> None:
        if not self.crash_loop_file.exists():
            return
        try:
            payload = json.loads(self.crash_loop_file.read_text(encoding="utf-8"))
        except Exception:
            return
        timestamps = payload.get("failure_timestamps", [])
        parsed: list[datetime] = []
        for item in timestamps:
            try:
                parsed.append(datetime.fromisoformat(str(item)))
            except Exception:
                continue
        self._failures = parsed

    def record_failure(self, code: str, message: str, *, severity: str = "error") -> IncidentRecord:
        now = self._now()
        self._failures.append(now)
        self._trim_window(now)
        self._save_crash_loop_state()

        incident = self._incidents.get(code)
        if incident is None:
            incident = IncidentRecord(
                incident_id=f"inc-{abs(hash((code, now.isoformat()))) % 1_000_000_000}",
                code=code,
                severity=severity,
                message=message,
                state="open",
                started_at=now.isoformat(),
                updated_at=now.isoformat(),
            )
        else:
            incident.message = message
            incident.state = "open"
            incident.severity = severity
            incident.updated_at = now.isoformat()
            incident.recovered_at = None
        self._incidents[code] = incident
        return incident

    def record_recovery(self, code: str, message: str | None = None) -> IncidentRecord | None:
        now = self._now()
        incident = self._incidents.get(code)
        if incident is None:
            return None
        incident.state = "resolved"
        incident.updated_at = now.isoformat()
        incident.recovered_at = now.isoformat()
        if message:
            incident.message = message
        self._incidents[code] = incident
        return incident

    def note_startup_success(self) -> None:
        self._failures = []
        self._save_crash_loop_state()

    def crash_loop_snapshot(self) -> CrashLoopSnapshot:
        now = self._now()
        self._trim_window(now)
        last_failure = self._failures[-1].isoformat() if self._failures else None
        active = len(self._failures) >= self.threshold
        return CrashLoopSnapshot(
            active=active,
            failure_count=len(self._failures),
            window_seconds=self.window_seconds,
            threshold=self.threshold,
            last_failure_at=last_failure,
        )

    def diagnostics(self) -> dict[str, Any]:
        incidents = [item.to_dict() for item in self._incidents.values()]
        incidents.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
        return {
            "incidents": incidents,
            "crash_loop": self.crash_loop_snapshot().to_dict(),
        }
