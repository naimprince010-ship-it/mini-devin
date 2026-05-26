from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mini_devin.reliability.ops_telemetry import (
    FileOpsTelemetryCollector,
    OpsTelemetryConfig,
    calculate_operational_score,
)


def _collector(tmp_path: Path) -> FileOpsTelemetryCollector:
    return FileOpsTelemetryCollector(
        events_file=tmp_path / "events.jsonl",
        state_file=tmp_path / "state.json",
        config=OpsTelemetryConfig(enabled=True, snapshot_interval_seconds=1, retention_hours=1, max_events=1_000),
    )


def test_runtime_snapshot_export_contains_kpis_and_score(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    now = datetime.now(timezone.utc)
    collector.record_deployment_event("deploy.start", now=now)
    collector.record_runtime_snapshot(
        {
            "status": "healthy",
            "checks": {
                "readiness": True,
                "degraded": False,
                "queue_backend": {"requested": "memory", "active": "memory", "degraded": False},
                "incidents": {"incidents": [], "crash_loop": {"active": False, "failure_count": 0}},
            },
        },
        resources={"rss_mb": 42.0},
        now=now + timedelta(seconds=10),
    )

    export = collector.export(hours=1, now=now + timedelta(seconds=10))
    assert export["schema"] == "ops.telemetry.v1"
    assert "kpis" in export
    assert "score" in export
    assert export["kpis"]["runtime_snapshots"] >= 1


def test_queue_degraded_dwell_accumulates_and_closes(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    start = datetime.now(timezone.utc)

    degraded_payload = {
        "status": "degraded",
        "checks": {
            "readiness": True,
            "degraded": True,
            "queue_backend": {"requested": "redis_streams", "active": "memory", "degraded": True},
            "incidents": {"incidents": [], "crash_loop": {"active": False, "failure_count": 0}},
        },
    }
    healthy_payload = {
        "status": "healthy",
        "checks": {
            "readiness": True,
            "degraded": False,
            "queue_backend": {"requested": "memory", "active": "memory", "degraded": False},
            "incidents": {"incidents": [], "crash_loop": {"active": False, "failure_count": 0}},
        },
    }

    collector.record_runtime_snapshot(degraded_payload, now=start)
    collector.record_runtime_snapshot(healthy_payload, now=start + timedelta(seconds=12))

    export = collector.export(hours=1, now=start + timedelta(seconds=12))
    assert export["kpis"]["queue_degraded_dwell_seconds"] >= 12.0


def test_readiness_convergence_measured_from_deploy_start(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    start = datetime.now(timezone.utc)
    collector.record_deployment_event("deploy.start", now=start)

    payload = {
        "status": "healthy",
        "checks": {
            "readiness": True,
            "degraded": False,
            "queue_backend": {"requested": "memory", "active": "memory", "degraded": False},
            "incidents": {"incidents": [], "crash_loop": {"active": False, "failure_count": 0}},
        },
    }
    collector.record_runtime_snapshot(payload, now=start + timedelta(seconds=30))
    export = collector.export(hours=1, now=start + timedelta(seconds=30))
    assert export["kpis"]["readiness_converged_seconds"] is not None
    assert export["kpis"]["readiness_converged_seconds"] >= 30.0


def test_retention_policy_prunes_old_rows(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    now = datetime.now(timezone.utc)

    collector.record_warning("old", "old warning", now=now - timedelta(hours=3))
    collector.record_warning("new", "new warning", now=now)
    collector._enforce_retention(now=now)

    rows = collector._read_recent_events(hours=4, now=now)
    assert len(rows) == 1
    assert rows[0]["metrics"]["code"] == "new"


def test_operational_score_bands() -> None:
    high = calculate_operational_score(
        {
            "readiness_success_ratio": 1.0,
            "warning_frequency_per_min": 0.0,
            "queue_degraded_dwell_seconds": 0.0,
            "readiness_converged_seconds": 30.0,
        }
    )
    risk = calculate_operational_score(
        {
            "readiness_success_ratio": 0.3,
            "warning_frequency_per_min": 4.0,
            "queue_degraded_dwell_seconds": 2_000.0,
            "readiness_converged_seconds": 1_200.0,
        }
    )
    assert high["band"] in {"high", "conditional"}
    assert risk["band"] == "risk"
