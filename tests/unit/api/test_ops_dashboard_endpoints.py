from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from mini_devin.reliability.ops_telemetry import FileOpsTelemetryCollector, OpsTelemetryConfig

os.environ.setdefault("PLODDER_SKIP_FRONTEND_STATIC", "1")

from mini_devin.api.app import app


def _collector(tmp_path: Path) -> FileOpsTelemetryCollector:
    collector = FileOpsTelemetryCollector(
        events_file=tmp_path / "events.jsonl",
        state_file=tmp_path / "state.json",
        config=OpsTelemetryConfig(enabled=True, snapshot_interval_seconds=1, retention_hours=12, max_events=10_000),
    )
    app.state.ops_telemetry = collector
    return collector


def _seed_dashboard_telemetry(collector: FileOpsTelemetryCollector) -> dict[str, str]:
    base = datetime.now(timezone.utc) - timedelta(minutes=10)
    collector.record_deployment_event("deploy.start", now=base)
    collector.record_deployment_event("deploy.preflight.complete", now=base + timedelta(seconds=5))

    degraded_open = {
        "status": "degraded",
        "checks": {
            "readiness": False,
            "degraded": True,
            "queue_backend": {
                "requested": "redis_streams",
                "active": "memory",
                "degraded": True,
            },
            "incidents": {
                "incidents": [
                    {
                        "incident_id": "inc-1",
                        "state": "open",
                        "updated_at": (base + timedelta(seconds=10)).isoformat(),
                    }
                ],
                "crash_loop": {"active": True, "failure_count": 2},
            },
        },
    }

    degraded_progress = {
        "status": "degraded",
        "checks": {
            "readiness": True,
            "degraded": True,
            "queue_backend": {
                "requested": "redis_streams",
                "active": "memory",
                "degraded": True,
            },
            "incidents": {
                "incidents": [
                    {
                        "incident_id": "inc-1",
                        "state": "open",
                        "updated_at": (base + timedelta(seconds=25)).isoformat(),
                    }
                ],
                "crash_loop": {"active": True, "failure_count": 3},
            },
        },
    }

    recovered = {
        "status": "healthy",
        "checks": {
            "readiness": True,
            "degraded": False,
            "queue_backend": {
                "requested": "memory",
                "active": "memory",
                "degraded": False,
            },
            "incidents": {
                "incidents": [
                    {
                        "incident_id": "inc-1",
                        "state": "resolved",
                        "updated_at": (base + timedelta(seconds=40)).isoformat(),
                    }
                ],
                "crash_loop": {"active": False, "failure_count": 0},
            },
        },
    }

    collector.record_runtime_snapshot(degraded_open, now=base + timedelta(seconds=10))
    collector.record_warning("ops.warning.1", "first warning", now=base + timedelta(seconds=12))
    collector.record_runtime_snapshot(degraded_progress, now=base + timedelta(seconds=25))
    collector.record_warning("ops.warning.2", "second warning", now=base + timedelta(seconds=26))
    collector.record_runtime_snapshot(recovered, now=base + timedelta(seconds=40))

    # Out-of-window signal for filtering assertions.
    collector.record_warning("ops.warning.old", "old warning", now=base - timedelta(hours=3))

    return {
        "base": base.isoformat(),
        "t_deploy_start": base.isoformat(),
        "t_deploy_preflight": (base + timedelta(seconds=5)).isoformat(),
        "t_recovered": (base + timedelta(seconds=40)).isoformat(),
    }


def test_dashboard_summary_endpoint(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    _seed_dashboard_telemetry(collector)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/ops/dashboard/summary", params={"hours": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.dashboard.v1"
    assert "kpis" in payload
    assert "score" in payload
    assert "timeline_counts" in payload


def test_dashboard_timeline_endpoint_returns_items(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    _seed_dashboard_telemetry(collector)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/ops/dashboard/timeline", params={"hours": 1, "page_size": 100})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.dashboard.v1"
    assert len(payload["items"]) >= 3
    kinds = {item["kind"] for item in payload["items"]}
    assert "runtime_health" in kinds
    assert "deployment_event" in kinds


def test_queue_degradation_timeline_endpoint(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    _seed_dashboard_telemetry(collector)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/ops/dashboard/timeline/queue-degradation", params={"hours": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.dashboard.v1"
    assert len(payload["items"]) >= 1
    assert payload["items"][0]["duration_seconds"] >= 1.0


def test_incident_lifecycle_timeline_endpoint(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    _seed_dashboard_telemetry(collector)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/ops/dashboard/timeline/incidents", params={"hours": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.dashboard.v1"
    assert len(payload["items"]) >= 2
    events = {item["event"] for item in payload["items"]}
    assert "opened" in events
    assert "resolved" in events


def test_score_history_endpoint(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    _seed_dashboard_telemetry(collector)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/ops/dashboard/trends/score-history", params={"hours": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.dashboard.v1"
    assert len(payload["items"]) >= 1
    assert "value" in payload["items"][0]
    assert "band" in payload["items"][0]


def test_dashboard_timeline_pagination(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    _seed_dashboard_telemetry(collector)

    with TestClient(app, raise_server_exceptions=False) as client:
        page_one = client.get("/api/ops/dashboard/timeline", params={"hours": 1, "page": 1, "page_size": 1})
        page_two = client.get("/api/ops/dashboard/timeline", params={"hours": 1, "page": 2, "page_size": 1})

    assert page_one.status_code == 200
    assert page_two.status_code == 200
    p1 = page_one.json()
    p2 = page_two.json()
    assert p1["pagination"]["page_size"] == 1
    assert p1["pagination"]["total_items"] >= 2
    assert len(p1["items"]) == 1
    assert len(p2["items"]) == 1
    assert p1["items"][0]["time"] != p2["items"][0]["time"]


def test_dashboard_time_window_filtering(tmp_path: Path) -> None:
    collector = _collector(tmp_path)
    marks = _seed_dashboard_telemetry(collector)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            "/api/ops/dashboard/timeline/deployments",
            params={
                "start_time": marks["t_deploy_preflight"],
                "end_time": marks["t_recovered"],
                "page_size": 20,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.dashboard.v1"
    assert len(payload["items"]) == 1
    assert payload["items"][0]["phase"] == "deploy.preflight.complete"
