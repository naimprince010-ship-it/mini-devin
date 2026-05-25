from __future__ import annotations

from fastapi.testclient import TestClient

from mini_devin.api.app import app
from mini_devin.orchestration.task_queue import create_queue_transport


def test_readiness_reports_starting_when_db_not_ready() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.startup_status = {
            "boot_started_at": "2026-01-01T00:00:00+00:00",
            "db_ready": False,
            "db_last_error": "database_init_timeout",
            "degraded": True,
            "preflight": {"startup_mode": "degraded", "checks": []},
            "issues": [],
        }
        response = client.get("/api/readiness")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "starting"
    assert payload["checks"]["readiness"] is False


def test_liveness_remains_healthy_during_degraded_mode() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.startup_status = {
            "boot_started_at": "2026-01-01T00:00:00+00:00",
            "db_ready": False,
            "db_last_error": "database_init_timeout",
            "degraded": True,
            "preflight": {"startup_mode": "degraded", "checks": []},
            "issues": [],
        }
        response = client.get("/api/liveness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["degraded"] is True


def test_health_reports_queue_backend_degradation(monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_QUEUE_BACKEND", "redis_streams")
    monkeypatch.setenv("PLODDER_QUEUE_FORCE_BACKEND_FAILURE", "1")
    create_queue_transport()

    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.startup_status = {
            "boot_started_at": "2026-01-01T00:00:00+00:00",
            "db_ready": True,
            "db_last_error": None,
            "degraded": False,
            "preflight": {"startup_mode": "normal", "checks": []},
            "issues": [],
        }
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    queue = payload["checks"]["queue_backend"]
    assert payload["status"] == "degraded"
    assert queue["requested"] == "redis_streams"
    assert queue["active"] == "memory"
    assert queue["degraded"] is True


def test_operator_visible_runtime_diagnostics_endpoint() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.startup_status = {
            "boot_started_at": "2026-01-01T00:00:00+00:00",
            "db_ready": True,
            "db_last_error": None,
            "degraded": False,
            "preflight": {"startup_mode": "normal", "checks": []},
            "issues": [],
            "stage_history": ["boot.begin", "preflight.complete", "db.init.start", "db.init.complete"],
        }
        response = client.get("/api/ops/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert "incidents" in payload
    assert "startup_sequence_issues" in payload
    assert "expected_startup_order" in payload


def test_deploy_preflight_returns_structured_issues(monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_QUEUE_FAILOVER_POLICY", "bad-policy")
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.startup_status = {
            "boot_started_at": "2026-01-01T00:00:00+00:00",
            "db_ready": True,
            "db_last_error": None,
            "degraded": False,
            "preflight": {"startup_mode": "normal", "checks": []},
            "issues": [],
            "stage_history": ["boot.begin"],
        }
        response = client.get("/api/ops/preflight")

    assert response.status_code == 200
    payload = response.json()
    assert "issues" in payload
    assert any(item["code"] == "config.invalid.queue_failover_policy" for item in payload["issues"])
