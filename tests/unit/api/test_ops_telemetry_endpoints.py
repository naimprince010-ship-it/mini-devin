from __future__ import annotations

import os
import importlib
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

os.environ.setdefault("PLODDER_SKIP_FRONTEND_STATIC", "1")

from mini_devin.api.app import app

app_module = importlib.import_module("mini_devin.api.app")


def test_ops_telemetry_export_endpoint_shapes_payload() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        # Ensure there is at least one snapshot event in the window.
        _ = client.get("/api/health")
        response = client.get("/api/ops/telemetry/export", params={"hours": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.telemetry.v1"
    assert "kpis" in payload
    assert "score" in payload
    assert "retention_policy" in payload


def test_ops_telemetry_score_endpoint_shapes_payload() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        _ = client.get("/api/health")
        response = client.get("/api/ops/telemetry/score", params={"hours": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "ops.telemetry.v1"
    assert "score" in payload
    assert "kpis" in payload
    assert "value" in payload["score"]
    assert "band" in payload["score"]


def test_status_exposes_governance_runtime_flags(monkeypatch) -> None:
    monkeypatch.setenv("PLODDER_GOVERNANCE_TELEMETRY", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_BUDGET_SIGNALS", "true")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_RETRY_SIGNALS", "false")
    monkeypatch.setenv("PLODDER_GOVERNANCE_EMIT_LOOP_SIGNALS", "true")
    monkeypatch.setattr(app_module.session_manager, "get_active_session_count", AsyncMock(return_value=0))
    monkeypatch.setattr(app_module.session_manager, "get_total_tasks_completed", AsyncMock(return_value=0))
    monkeypatch.setattr(app_module.session_manager, "get_uptime_seconds", lambda: 0.0)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    deployment_env = response.json().get("deployment_env", {})
    assert deployment_env["governance_telemetry_enabled"] is True
    assert deployment_env["governance_emit_budget_signals"] is True
    assert deployment_env["governance_emit_retry_signals"] is False
    assert deployment_env["governance_emit_loop_signals"] is True
