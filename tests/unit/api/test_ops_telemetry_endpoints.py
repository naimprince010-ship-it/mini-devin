from __future__ import annotations

import os

from fastapi.testclient import TestClient

os.environ.setdefault("PLODDER_SKIP_FRONTEND_STATIC", "1")

from mini_devin.api.app import app


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
