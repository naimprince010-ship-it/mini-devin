from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mini_devin.api.orchestration_routes import build_orchestration_router


class _FakeSessionManager:
    async def get_session(self, session_id: str):
        if session_id == "missing":
            return None
        return SimpleNamespace(session_id=session_id)


class _FakeOrchestrator:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self._snapshot = {
            "backend": {
                "limit": 1,
                "active": 1,
                "waiting": 2,
                "utilization": 1.0,
                "saturated": True,
                "active_units": ["u1"],
            },
            "frontend": {
                "limit": 2,
                "active": 0,
                "waiting": 0,
                "utilization": 0.0,
                "saturated": False,
                "active_units": [],
            },
        }

    async def run_goal(self, sid: str, goal: str, connection_manager=None):
        del sid, goal, connection_manager
        return SimpleNamespace(to_dict=lambda: {"ok": True})

    def worker_pool_snapshot(self):
        return dict(self._snapshot)


def _build_client(monkeypatch) -> TestClient:
    from mini_devin.orchestration import plodder_orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "PlodderOrchestrator", _FakeOrchestrator)
    app = FastAPI()
    app.include_router(
        build_orchestration_router(_FakeSessionManager(), connection_manager=None),
        prefix="/api",
    )
    return TestClient(app, raise_server_exceptions=False)


def test_worker_pool_metrics_available_after_run(monkeypatch) -> None:
    with _build_client(monkeypatch) as client:
        run_res = client.post(
            "/api/orchestration/run",
            json={"session_id": "sess-1", "goal": "Implement API endpoint"},
        )
        assert run_res.status_code == 200

        metrics_res = client.get("/api/orchestration/sessions/sess-1/worker-pools")

    assert metrics_res.status_code == 200
    payload = metrics_res.json()
    assert payload["session_id"] == "sess-1"
    assert payload["source"] in {"active", "last_run"}
    assert "backend" in payload["pools"]
    assert payload["pools"]["backend"]["saturated"] is True
    assert "backend" in payload["saturated_roles"]


def test_worker_pool_metrics_404_before_any_run(monkeypatch) -> None:
    with _build_client(monkeypatch) as client:
        res = client.get("/api/orchestration/sessions/new-session/worker-pools")

    assert res.status_code == 404
    assert "Run orchestration first" in res.json().get("detail", "")
