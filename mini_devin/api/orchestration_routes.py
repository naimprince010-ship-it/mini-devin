"""HTTP routes for Plodder Orchestrator (manager–worker)."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class OrchestratorRunRequest(BaseModel):
    session_id: str
    goal: str
    max_replans: int = 3
    max_retries_per_subtask: int = 2


_ACTIVE_ORCHESTRATORS: dict[str, Any] = {}
_LAST_POOL_METRICS: dict[str, dict[str, Any]] = {}
_POOL_METRICS_UPDATED_AT: dict[str, str] = {}
_STATE_LOCK = Lock()


def build_orchestration_router(session_manager: Any, connection_manager: Any) -> APIRouter:
    """Mount with ``app.include_router(router, prefix='/api')``."""

    router = APIRouter()

    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _compute_saturated_roles(snapshot: dict[str, dict[str, Any]]) -> list[str]:
        return sorted(
            role
            for role, stats in snapshot.items()
            if bool((stats or {}).get("saturated"))
        )

    @router.post("/orchestration/run")
    async def run_plodder_orchestrator(req: OrchestratorRunRequest):
        goal = (req.goal or "").strip()
        if not goal:
            raise HTTPException(status_code=400, detail="goal is required")
        sid = (req.session_id or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail="session_id is required")
        sess = await session_manager.get_session(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        from ..orchestration.plodder_orchestrator import PlodderOrchestrator

        orch = PlodderOrchestrator(
            session_manager,
            max_retries_per_subtask=req.max_retries_per_subtask,
            max_replans=req.max_replans,
        )
        with _STATE_LOCK:
            _ACTIVE_ORCHESTRATORS[sid] = orch
        print(f"[Orchestration] run started session_id={sid} goal={goal[:120]}")
        try:
            result = await orch.run_goal(sid, goal, connection_manager=connection_manager)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            snapshot: dict[str, dict[str, Any]] = {}
            with contextlib.suppress(Exception):
                snapshot = dict(orch.worker_pool_snapshot())
            with _STATE_LOCK:
                current = _ACTIVE_ORCHESTRATORS.get(sid)
                if current is orch:
                    _ACTIVE_ORCHESTRATORS.pop(sid, None)
                if snapshot:
                    _LAST_POOL_METRICS[sid] = snapshot
                    _POOL_METRICS_UPDATED_AT[sid] = _utc_now_iso()
            if snapshot:
                saturated = _compute_saturated_roles(snapshot)
                print(
                    f"[Orchestration] pool-metrics session_id={sid} saturated_roles={','.join(saturated) or '-'}"
                )

        return result.to_dict()

    @router.get("/orchestration/sessions/{session_id}/worker-pools")
    async def get_worker_pool_metrics(session_id: str):
        sid = (session_id or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail="session_id is required")
        sess = await session_manager.get_session(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        with _STATE_LOCK:
            active = _ACTIVE_ORCHESTRATORS.get(sid)
            cached = dict(_LAST_POOL_METRICS.get(sid) or {})
            updated_at = _POOL_METRICS_UPDATED_AT.get(sid)

        if active is not None:
            pools = dict(active.worker_pool_snapshot())
            source = "active"
            updated_at = _utc_now_iso()
        elif cached:
            pools = cached
            source = "last_run"
            updated_at = updated_at or _utc_now_iso()
        else:
            raise HTTPException(
                status_code=404,
                detail="No worker pool metrics for this session yet. Run orchestration first.",
            )

        saturated_roles = _compute_saturated_roles(pools)
        return {
            "session_id": sid,
            "source": source,
            "updated_at": updated_at,
            "saturated_roles": saturated_roles,
            "pools": pools,
        }

    return router
