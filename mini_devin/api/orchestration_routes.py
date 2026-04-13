"""HTTP routes for Plodder Orchestrator (manager–worker)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class OrchestratorRunRequest(BaseModel):
    session_id: str
    goal: str
    max_replans: int = 3
    max_retries_per_subtask: int = 2


def build_orchestration_router(session_manager: Any, connection_manager: Any) -> APIRouter:
    """Mount with ``app.include_router(router, prefix='/api')``."""

    router = APIRouter()

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
        try:
            result = await orch.run_goal(sid, goal, connection_manager=connection_manager)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from e
        return result.to_dict()

    return router
