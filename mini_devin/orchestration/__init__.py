"""
Plodder multi-agent orchestration (OpenHands-style manager–worker).

HTTP: importing ``mini_devin.api`` mounts ``POST /api/orchestration/run`` (see ``api/__init__.py``).
Use ``uvicorn mini_devin.api:app`` (not ``mini_devin.api.app:app``) so that import runs.

Alternatively mount manually::

    from mini_devin.api.orchestration_routes import build_orchestration_router
    app.include_router(build_orchestration_router(session_manager, connection_manager), prefix="/api")

Body: ``{"session_id": "...", "goal": "..."}``.
"""

from .plodder_orchestrator import OrchestratorRunResult, PlodderOrchestrator
from .protocol import PlanEvent, SupervisorAction, WorkerObservation
from .task_scheduler import (
    SchedulableUnit,
    auto_skip_when_dep_skipped,
    run_dag,
    topological_layers,
    transitive_descendants,
)
from .worker_runtime import WorkerRuntime, orchestrator_worker_use_sandbox

__all__ = [
    "OrchestratorRunResult",
    "PlodderOrchestrator",
    "PlanEvent",
    "SupervisorAction",
    "WorkerObservation",
    "SchedulableUnit",
    "auto_skip_when_dep_skipped",
    "run_dag",
    "topological_layers",
    "transitive_descendants",
    "WorkerRuntime",
    "orchestrator_worker_use_sandbox",
]
