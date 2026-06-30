from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mini_devin.orchestration.capability_router import SpecialistRole
from mini_devin.orchestration.plodder_orchestrator import PlodderOrchestrator
from mini_devin.orchestration.task_scheduler import SchedulableUnit
from mini_devin.orchestration.worker_pools import SpecialistWorkerPoolManager


def test_pool_enforces_per_role_limit() -> None:
    mgr = SpecialistWorkerPoolManager(
        limits={
            SpecialistRole.FRONTEND: 1,
            SpecialistRole.BACKEND: 2,
            SpecialistRole.QA: 1,
            SpecialistRole.DEVOPS: 1,
            SpecialistRole.GENERALIST: 1,
        }
    )

    state = {"active": 0, "max_active": 0}

    async def one(idx: int) -> None:
        async with mgr.acquire(SpecialistRole.FRONTEND, f"u{idx}"):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            await asyncio.sleep(0.03)
            state["active"] -= 1

    async def run() -> None:
        await asyncio.gather(*(one(i) for i in range(4)))

    asyncio.run(run())
    assert state["max_active"] == 1


def test_pool_isolation_between_roles() -> None:
    mgr = SpecialistWorkerPoolManager(
        limits={
            SpecialistRole.FRONTEND: 1,
            SpecialistRole.BACKEND: 1,
            SpecialistRole.QA: 1,
            SpecialistRole.DEVOPS: 1,
            SpecialistRole.GENERALIST: 1,
        }
    )

    marks: dict[str, float] = {}

    async def frontend_task() -> None:
        async with mgr.acquire(SpecialistRole.FRONTEND, "fe"):
            marks["frontend_start"] = asyncio.get_running_loop().time()
            await asyncio.sleep(0.08)
            marks["frontend_end"] = asyncio.get_running_loop().time()

    async def backend_task() -> None:
        await asyncio.sleep(0.01)
        async with mgr.acquire(SpecialistRole.BACKEND, "be"):
            marks["backend_start"] = asyncio.get_running_loop().time()
            await asyncio.sleep(0.01)
            marks["backend_end"] = asyncio.get_running_loop().time()

    async def run() -> None:
        await asyncio.gather(frontend_task(), backend_task())

    asyncio.run(run())
    assert marks["backend_start"] < marks["frontend_end"]


class _FakeSessionManager:
    async def create_session(self, **kwargs):
        return SimpleNamespace(session_id="worker-sess", **kwargs)

    async def create_task(self, session_id: str, description: str, connection_manager=None):
        del session_id, description, connection_manager
        return SimpleNamespace(task_id="task-1")

    async def run_task(self, session_id: str, task_id: str, connection_manager=None):
        del session_id, task_id, connection_manager
        return None

    async def get_task(self, session_id: str, task_id: str):
        del session_id, task_id
        return SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            result=SimpleNamespace(summary="ok", status="completed"),
            error_message=None,
        )


def test_orchestrator_observation_includes_specialist_pool_metadata() -> None:
    orchestrator = PlodderOrchestrator(session_manager=_FakeSessionManager(), supervisor=object())
    orchestrator._worker_pools = SpecialistWorkerPoolManager(
        limits={
            SpecialistRole.FRONTEND: 1,
            SpecialistRole.BACKEND: 1,
            SpecialistRole.QA: 1,
            SpecialistRole.DEVOPS: 1,
            SpecialistRole.GENERALIST: 1,
        }
    )

    unit = SchedulableUnit(
        id="backend-fix",
        goal="Fix FastAPI endpoint validation and session DB write",
        acceptance_criteria=["add tests"],
        depends_on=(),
    )

    async def run_one():
        return await orchestrator._execute_worker_unit(
            unit,
            prior_obs={},
            parent=SimpleNamespace(model="gpt-4o", max_iterations=20),
            workspace=".",
            connection_manager=None,
            actions=[],
        )

    obs = asyncio.run(run_one())
    assert obs.success is True
    assert obs.result.get("specialist_role") == SpecialistRole.BACKEND.value
    pool = obs.result.get("worker_pool")
    assert isinstance(pool, dict)
    assert pool.get("role") == SpecialistRole.BACKEND.value
    assert int(pool.get("limit", 0)) == 1
