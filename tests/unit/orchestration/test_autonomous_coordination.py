from __future__ import annotations

from pathlib import Path

from mini_devin.orchestration.autonomous_coordination import (
    AgentBudgetPolicy,
    AgentBudgetState,
    AutonomousCoordinationConfig,
    AutonomousCoordinatorRuntime,
    RecoveryAction,
    RecoveryDecision,
)
from mini_devin.orchestration.protocol import WorkerObservation
from mini_devin.orchestration.task_scheduler import SchedulableUnit


def _unit(unit_id: str, *, depends_on: tuple[str, ...] = ()) -> SchedulableUnit:
    return SchedulableUnit(
        id=unit_id,
        goal=f"goal-{unit_id}",
        acceptance_criteria=[f"ac-{unit_id}"],
        depends_on=depends_on,
    )


def _obs(unit_id: str, success: bool, *, error: str | None = None) -> WorkerObservation:
    return WorkerObservation.from_task_outcome(
        action_id=f"act-{unit_id}",
        subtask_id=unit_id,
        success=success,
        summary=f"summary-{unit_id}",
        worker_session_id="worker-session",
        task_id=f"task-{unit_id}",
        error=error,
    )


async def _planner(goal: str, workspace: str) -> tuple[list[SchedulableUnit], str]:
    del goal, workspace
    return [_unit("a"), _unit("b", depends_on=("a",))], "planner-reasoning"


async def _implementer_success(unit: SchedulableUnit) -> WorkerObservation:
    return _obs(unit.id, True)


def test_task_decomposition_and_execution(tmp_path: Path) -> None:
    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-1",
        planner_fn=_planner,
        implementer_fn=_implementer_success,
    )

    result = __import__("asyncio").run(runtime.run("ship feature"))

    assert result.terminated_reason == "completed"
    assert set(result.observations.keys()) == {"a", "b"}
    assert any(evt.event_type == "decomposition.completed" for evt in result.events)


def test_verifier_gating_blocks_promotion(tmp_path: Path) -> None:
    async def verifier_fail(unit: SchedulableUnit, obs: WorkerObservation) -> bool:
        del unit, obs
        return False

    async def recovery_abort(unit: SchedulableUnit, obs: WorkerObservation, failure_count: int) -> RecoveryDecision:
        del unit, obs, failure_count
        return RecoveryDecision(action=RecoveryAction.ABORT, reason="gate_failed")

    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-2",
        planner_fn=_planner,
        implementer_fn=_implementer_success,
        verifier_fn=verifier_fail,
        recovery_fn=recovery_abort,
    )

    result = __import__("asyncio").run(runtime.run("verify gate"))

    assert result.terminated_reason is not None
    assert result.terminated_reason.startswith("aborted:")
    assert not any(evt.event_type == "unit.promoted" for evt in result.events)


def test_replan_triggering(tmp_path: Path) -> None:
    async def planner_one(goal: str, workspace: str) -> tuple[list[SchedulableUnit], str]:
        del goal, workspace
        return [_unit("seed")], "seed-plan"

    async def implementer(unit: SchedulableUnit) -> WorkerObservation:
        if unit.id == "seed":
            return _obs(unit.id, False, error="seed failed")
        return _obs(unit.id, True)

    async def verifier(unit: SchedulableUnit, obs: WorkerObservation) -> bool:
        return bool(obs.success)

    async def recovery_replan(unit: SchedulableUnit, obs: WorkerObservation, failure_count: int) -> RecoveryDecision:
        del unit, obs, failure_count
        return RecoveryDecision(action=RecoveryAction.REPLAN, reason="need_replan")

    async def replan_fn(unit: SchedulableUnit, obs: WorkerObservation, board) -> list[SchedulableUnit]:
        del unit, obs, board
        return [_unit("replacement")]

    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-3",
        planner_fn=planner_one,
        implementer_fn=implementer,
        verifier_fn=verifier,
        recovery_fn=recovery_replan,
        replan_fn=replan_fn,
    )

    result = __import__("asyncio").run(runtime.run("replan flow"))

    assert result.replans >= 1
    assert "replacement" in result.observations
    assert any(evt.event_type == "replan.applied" for evt in result.events)


def test_shared_memory_synchronization(tmp_path: Path) -> None:
    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-4",
        planner_fn=_planner,
        implementer_fn=_implementer_success,
    )

    result = __import__("asyncio").run(runtime.run("sync memory"))

    assert result.blackboard_version > 0
    assert len(result.events) > 0


def test_budget_coordination(tmp_path: Path) -> None:
    budget = AgentBudgetState(policy=AgentBudgetPolicy(max_total_actions=1))
    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-5",
        planner_fn=_planner,
        implementer_fn=_implementer_success,
        budget_state=budget,
    )

    result = __import__("asyncio").run(runtime.run("budget"))

    assert result.terminated_reason == "budget_exhausted:implementer"


def test_recovery_escalation_limit(tmp_path: Path) -> None:
    async def planner_one(goal: str, workspace: str) -> tuple[list[SchedulableUnit], str]:
        del goal, workspace
        return [_unit("retry-me")], "retry"

    async def implementer_fail(unit: SchedulableUnit) -> WorkerObservation:
        return _obs(unit.id, False, error="boom")

    async def verifier(unit: SchedulableUnit, obs: WorkerObservation) -> bool:
        del unit, obs
        return False

    async def recovery_retry(unit: SchedulableUnit, obs: WorkerObservation, failure_count: int) -> RecoveryDecision:
        del unit, obs, failure_count
        return RecoveryDecision(action=RecoveryAction.RETRY, reason="retry")

    budget = AgentBudgetState(policy=AgentBudgetPolicy(max_recovery_escalations=1, max_total_actions=20))
    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-6",
        planner_fn=planner_one,
        implementer_fn=implementer_fail,
        verifier_fn=verifier,
        recovery_fn=recovery_retry,
        budget_state=budget,
    )

    result = __import__("asyncio").run(runtime.run("recovery"))

    assert result.terminated_reason == "budget_exhausted:recovery_escalations"


def test_bounded_execution_termination_on_deadlock(tmp_path: Path) -> None:
    async def planner_deadlock(goal: str, workspace: str) -> tuple[list[SchedulableUnit], str]:
        del goal, workspace
        return [_unit("blocked", depends_on=("missing",))], "deadlock"

    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-7",
        planner_fn=planner_deadlock,
        implementer_fn=_implementer_success,
        config=AutonomousCoordinationConfig(max_cycles=2),
    )

    result = __import__("asyncio").run(runtime.run("deadlock"))

    assert result.terminated_reason == "bounded_deadlock"
    assert any("No runnable units" in err for err in result.errors)


def test_orchestrator_recovers_from_implementer_exception(tmp_path: Path) -> None:
    async def planner_one(goal: str, workspace: str) -> tuple[list[SchedulableUnit], str]:
        del goal, workspace
        return [_unit("unstable")], "single"

    async def implementer_boom(unit: SchedulableUnit) -> WorkerObservation:
        del unit
        raise RuntimeError("sandbox startup failed")

    async def verifier(unit: SchedulableUnit, obs: WorkerObservation) -> bool:
        del unit
        return bool(obs.success)

    async def recovery_abort(unit: SchedulableUnit, obs: WorkerObservation, failure_count: int) -> RecoveryDecision:
        del unit, obs, failure_count
        return RecoveryDecision(action=RecoveryAction.ABORT, reason="implementer_crash")

    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-8",
        planner_fn=planner_one,
        implementer_fn=implementer_boom,
        verifier_fn=verifier,
        recovery_fn=recovery_abort,
    )

    result = __import__("asyncio").run(runtime.run("recover from crash"))

    assert result.terminated_reason is not None
    assert result.terminated_reason.startswith("aborted:unstable")
    assert "unstable" in result.observations
    assert result.observations["unstable"].success is False
