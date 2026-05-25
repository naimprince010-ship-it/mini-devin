from __future__ import annotations

from pathlib import Path

from mini_devin.orchestration.autonomous_coordination import AutonomousCoordinatorRuntime
from mini_devin.orchestration.hitl_governance import (
    ApprovalResolution,
    ApprovalStatus,
    GovernancePolicy,
    HITLGovernor,
    RiskLevel,
)
from mini_devin.orchestration.protocol import WorkerObservation
from mini_devin.orchestration.task_scheduler import SchedulableUnit


def _unit(unit_id: str, *, goal: str) -> SchedulableUnit:
    return SchedulableUnit(
        id=unit_id,
        goal=goal,
        acceptance_criteria=["ac-1"],
        depends_on=(),
    )


async def _planner(goal: str, workspace: str) -> tuple[list[SchedulableUnit], str]:
    del goal, workspace
    return [_unit("u-1", goal="deploy production migration")], "reason"


async def _implementer(unit: SchedulableUnit) -> WorkerObservation:
    return WorkerObservation.from_task_outcome(
        action_id=f"act-{unit.id}",
        subtask_id=unit.id,
        success=True,
        summary="ok",
        worker_session_id="worker-1",
        task_id=f"task-{unit.id}",
    )


def test_policy_classification() -> None:
    policy = GovernancePolicy()
    classification = HITLGovernor.classify_action(
        goal="deploy production migration",
        acceptance=["no downtime"],
        prior_failures=1,
        policy=policy,
    )

    assert classification.requires_approval
    assert classification.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


def test_risk_scoring_increases_with_failures() -> None:
    policy = GovernancePolicy()
    low = HITLGovernor.classify_action(
        goal="inspect logs",
        acceptance=["collect output"],
        prior_failures=0,
        policy=policy,
    )
    high = HITLGovernor.classify_action(
        goal="inspect logs",
        acceptance=["collect output"],
        prior_failures=4,
        policy=policy,
    )

    assert high.risk_score > low.risk_score


def test_approval_gating_blocks_without_decision(tmp_path: Path) -> None:
    governor = HITLGovernor(
        tmp_path,
        policy=GovernancePolicy(approval_threshold=0.2, max_escalations=1),
        operator_hook=lambda _request, _level: None,
    )

    decision = governor.evaluate(
        session_id="sess-1",
        unit_id="u-1",
        goal="deploy production migration",
        acceptance=["ac-1"],
        prior_failures=0,
    )

    assert not decision.allowed
    assert decision.approval_required
    assert decision.approval_status == ApprovalStatus.ESCALATED


def test_escalation_triggering(tmp_path: Path) -> None:
    calls: list[int] = []

    def no_decision(_request, level: int):
        calls.append(level)
        return None

    governor = HITLGovernor(
        tmp_path,
        policy=GovernancePolicy(approval_threshold=0.2, max_escalations=2),
        operator_hook=no_decision,
    )

    decision = governor.evaluate(
        session_id="sess-2",
        unit_id="u-2",
        goal="deploy production",
        acceptance=["ac-1"],
        prior_failures=1,
    )

    assert decision.approval_status == ApprovalStatus.ESCALATED
    assert decision.escalation_level == 2
    assert calls == [0, 1, 2]


def test_operator_override(tmp_path: Path) -> None:
    def approve_override(_request, _level: int):
        return ApprovalResolution(
            status=ApprovalStatus.APPROVED,
            actor_id="operator-1",
            reason="manual override",
            override=True,
        )

    governor = HITLGovernor(
        tmp_path,
        policy=GovernancePolicy(approval_threshold=0.1, allow_operator_override=True),
        operator_hook=approve_override,
    )

    decision = governor.evaluate(
        session_id="sess-3",
        unit_id="u-3",
        goal="deploy production",
        acceptance=["ac-1"],
        prior_failures=0,
    )

    assert decision.allowed
    assert decision.approval_status == ApprovalStatus.OVERRIDDEN


def test_reversible_action_markers() -> None:
    policy = GovernancePolicy()
    classification = HITLGovernor.classify_action(
        goal="dry-run inspect logs",
        acceptance=["read only"],
        prior_failures=0,
        policy=policy,
    )

    assert classification.reversible


def test_audit_log_integrity(tmp_path: Path) -> None:
    def approve(_request, _level: int):
        return ApprovalResolution(status=ApprovalStatus.APPROVED, actor_id="operator-2", reason="approved")

    governor = HITLGovernor(
        tmp_path,
        policy=GovernancePolicy(approval_threshold=0.2),
        operator_hook=approve,
    )

    governor.evaluate(
        session_id="sess-4",
        unit_id="u-4",
        goal="deploy production",
        acceptance=["ac-1"],
        prior_failures=0,
    )
    governor.evaluate(
        session_id="sess-4",
        unit_id="u-5",
        goal="delete old table",
        acceptance=["ac-1"],
        prior_failures=0,
    )

    assert governor.store.verify_integrity()
    records = governor.store.list()
    assert len(records) == 2
    assert records[0].record_hash
    assert records[1].prev_hash == records[0].record_hash


def test_safe_mode_enforcement(tmp_path: Path) -> None:
    governor = HITLGovernor(
        tmp_path,
        policy=GovernancePolicy(safe_mode=True, approval_threshold=1.0),
        operator_hook=lambda _request, _level: None,
    )

    decision = governor.evaluate(
        session_id="sess-5",
        unit_id="u-6",
        goal="inspect docs",
        acceptance=["ac-1"],
        prior_failures=0,
    )

    assert decision.approval_required
    assert not decision.allowed


def test_runtime_policy_enforcement_boundary_blocks_execution(tmp_path: Path) -> None:
    governor = HITLGovernor(
        tmp_path,
        policy=GovernancePolicy(approval_threshold=0.2, max_escalations=1),
        operator_hook=lambda _request, _level: None,
    )
    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-runtime-block",
        planner_fn=_planner,
        implementer_fn=_implementer,
        governance=governor,
    )

    result = __import__("asyncio").run(runtime.run("goal"))

    assert result.terminated_reason == "approval_blocked:u-1"
    assert result.approvals_required >= 1
    assert result.approvals_blocked >= 1
    assert any(event.event_type == "approval.gate" for event in result.events)


def test_runtime_approval_allows_with_operator_review(tmp_path: Path) -> None:
    def approve(_request, _level: int):
        return ApprovalResolution(status=ApprovalStatus.APPROVED, actor_id="operator-3", reason="approved")

    governor = HITLGovernor(
        tmp_path,
        policy=GovernancePolicy(approval_threshold=0.1),
        operator_hook=approve,
    )
    runtime = AutonomousCoordinatorRuntime(
        workspace=str(tmp_path),
        session_id="sess-runtime-allow",
        planner_fn=_planner,
        implementer_fn=_implementer,
        governance=governor,
    )

    result = __import__("asyncio").run(runtime.run("goal"))

    assert result.terminated_reason == "completed"
    assert "u-1" in result.observations
