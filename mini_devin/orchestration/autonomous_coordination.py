"""Feature-flagged autonomous planning and multi-agent coordination runtime.

This module is additive and bounded. It introduces a local coordinator graph with
planner/implementer/verifier/recovery roles, shared blackboard state, event messaging,
budget coordination, and replan hooks while preserving existing single-agent behavior
when disabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping

from mini_devin.contracts.protocols import DurableCheckpoint
from mini_devin.core.model_gateway import ModelGateway

from .checkpoint_store import JsonlCheckpointStore
from .execution_learning import (
    ExecutionLearningMemory,
    LearningSignal,
    build_failure_fingerprint,
    execution_learning_enabled,
    score_task_outcome_quality,
)
from .hitl_governance import ApprovalStatus, HITLGovernor, GovernanceDecision, governance_enabled
from .observability import TimelineRecord, emit_worker_metric, record_timeline_event
from .protocol import SupervisorAction, WorkerObservation
from .runtime_contracts import FileTypedEventEmitter
from .task_scheduler import SchedulableUnit


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def autonomous_coordination_enabled() -> bool:
    return _flag_enabled("PLODDER_AUTONOMOUS_COORDINATION")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    VERIFIER = "verifier"
    RECOVERY = "recovery"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REPLAN = "replan"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class AgentMessage:
    source: AgentRole
    target: AgentRole
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "target": self.target.value,
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "ts": self.ts.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    replacement_subtasks: tuple[SchedulableUnit, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentBudgetPolicy:
    max_total_actions: int = 50
    max_replans: int = 5
    max_recovery_escalations: int = 8
    max_failures: int = 12
    per_role_limits: Mapping[AgentRole, int] = field(
        default_factory=lambda: {
            AgentRole.PLANNER: 10,
            AgentRole.IMPLEMENTER: 40,
            AgentRole.VERIFIER: 40,
            AgentRole.RECOVERY: 16,
        }
    )


@dataclass(slots=True)
class AgentBudgetState:
    policy: AgentBudgetPolicy = field(default_factory=AgentBudgetPolicy)
    total_actions: int = 0
    replans: int = 0
    recovery_escalations: int = 0
    failures: int = 0
    role_actions: dict[AgentRole, int] = field(default_factory=dict)

    def can_consume(self, role: AgentRole) -> bool:
        if self.total_actions >= self.policy.max_total_actions:
            return False
        limit = int(self.policy.per_role_limits.get(role, self.policy.max_total_actions))
        return self.role_actions.get(role, 0) < limit

    def consume(self, role: AgentRole) -> bool:
        if not self.can_consume(role):
            return False
        self.total_actions += 1
        self.role_actions[role] = self.role_actions.get(role, 0) + 1
        return True

    def note_replan(self) -> bool:
        if self.replans >= self.policy.max_replans:
            return False
        self.replans += 1
        return True

    def note_recovery(self) -> bool:
        if self.recovery_escalations >= self.policy.max_recovery_escalations:
            return False
        self.recovery_escalations += 1
        return True

    def note_failure(self) -> bool:
        self.failures += 1
        return self.failures <= self.policy.max_failures


@dataclass(slots=True)
class SharedBlackboard:
    goal: str
    workspace: str
    session_id: str
    plan_reasoning: str = ""
    queued_units: list[SchedulableUnit] = field(default_factory=list)
    completed_units: set[str] = field(default_factory=set)
    failed_units: set[str] = field(default_factory=set)
    observations: dict[str, WorkerObservation] = field(default_factory=dict)
    verification_log: dict[str, bool] = field(default_factory=dict)
    messages: list[AgentMessage] = field(default_factory=list)
    version: int = 0

    def add_message(self, msg: AgentMessage) -> None:
        self.messages.append(msg)
        self.version += 1

    def synchronize(self) -> int:
        self.version += 1
        return self.version


@dataclass(frozen=True, slots=True)
class AutonomousCoordinationConfig:
    max_cycles: int = 64
    replan_on_verifier_failure: bool = True
    stop_on_unverifiable_failure: bool = False


@dataclass(slots=True)
class AutonomousCoordinationResult:
    observations: dict[str, WorkerObservation] = field(default_factory=dict)
    actions: list[SupervisorAction] = field(default_factory=list)
    events: list[AgentMessage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    replans: int = 0
    terminated_reason: str | None = None
    cycles: int = 0
    blackboard_version: int = 0
    learned_entries: int = 0
    strategy_adaptations: int = 0
    approvals_required: int = 0
    approvals_blocked: int = 0
    approvals_overridden: int = 0


PlannerFn = Callable[[str, str], Awaitable[tuple[list[SchedulableUnit], str]]]
ImplementerFn = Callable[[SchedulableUnit], Awaitable[WorkerObservation]]
VerifierFn = Callable[[SchedulableUnit, WorkerObservation], Awaitable[bool]]
RecoveryFn = Callable[[SchedulableUnit, WorkerObservation, int], Awaitable[RecoveryDecision]]
ReplanFn = Callable[[SchedulableUnit, WorkerObservation, SharedBlackboard], Awaitable[list[SchedulableUnit]]]


class AutonomousCoordinatorRuntime:
    """Bounded multi-agent coordinator runtime.

    Integrates with queue/sandbox/model-gateway/checkpoint/observability layers through
    light-weight hooks and event recording.
    """

    def __init__(
        self,
        *,
        workspace: str,
        session_id: str,
        planner_fn: PlannerFn,
        implementer_fn: ImplementerFn,
        verifier_fn: VerifierFn | None = None,
        recovery_fn: RecoveryFn | None = None,
        replan_fn: ReplanFn | None = None,
        checkpoint_store: JsonlCheckpointStore | None = None,
        typed_emitter: FileTypedEventEmitter | None = None,
        model_gateway: ModelGateway | None = None,
        queue_boundary: Any | None = None,
        sandbox_boundary: Any | None = None,
        config: AutonomousCoordinationConfig | None = None,
        budget_state: AgentBudgetState | None = None,
        learning_memory: ExecutionLearningMemory | None = None,
        governance: HITLGovernor | None = None,
    ) -> None:
        self.workspace = workspace
        self.session_id = session_id
        self.planner_fn = planner_fn
        self.implementer_fn = implementer_fn
        self.verifier_fn = verifier_fn or self._default_verifier
        self.recovery_fn = recovery_fn or self._default_recovery
        self.replan_fn = replan_fn
        self.checkpoint_store = checkpoint_store or JsonlCheckpointStore(workspace)
        self.typed_emitter = typed_emitter or FileTypedEventEmitter(workspace)
        self.model_gateway = model_gateway
        self.queue_boundary = queue_boundary
        self.sandbox_boundary = sandbox_boundary
        self.config = config or AutonomousCoordinationConfig()
        self.budget = budget_state or AgentBudgetState()
        self.learning_memory = learning_memory
        if self.learning_memory is None and execution_learning_enabled():
            self.learning_memory = ExecutionLearningMemory(workspace)
        self.governance = governance
        if self.governance is None and governance_enabled():
            self.governance = HITLGovernor(workspace)

    async def _default_verifier(self, unit: SchedulableUnit, obs: WorkerObservation) -> bool:
        del unit
        return bool(obs.success)

    async def _default_recovery(
        self,
        unit: SchedulableUnit,
        obs: WorkerObservation,
        failure_count: int,
    ) -> RecoveryDecision:
        del unit, obs
        if failure_count <= 1:
            return RecoveryDecision(action=RecoveryAction.RETRY, reason="first_failure_retry")
        if self.replan_fn is not None and self.config.replan_on_verifier_failure:
            return RecoveryDecision(action=RecoveryAction.REPLAN, reason="trigger_replan")
        if self.config.stop_on_unverifiable_failure:
            return RecoveryDecision(action=RecoveryAction.ABORT, reason="verifier_gate_failed")
        return RecoveryDecision(action=RecoveryAction.RETRY, reason="bounded_retry")

    def _emit_message(self, board: SharedBlackboard, msg: AgentMessage) -> None:
        board.add_message(msg)
        payload = msg.to_dict()
        self.typed_emitter.emit({"event_type": "coordination.message", **payload})
        record_timeline_event(
            self.workspace,
            TimelineRecord(
                event_type="coordination.message",
                source="coordination",
                session_id=board.session_id,
                task_id=payload["payload"].get("unit_id") if isinstance(payload.get("payload"), dict) else None,
                unit_id=payload["payload"].get("unit_id") if isinstance(payload.get("payload"), dict) else None,
                status=payload["event_type"],
                payload=payload,
            ),
        )

    def _checkpoint_blackboard(self, board: SharedBlackboard, suffix: str) -> None:
        self.checkpoint_store.save(
            DurableCheckpoint(
                checkpoint_id=f"coordination:{board.session_id}:{suffix}",
                scope_id=board.session_id,
                state={
                    "goal": board.goal,
                    "version": board.version,
                    "queued_units": [u.id for u in board.queued_units],
                    "completed_units": sorted(board.completed_units),
                    "failed_units": sorted(board.failed_units),
                    "verification_log": dict(board.verification_log),
                },
                metadata={
                    "messages": len(board.messages),
                    "queue_boundary": bool(self.queue_boundary is not None),
                    "sandbox_boundary": bool(self.sandbox_boundary is not None),
                    "model_gateway": bool(self.model_gateway is not None),
                },
            )
        )

    @staticmethod
    def _next_runnable_unit(board: SharedBlackboard) -> SchedulableUnit | None:
        for unit in board.queued_units:
            if unit.id in board.completed_units or unit.id in board.failed_units:
                continue
            if all(dep in board.completed_units for dep in unit.depends_on):
                return unit
        return None

    async def _replan(self, board: SharedBlackboard, unit: SchedulableUnit, obs: WorkerObservation) -> list[SchedulableUnit]:
        if self.replan_fn is None:
            return []
        replanned = await self.replan_fn(unit, obs, board)
        return [u for u in replanned if isinstance(u, SchedulableUnit)]

    def _failure_fingerprint(self, unit: SchedulableUnit, obs: WorkerObservation) -> str:
        return build_failure_fingerprint(
            strategy_key=unit.goal,
            error=str(obs.result.get("error") or ""),
            status=str(obs.result.get("status") or ""),
        )

    def _learn_outcome(
        self,
        *,
        unit: SchedulableUnit,
        obs: WorkerObservation,
        verified: bool,
        replay_driven: bool = False,
    ) -> int:
        if self.learning_memory is None:
            return 0
        quality = score_task_outcome_quality(
            success=bool(obs.success),
            verifier_passed=verified,
            summary=str(obs.result.get("summary") or ""),
            error=obs.result.get("error"),
        )
        self.learning_memory.remember(
            LearningSignal(
                session_id=self.session_id,
                unit_id=unit.id,
                strategy_key=unit.goal,
                fingerprint=self._failure_fingerprint(unit, obs),
                success=bool(obs.success),
                quality=quality,
                verifier_passed=verified,
                replay_driven=replay_driven,
                metadata={
                    "acceptance": list(unit.acceptance_criteria),
                    "status": obs.result.get("status"),
                },
            )
        )
        return 1

    def _learned_recovery_adaptation(self, unit: SchedulableUnit, obs: WorkerObservation, decision: RecoveryDecision) -> RecoveryDecision:
        if self.learning_memory is None:
            return decision
        if decision.action != RecoveryAction.RETRY:
            return decision
        pattern = self.learning_memory.retrieve_failure_pattern(self._failure_fingerprint(unit, obs))
        if pattern is None:
            return decision
        if pattern.outcome != "failure":
            return decision
        if pattern.retention_score() >= -0.3:
            return decision
        if self.replan_fn is not None and self.config.replan_on_verifier_failure:
            return RecoveryDecision(action=RecoveryAction.REPLAN, reason="learned_failure_pattern")
        return decision

    def _checkpoint_approval_gate(self, board: SharedBlackboard, unit: SchedulableUnit, decision: GovernanceDecision) -> None:
        self.checkpoint_store.save(
            DurableCheckpoint(
                checkpoint_id=f"approval:{board.session_id}:{unit.id}:{board.version}",
                scope_id=board.session_id,
                state={
                    "unit_id": unit.id,
                    "goal": unit.goal,
                    "approval_required": decision.approval_required,
                    "approval_status": decision.approval_status.value if decision.approval_status else None,
                    "allowed": decision.allowed,
                    "risk_score": decision.classification.risk_score,
                    "risk_level": decision.classification.risk_level.value,
                    "reversible": decision.reversible_marker,
                },
                metadata={
                    "escalation_level": decision.escalation_level,
                    "reason": decision.reason,
                    "request_id": decision.request_id,
                },
            )
        )

    async def run(self, goal: str) -> AutonomousCoordinationResult:
        result = AutonomousCoordinationResult()
        if self.learning_memory is not None:
            result.learned_entries += self.learning_memory.ingest_replay_learning(max_lines=300)
        units, reasoning = await self.planner_fn(goal, self.workspace)
        board = SharedBlackboard(
            goal=goal,
            workspace=self.workspace,
            session_id=self.session_id,
            plan_reasoning=reasoning,
            queued_units=list(units),
        )

        self._emit_message(
            board,
            AgentMessage(
                source=AgentRole.COORDINATOR,
                target=AgentRole.PLANNER,
                event_type="decomposition.completed",
                payload={"units": [u.id for u in units], "reasoning": reasoning},
            ),
        )
        self._checkpoint_blackboard(board, "initial")

        failure_counts: dict[str, int] = {}

        for cycle in range(1, self.config.max_cycles + 1):
            result.cycles = cycle
            if not self.budget.consume(AgentRole.COORDINATOR):
                result.terminated_reason = "budget_exhausted:coordinator"
                break

            unit = self._next_runnable_unit(board)
            if unit is None:
                if all(u.id in board.completed_units or u.id in board.failed_units for u in board.queued_units):
                    result.terminated_reason = "completed"
                else:
                    result.terminated_reason = "bounded_deadlock"
                    result.errors.append("No runnable units within bounded graph")
                break

            if self.governance is not None:
                gate = self.governance.evaluate(
                    session_id=board.session_id,
                    unit_id=unit.id,
                    goal=unit.goal,
                    acceptance=unit.acceptance_criteria,
                    prior_failures=failure_counts.get(unit.id, 0),
                )
                if gate.approval_required:
                    result.approvals_required += 1
                if gate.approval_status == ApprovalStatus.OVERRIDDEN:
                    result.approvals_overridden += 1

                self._emit_message(
                    board,
                    AgentMessage(
                        source=AgentRole.COORDINATOR,
                        target=AgentRole.IMPLEMENTER,
                        event_type="approval.gate",
                        payload={
                            "unit_id": unit.id,
                            "request_id": gate.request_id,
                            "allowed": gate.allowed,
                            "approval_required": gate.approval_required,
                            "approval_status": gate.approval_status.value if gate.approval_status else "none",
                            "risk_score": gate.classification.risk_score,
                            "risk_level": gate.classification.risk_level.value,
                            "reversible": gate.reversible_marker,
                            "reason": gate.reason,
                            "escalation_level": gate.escalation_level,
                        },
                    ),
                )
                self._checkpoint_approval_gate(board, unit, gate)
                emit_worker_metric(
                    self.workspace,
                    "worker.approval.gate",
                    1.0,
                    labels={
                        "unit_id": unit.id,
                        "allowed": str(gate.allowed).lower(),
                        "risk_level": gate.classification.risk_level.value,
                    },
                )
                if not gate.allowed:
                    board.failed_units.add(unit.id)
                    result.approvals_blocked += 1
                    result.terminated_reason = f"approval_blocked:{unit.id}"
                    result.errors.append(f"Approval gate blocked execution for {unit.id}: {gate.reason}")
                    break

            strategy_candidates = []
            if self.learning_memory is not None:
                strategy_candidates = self.learning_memory.retrieve_strategies(strategy_hint=unit.goal, limit=1)
                if strategy_candidates:
                    result.strategy_adaptations += 1
                    self._emit_message(
                        board,
                        AgentMessage(
                            source=AgentRole.COORDINATOR,
                            target=AgentRole.PLANNER,
                            event_type="strategy.adapted",
                            payload={
                                "unit_id": unit.id,
                                "strategy": strategy_candidates[0].strategy_key,
                                "score": strategy_candidates[0].score,
                            },
                        ),
                    )

            self._emit_message(
                board,
                AgentMessage(
                    source=AgentRole.PLANNER,
                    target=AgentRole.IMPLEMENTER,
                    event_type="unit.dispatched",
                    payload={
                        "unit_id": unit.id,
                        "goal": unit.goal,
                        "strategy_hint": strategy_candidates[0].strategy_key if strategy_candidates else "",
                        "reversible": bool(self.governance and self.governance.classify_action(
                            goal=unit.goal,
                            acceptance=unit.acceptance_criteria,
                            prior_failures=failure_counts.get(unit.id, 0),
                            policy=self.governance.policy,
                        ).reversible),
                    },
                ),
            )

            if not self.budget.consume(AgentRole.IMPLEMENTER):
                result.terminated_reason = "budget_exhausted:implementer"
                break

            obs = await self.implementer_fn(unit)
            board.observations[unit.id] = obs
            result.observations[unit.id] = obs

            self._emit_message(
                board,
                AgentMessage(
                    source=AgentRole.IMPLEMENTER,
                    target=AgentRole.VERIFIER,
                    event_type="unit.implemented",
                    payload={"unit_id": unit.id, "success": obs.success},
                ),
            )

            if not self.budget.consume(AgentRole.VERIFIER):
                result.terminated_reason = "budget_exhausted:verifier"
                break

            verified = await self.verifier_fn(unit, obs)
            board.verification_log[unit.id] = verified
            result.learned_entries += self._learn_outcome(unit=unit, obs=obs, verified=verified)

            if verified:
                board.completed_units.add(unit.id)
                self._emit_message(
                    board,
                    AgentMessage(
                        source=AgentRole.VERIFIER,
                        target=AgentRole.COORDINATOR,
                        event_type="unit.promoted",
                        payload={"unit_id": unit.id},
                    ),
                )
                emit_worker_metric(
                    self.workspace,
                    "worker.coordination.promoted",
                    1.0,
                    labels={"unit_id": unit.id},
                )
            else:
                board.failed_units.add(unit.id)
                failure_counts[unit.id] = failure_counts.get(unit.id, 0) + 1
                still_allowed = self.budget.note_failure()
                if not still_allowed:
                    result.terminated_reason = "budget_exhausted:failures"
                    result.errors.append("Failure budget exhausted")
                    break

                if not self.budget.consume(AgentRole.RECOVERY):
                    result.terminated_reason = "budget_exhausted:recovery"
                    break
                if not self.budget.note_recovery():
                    result.terminated_reason = "budget_exhausted:recovery_escalations"
                    break

                decision = await self.recovery_fn(unit, obs, failure_counts[unit.id])
                adapted_decision = self._learned_recovery_adaptation(unit, obs, decision)
                if adapted_decision.action != decision.action or adapted_decision.reason != decision.reason:
                    result.strategy_adaptations += 1
                decision = adapted_decision
                self._emit_message(
                    board,
                    AgentMessage(
                        source=AgentRole.RECOVERY,
                        target=AgentRole.COORDINATOR,
                        event_type="recovery.decision",
                        payload={
                            "unit_id": unit.id,
                            "action": decision.action.value,
                            "reason": decision.reason,
                        },
                    ),
                )

                if decision.action == RecoveryAction.ABORT:
                    result.terminated_reason = f"aborted:{unit.id}:{decision.reason}"
                    break
                if decision.action == RecoveryAction.RETRY:
                    board.failed_units.discard(unit.id)
                elif decision.action == RecoveryAction.REPLAN:
                    if not self.budget.note_replan():
                        result.terminated_reason = "budget_exhausted:replans"
                        break
                    replanned = list(decision.replacement_subtasks)
                    if not replanned:
                        replanned = await self._replan(board, unit, obs)
                    if replanned:
                        known = {u.id for u in board.queued_units}
                        for candidate in replanned:
                            if candidate.id not in known:
                                board.queued_units.append(candidate)
                                known.add(candidate.id)
                        result.replans += 1
                        self._emit_message(
                            board,
                            AgentMessage(
                                source=AgentRole.PLANNER,
                                target=AgentRole.COORDINATOR,
                                event_type="replan.applied",
                                payload={"unit_id": unit.id, "added": [u.id for u in replanned]},
                            ),
                        )

            board.synchronize()
            self._checkpoint_blackboard(board, f"cycle-{cycle}")

        if result.terminated_reason is None:
            result.terminated_reason = "max_cycles_reached"

        result.events = list(board.messages)
        result.blackboard_version = board.version
        result.actions = [
            SupervisorAction.dispatch(action_id=f"coord-{idx}", subtask_id=u.id, goal=u.goal, acceptance=u.acceptance_criteria)
            for idx, u in enumerate(board.queued_units, start=1)
        ]
        return result
