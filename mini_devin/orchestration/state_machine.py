"""Orchestrator state machine interface for the v2 runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class OrchestratorPhase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StateTransition:
    """One state transition emitted by the orchestrator."""

    from_state: str
    to_state: str
    event_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class OrchestratorStateMachine(Protocol):
    """Minimal control-plane contract for task progression."""

    current_state: str

    def allowed_transitions(self) -> Mapping[str, tuple[str, ...]]:
        ...

    def can_transition(self, to_state: str) -> bool:
        ...

    def transition(self, to_state: str, *, event_name: str, metadata: Mapping[str, Any] | None = None) -> str:
        ...

    def snapshot(self) -> dict[str, Any]:
        ...
"""Orchestrator state machine interface and snapshot types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class OrchestratorPhase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    RUNNING = "running"
    VERIFYING = "verifying"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class OrchestratorStateSnapshot:
    """Durable view of a task/session state."""

    session_id: str
    task_id: str
    phase: OrchestratorPhase
    last_event_id: str | None = None
    checkpoint_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class OrchestratorStateMachine(Protocol):
    """Minimal interface for a durable orchestrator workflow state machine."""

    def current_state(self) -> OrchestratorStateSnapshot:
        ...

    def can_transition(self, next_phase: OrchestratorPhase) -> bool:
        ...

    def transition_to(
        self,
        next_phase: OrchestratorPhase,
        *,
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> OrchestratorStateSnapshot:
        ...

    def restore_checkpoint(self, checkpoint_id: str) -> OrchestratorStateSnapshot:
        ...
