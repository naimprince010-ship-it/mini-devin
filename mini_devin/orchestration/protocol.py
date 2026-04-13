"""
JSON-shaped messages between Supervisor (Orchestrator) and Workers.

OpenHands-style separation: structured Action → Worker runs tools → Observation back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PlanEvent:
    """Supervisor emits this after decomposing a large goal (event-driven planning)."""

    type: Literal["plan.created", "plan.updated"] = "plan.created"
    goal: str = ""
    workspace: str = ""
    subtasks: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    pivoted: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "goal": self.goal,
            "workspace": self.workspace,
            "subtasks": self.subtasks,
            "reasoning": self.reasoning,
            "pivoted": self.pivoted,
        }


@dataclass
class SupervisorAction:
    """JSON-RPC–style instruction from supervisor to runtime (dispatch only, no code)."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["worker.dispatch"] = "worker.dispatch"
    id: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "id": self.id,
            "params": self.params,
        }

    @classmethod
    def dispatch(cls, action_id: str, subtask_id: str, goal: str, acceptance: list[str]) -> SupervisorAction:
        return cls(
            id=action_id,
            params={
                "subtask_id": subtask_id,
                "goal": goal,
                "acceptance_criteria": acceptance,
            },
        )


@dataclass
class WorkerObservation:
    """Worker reply after executing assigned sub-goal (tools + agent loop live here)."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str = ""
    result: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"jsonrpc": self.jsonrpc, "id": self.id, "result": self.result}

    @property
    def success(self) -> bool:
        return bool(self.result.get("success"))

    @property
    def subtask_id(self) -> str:
        return str(self.result.get("subtask_id", ""))

    @classmethod
    def from_task_outcome(
        cls,
        action_id: str,
        subtask_id: str,
        *,
        success: bool,
        summary: str,
        worker_session_id: str,
        task_id: str,
        error: str | None = None,
    ) -> WorkerObservation:
        status = "completed" if success else "failed"
        return cls(
            id=action_id,
            result={
                "subtask_id": subtask_id,
                "success": success,
                "status": status,
                "summary": summary,
                "worker_session_id": worker_session_id,
                "task_id": task_id,
                "error": error,
            },
        )
