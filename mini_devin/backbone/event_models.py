"""
Pydantic-first Action / Observation events for the modular backbone.

These types are the **contract** between policy (LLM / AgentController) and
execution (Runtime). The legacy :class:`~mini_devin.orchestrator.agent.Agent`
continues to use tools directly until bridged here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EventEnvelope(BaseModel):
    """Common metadata for every published event."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: datetime = Field(default_factory=_utcnow)
    causation_id: str | None = None
    """Set on observations to reference the triggering action ``id``."""


# --- Actions (intent; runtime must not mutate these) ---------------------------------


class CmdRunAction(EventEnvelope):
    """Request to run a shell command (maps to ``terminal`` tool semantics)."""

    type: Literal["CmdRunAction"] = "CmdRunAction"
    command: str
    cwd: str | None = None
    timeout_sec: int | None = Field(default=30, ge=1, le=3600)


class FileWriteAction(EventEnvelope):
    """Request to write or replace file contents."""

    type: Literal["FileWriteAction"] = "FileWriteAction"
    path: str
    content: str


class GenericToolAction(EventEnvelope):
    """Bridge: arbitrary named tool + JSON arguments (migration path from current tools)."""

    type: Literal["GenericToolAction"] = "GenericToolAction"
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


Action = Annotated[
    Union[CmdRunAction, FileWriteAction, GenericToolAction],
    Field(discriminator="type"),
]


# --- Observations (facts produced by runtime) ----------------------------------------


class CommandOutputObservation(EventEnvelope):
    """Stdout/stderr/exit from a command."""

    type: Literal["CommandOutputObservation"] = "CommandOutputObservation"
    source_action_id: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class FileWriteObservation(EventEnvelope):
    """Acknowledgement / metadata after a file write."""

    type: Literal["FileWriteObservation"] = "FileWriteObservation"
    source_action_id: str
    path: str
    bytes_written: int = 0
    success: bool = True
    error: str | None = None


class ErrorObservation(EventEnvelope):
    """Runtime or validation failure."""

    type: Literal["ErrorObservation"] = "ErrorObservation"
    source_action_id: str | None = None
    message: str
    error_type: str = "runtime"


class GenericToolObservation(EventEnvelope):
    """Structured tool result (string body + optional success flag)."""

    type: Literal["GenericToolObservation"] = "GenericToolObservation"
    source_action_id: str
    tool_name: str
    content: str
    success: bool = True


Observation = Annotated[
    Union[
        CommandOutputObservation,
        FileWriteObservation,
        ErrorObservation,
        GenericToolObservation,
    ],
    Field(discriminator="type"),
]


class AgentMessageEvent(EventEnvelope):
    """Natural-language user/assistant/system line (optional in backbone timeline)."""

    type: Literal["AgentMessageEvent"] = "AgentMessageEvent"
    role: Literal["user", "assistant", "system"]
    content: str


AgentEvent = Annotated[
    Union[
        CmdRunAction,
        FileWriteAction,
        GenericToolAction,
        CommandOutputObservation,
        FileWriteObservation,
        ErrorObservation,
        GenericToolObservation,
        AgentMessageEvent,
    ],
    Field(discriminator="type"),
]

agent_event_adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)


def parse_agent_event(data: dict[str, Any]) -> AgentEvent:
    """Validate a JSON dict into a concrete :class:`AgentEvent`."""
    return agent_event_adapter.validate_python(data)


def is_action_event(event: AgentEvent) -> bool:
    return event.type in ("CmdRunAction", "FileWriteAction", "GenericToolAction")  # type: ignore[union-attr]


def is_observation_event(event: AgentEvent) -> bool:
    return event.type in (
        "CommandOutputObservation",
        "FileWriteObservation",
        "ErrorObservation",
        "GenericToolObservation",
    )  # type: ignore[union-attr]
