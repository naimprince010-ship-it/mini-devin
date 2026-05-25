"""Durable checkpoint abstraction for orchestrator state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, runtime_checkable


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Serialized checkpoint state that can be stored durably."""

    checkpoint_id: str
    scope_id: str
    created_at: datetime = field(default_factory=_utcnow)
    state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DurableCheckpointStore(Protocol):
    """Persistence interface for checkpoint snapshots."""

    def save(self, checkpoint: CheckpointRecord) -> None:
        ...

    def load(self, checkpoint_id: str) -> CheckpointRecord | None:
        ...

    def list(self, scope_id: str | None = None) -> list[CheckpointRecord]:
        ...

    def delete(self, checkpoint_id: str) -> None:
        ...
"""Durable checkpoint abstractions for orchestrator recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class CheckpointRecord:
    """Metadata needed to recreate or restore a task checkpoint."""

    checkpoint_id: str
    session_id: str
    task_id: str
    created_at: datetime
    resource_ref: str
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DurableCheckpointStore(Protocol):
    """Persist checkpoints in a durable backing store."""

    def save(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        ...

    def load(self, checkpoint_id: str) -> CheckpointRecord | None:
        ...

    def list_for_session(self, session_id: str) -> list[CheckpointRecord]:
        ...

    def delete(self, checkpoint_id: str) -> bool:
        ...
