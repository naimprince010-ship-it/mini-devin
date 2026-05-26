"""Shared protocol definitions for runtime coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, runtime_checkable


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class DurableCheckpoint:
    """Serializable checkpoint payload for durable task recovery."""

    checkpoint_id: str
    scope_id: str
    created_at: datetime = field(default_factory=utcnow)
    state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TypedEventEmitter(Protocol):
    """Minimal typed event emitter interface."""

    def emit(self, event: Mapping[str, Any]) -> None:
        ...


@runtime_checkable
class DurableCheckpointStore(Protocol):
    """Persistence boundary for task checkpoints."""

    def save(self, checkpoint: DurableCheckpoint) -> None:
        ...

    def load(self, checkpoint_id: str) -> DurableCheckpoint | None:
        ...

    def list(self, scope_id: str | None = None) -> list[DurableCheckpoint]:
        ...

    def delete(self, checkpoint_id: str) -> None:
        ...
