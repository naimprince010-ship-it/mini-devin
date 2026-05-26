"""Bootstrap scaffold for future worker processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class WorkerBootstrapConfig:
    """Configuration needed to start a worker process later."""

    session_id: str
    workspace: Path
    worker_id: str = ""
    queue_name: str = "plodder.worker"
    redis_url: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkerBootstrapSkeleton:
    """Inert bootstrap scaffold; intentionally does not start runtime behavior yet."""

    config: WorkerBootstrapConfig

    def describe(self) -> dict[str, Any]:
        return {
            "session_id": self.config.session_id,
            "workspace": str(self.config.workspace),
            "worker_id": self.config.worker_id,
            "queue_name": self.config.queue_name,
            "redis_url": bool(self.config.redis_url),
            "metadata": dict(self.config.metadata),
        }

    def bootstrap(self) -> dict[str, Any]:
        return {
            "started": False,
            "reason": "worker bootstrap is scaffold-only in phase 1",
            "config": self.describe(),
        }
"""Worker bootstrap skeleton.

This module intentionally does not start real workers yet. It only defines the
configuration and lifecycle surface needed for the Phase 1 split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class WorkerBootstrapConfig:
    """Inputs needed to prepare a worker process in later phases."""

    workspace: Path
    session_id: str
    task_id: str
    checkpoint_id: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class WorkerBootstrap(Protocol):
    """Skeleton lifecycle for a worker bootstrapper."""

    def build_environment(self) -> dict[str, str]:
        ...

    def build_launch_payload(self) -> dict[str, Any]:
        ...

    def launch(self) -> Any:
        ...

    def shutdown(self) -> None:
        ...


class WorkerBootstrapSkeleton:
    """No-op implementation for the future worker bootstrapper."""

    def __init__(self, config: WorkerBootstrapConfig) -> None:
        self.config = config

    def build_environment(self) -> dict[str, str]:
        env = dict(self.config.environment)
        env.setdefault("PLODDER_SESSION_ID", self.config.session_id)
        env.setdefault("PLODDER_TASK_ID", self.config.task_id)
        return env

    def build_launch_payload(self) -> dict[str, Any]:
        return {
            "workspace": str(self.config.workspace),
            "session_id": self.config.session_id,
            "task_id": self.config.task_id,
            "checkpoint_id": self.config.checkpoint_id,
            "metadata": dict(self.config.metadata),
        }

    def launch(self) -> Any:
        raise NotImplementedError("Worker bootstrap launch is not wired yet.")

    def shutdown(self) -> None:
        raise NotImplementedError("Worker bootstrap shutdown is not wired yet.")


def create_worker_bootstrap(config: WorkerBootstrapConfig) -> WorkerBootstrapSkeleton:
    return WorkerBootstrapSkeleton(config)
