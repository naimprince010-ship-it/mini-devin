"""
Runtime isolation for workers (OpenHands Runtime mindset).

Each worker gets a **fresh agent session** (clean LLM conversation + cancel channel) while
sharing the **same workspace directory** on disk so edits remain visible to all workers.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from .sandbox_runtime import (
    SandboxIsolationCoordinator,
    SandboxResourceLimits,
    TaskSandboxContext,
    TaskSecretScope,
    build_default_sandbox_limits,
    sandbox_hardening_enabled,
)


def orchestrator_worker_use_sandbox() -> bool:
    return (os.environ.get("ORCHESTRATOR_WORKER_USE_SANDBOX", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def orchestrator_worker_sandbox_hardening_enabled() -> bool:
    return sandbox_hardening_enabled()


class WorkerRuntime:
    """
    Thin factory for worker sessions — analogous to ``Runtime`` + sandbox hooks in OpenHands.

    We do not start a new Docker container per sub-task by default (heavy on Railway); callers
    can enable ``use_sandbox`` via ``ORCHESTRATOR_WORKER_USE_SANDBOX=1`` to match container isolation.
    """

    @staticmethod
    async def create_worker_session(
        session_manager: Any,
        *,
        shared_workspace: str,
        model: str | None = None,
        max_iterations: int | None = None,
        use_sandbox: bool | None = None,
    ) -> Any:
        """
        Spawn a new DB-backed session whose Agent uses ``shared_workspace`` as cwd.

        Returns the same ``Session`` type the rest of Plodder uses (``DatabaseSessionManager``).
        """
        ws = os.path.abspath(os.path.expanduser(shared_workspace))
        sb = orchestrator_worker_use_sandbox() if use_sandbox is None else bool(use_sandbox)

        kwargs: dict[str, Any] = {
            "working_directory": ws,
            "use_sandbox": sb,
        }
        if model:
            kwargs["model"] = model
        if max_iterations is not None:
            kwargs["max_iterations"] = max_iterations

        return await session_manager.create_session(**kwargs)

    @staticmethod
    def build_sandbox_limits(*, cpu_cores: float = 1.0, memory_mb: int = 512, timeout_seconds: int = 300) -> SandboxResourceLimits:
        return build_default_sandbox_limits(cpu_cores=cpu_cores, memory_mb=memory_mb, timeout_seconds=timeout_seconds)

    @staticmethod
    def prepare_task_sandbox(
        workspace: str,
        *,
        task_id: str,
        session_id: str,
        limits: SandboxResourceLimits | None = None,
        secret_scope: TaskSecretScope | None = None,
        checkpoint_store: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskSandboxContext:
        coordinator = SandboxIsolationCoordinator(workspace, checkpoint_store=checkpoint_store)
        return coordinator.prepare(
            task_id=task_id,
            session_id=session_id,
            limits=limits,
            secret_scope=secret_scope,
            metadata=metadata,
        )

    @staticmethod
    def build_task_sandbox_environment(
        workspace: str,
        *,
        task_id: str,
        session_id: str,
        base_environment: dict[str, str] | None = None,
        limits: SandboxResourceLimits | None = None,
        secret_scope: TaskSecretScope | None = None,
    ) -> dict[str, str]:
        coordinator = SandboxIsolationCoordinator(workspace)
        context = coordinator.prepare(
            task_id=task_id,
            session_id=session_id,
            limits=limits,
            secret_scope=secret_scope,
        )
        return coordinator.build_worker_environment(context, base_environment=base_environment)

    @staticmethod
    def new_action_id() -> str:
        return f"act-{uuid.uuid4().hex[:12]}"
