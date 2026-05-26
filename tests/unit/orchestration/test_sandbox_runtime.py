from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_devin.orchestration.checkpoint_store import JsonlCheckpointStore
from mini_devin.orchestration.sandbox_runtime import (
    SandboxIsolationCoordinator,
    SandboxPhase,
    SandboxResourceLimits,
    TaskSecretScope,
)
from mini_devin.orchestration.worker_runtime import WorkerRuntime


@pytest.fixture(autouse=True)
def _sandbox_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PLODDER_WORKER_SANDBOX_HARDENING",
        "PLODDER_SANDBOX_AUDIT",
        "PLODDER_SANDBOX_QUARANTINE",
        "PLODDER_WORKER_CRASH_RECOVERY",
        "PLODDER_FS_ISOLATION",
        "PLODDER_TASK_SECRET_SCOPING",
        "PLODDER_OBSERVABILITY",
        "PLODDER_TIMELINE_RECORDING",
        "PLODDER_WORKER_METRICS",
    ):
        monkeypatch.setenv(name, "1")


def test_sandbox_lifecycle_transitions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    coordinator = SandboxIsolationCoordinator(workspace)

    prepared = WorkerRuntime.prepare_task_sandbox(
        str(workspace),
        task_id="task-1",
        session_id="sess-1",
        metadata={"queue": "plodder.task_queue"},
    )
    assert prepared.phase == SandboxPhase.PREPARED
    assert prepared.workspace_dir.exists()

    active = coordinator.activate(prepared)
    released = coordinator.release(active)

    assert active.phase == SandboxPhase.ACTIVE
    assert released.phase == SandboxPhase.RELEASED

    checkpoint = JsonlCheckpointStore(workspace).load("sandbox:sess-1:task-1")
    assert checkpoint is not None
    assert checkpoint.state["phase"] == "released"


def test_resource_limit_enforcement() -> None:
    limits = SandboxResourceLimits(cpu_cores=2.0, memory_mb=1024, timeout_seconds=30, pids_limit=128)
    limits.validate()

    assert limits.clamp_timeout(120) == 30
    assert limits.clamp_timeout(10) == 10

    with pytest.raises(ValueError):
        SandboxResourceLimits(cpu_cores=0, memory_mb=1024, timeout_seconds=30, pids_limit=128).validate()


def test_audit_log_creation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    coordinator = SandboxIsolationCoordinator(workspace)

    ctx = coordinator.prepare(task_id="task-1", session_id="sess-1")
    coordinator.activate(ctx)

    audit_path = workspace / ".plodder" / "sandboxes" / "task-1" / "audit.jsonl"
    assert audit_path.is_file()

    entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8" ).splitlines() if line.strip()]
    assert [entry["event_type"] for entry in entries] == ["sandbox.prepared", "sandbox.activated"]


def test_quarantine_and_recovery_transitions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    coordinator = SandboxIsolationCoordinator(workspace)

    ctx = coordinator.prepare(task_id="task-1", session_id="sess-1")
    quarantined = coordinator.record_crash(ctx, reason="worker exited unexpectedly", exit_code=137)
    recovered = coordinator.recover_quarantined_context(quarantined)

    assert quarantined.phase == SandboxPhase.QUARANTINED
    assert quarantined.crash_count == 1
    assert recovered.phase == SandboxPhase.ACTIVE
    assert recovered.last_reason == "worker exited unexpectedly"

    quarantine_path = workspace / ".plodder" / "sandboxes" / "task-1" / "quarantine" / "quarantine.json"
    assert quarantine_path.is_file()


def test_secret_scoping(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    coordinator = SandboxIsolationCoordinator(workspace)
    scope = TaskSecretScope(
        task_id="task-1",
        environment={"API_KEY": "secret-value"},
        files={"tokens/api.txt": "top-secret"},
    )

    ctx = coordinator.prepare(task_id="task-1", session_id="sess-1", secret_scope=scope)
    env = coordinator.build_worker_environment(ctx, base_environment={"BASE_ENV": "1"})

    assert env["BASE_ENV"] == "1"
    assert env["API_KEY"] == "secret-value"
    assert ctx.secret_scope.redacted_environment()["API_KEY"] == "***"
    assert (ctx.secrets_dir / "tokens" / "api.txt").read_text(encoding="utf-8") == "top-secret"


def test_filesystem_isolation_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    coordinator = SandboxIsolationCoordinator(workspace)

    ctx = coordinator.prepare(task_id="task-1", session_id="sess-1")
    boundary = coordinator.build_filesystem_boundary(ctx)

    assert str(ctx.workspace_dir) in boundary
    assert str(ctx.secrets_dir) in boundary
    assert str(ctx.quarantine_dir) in boundary
