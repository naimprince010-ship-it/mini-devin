"""Local sandbox lifecycle scaffolding for worker hardening.

This module stays additive and file-backed. It does not launch containers yet; it prepares
per-task sandbox directories, writes audit logs, tracks lifecycle transitions, scopes secrets,
and provides recovery/quarantine hooks for future runtime backends.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .checkpoint_store import JsonlCheckpointStore
from mini_devin.contracts.protocols import DurableCheckpoint
from .observability import TimelineRecord, emit_worker_metric, record_timeline_event


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def sandbox_hardening_enabled() -> bool:
    return _flag_enabled("PLODDER_WORKER_SANDBOX_HARDENING")


def sandbox_audit_enabled() -> bool:
    return _flag_enabled("PLODDER_SANDBOX_AUDIT", True)


def sandbox_quarantine_enabled() -> bool:
    return _flag_enabled("PLODDER_SANDBOX_QUARANTINE", True)


def sandbox_recovery_enabled() -> bool:
    return _flag_enabled("PLODDER_WORKER_CRASH_RECOVERY", True)


def filesystem_isolation_enabled() -> bool:
    return _flag_enabled("PLODDER_FS_ISOLATION", True)


def secret_scoping_enabled() -> bool:
    return _flag_enabled("PLODDER_TASK_SECRET_SCOPING", True)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SandboxPhase(str, Enum):
    PREPARED = "prepared"
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    CRASHED = "crashed"
    RECOVERING = "recovering"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class SandboxResourceLimits:
    cpu_cores: float = 1.0
    memory_mb: int = 512
    timeout_seconds: int = 300
    pids_limit: int = 64

    def validate(self) -> None:
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be greater than 0")
        if self.memory_mb <= 0:
            raise ValueError("memory_mb must be greater than 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if self.pids_limit <= 0:
            raise ValueError("pids_limit must be greater than 0")

    def clamp_timeout(self, requested_timeout: int | None = None) -> int:
        requested = self.timeout_seconds if requested_timeout is None else int(requested_timeout)
        return max(1, min(int(self.timeout_seconds), requested))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "timeout_seconds": self.timeout_seconds,
            "pids_limit": self.pids_limit,
        }


@dataclass(frozen=True, slots=True)
class TaskSecretScope:
    task_id: str
    environment: Mapping[str, str] = field(default_factory=dict)
    files: Mapping[str, str] = field(default_factory=dict)

    def materialize_environment(self, base_environment: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(base_environment or {})
        if secret_scoping_enabled():
            env.update({str(key): str(value) for key, value in self.environment.items()})
        return env

    def redacted_environment(self) -> dict[str, str]:
        return {str(key): "***" for key in self.environment}

    def materialize_files(self, root: Path) -> list[Path]:
        written: list[Path] = []
        if not secret_scoping_enabled():
            return written
        for rel_path, content in self.files.items():
            target = (root / rel_path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
            written.append(target)
        return written

    def redacted_files(self) -> dict[str, str]:
        return {str(path): "***" for path in self.files}


@dataclass(frozen=True, slots=True)
class SandboxAuditEntry:
    event_type: str
    task_id: str
    session_id: str
    phase: SandboxPhase
    ts: datetime = field(default_factory=_utcnow)
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ts": self.ts.isoformat(),
            "event_type": self.event_type,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "phase": self.phase.value,
            "details": dict(self.details),
        }
        if self.reason is not None:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True, slots=True)
class TaskSandboxContext:
    task_id: str
    session_id: str
    workspace_root: Path
    sandbox_root: Path
    workspace_dir: Path
    secrets_dir: Path
    quarantine_dir: Path
    audit_log_path: Path
    checkpoint_id: str
    phase: SandboxPhase = SandboxPhase.PREPARED
    limits: SandboxResourceLimits = field(default_factory=SandboxResourceLimits)
    secret_scope: TaskSecretScope = field(default_factory=lambda: TaskSecretScope(task_id=""))
    crash_count: int = 0
    last_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "sandbox_root": str(self.sandbox_root),
            "workspace_dir": str(self.workspace_dir),
            "secrets_dir": str(self.secrets_dir),
            "quarantine_dir": str(self.quarantine_dir),
            "audit_log_path": str(self.audit_log_path),
            "checkpoint_id": self.checkpoint_id,
            "phase": self.phase.value,
            "limits": self.limits.to_dict(),
            "secret_scope": {
                "task_id": self.secret_scope.task_id,
                "environment": self.secret_scope.redacted_environment(),
                "files": self.secret_scope.redacted_files(),
            },
            "crash_count": self.crash_count,
            "last_reason": self.last_reason,
        }


class SandboxIsolationCoordinator:
    """File-backed sandbox lifecycle manager.

    The coordinator prepares a per-task filesystem boundary now and exposes lifecycle hooks
    for later container or VM execution backends.
    """

    def __init__(self, workspace: str | Path, *, checkpoint_store: JsonlCheckpointStore | None = None) -> None:
        self.workspace_root = Path(workspace).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._plodder_root = self.workspace_root / ".plodder"
        self._plodder_root.mkdir(parents=True, exist_ok=True)
        self._sandbox_root = self._plodder_root / "sandboxes"
        self._sandbox_root.mkdir(parents=True, exist_ok=True)
        self._checkpoint_store = checkpoint_store or JsonlCheckpointStore(self.workspace_root)

    def _task_root(self, task_id: str) -> Path:
        root = self._sandbox_root / task_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _audit_path(self, task_id: str) -> Path:
        return self._task_root(task_id) / "audit.jsonl"

    def _write_audit(self, entry: SandboxAuditEntry) -> None:
        if not sandbox_audit_enabled():
            return
        path = self._audit_path(entry.task_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), default=str, ensure_ascii=False) + "\n")

    def _save_checkpoint(self, context: TaskSandboxContext) -> None:
        self._checkpoint_store.save(
            DurableCheckpoint(
                checkpoint_id=context.checkpoint_id,
                scope_id=context.task_id,
                state={
                    "task_id": context.task_id,
                    "session_id": context.session_id,
                    "phase": context.phase.value,
                    "sandbox_root": str(context.sandbox_root),
                    "workspace_dir": str(context.workspace_dir),
                    "crash_count": context.crash_count,
                    "last_reason": context.last_reason,
                },
                metadata={"limits": context.limits.to_dict()},
            )
        )

    def prepare(
        self,
        *,
        task_id: str,
        session_id: str,
        limits: SandboxResourceLimits | None = None,
        secret_scope: TaskSecretScope | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskSandboxContext:
        limits = limits or SandboxResourceLimits()
        limits.validate()
        task_root = self._task_root(task_id)
        workspace_dir = task_root / "workspace"
        secrets_dir = task_root / "secrets"
        quarantine_dir = task_root / "quarantine"
        for directory in (workspace_dir, secrets_dir, quarantine_dir):
            directory.mkdir(parents=True, exist_ok=True)

        scope = secret_scope or TaskSecretScope(task_id=task_id)
        scope.materialize_files(secrets_dir)

        context = TaskSandboxContext(
            task_id=task_id,
            session_id=session_id,
            workspace_root=self.workspace_root,
            sandbox_root=task_root,
            workspace_dir=workspace_dir,
            secrets_dir=secrets_dir,
            quarantine_dir=quarantine_dir,
            audit_log_path=self._audit_path(task_id),
            checkpoint_id=f"sandbox:{session_id}:{task_id}",
            phase=SandboxPhase.PREPARED,
            limits=limits,
            secret_scope=scope,
        )
        self._write_audit(
            SandboxAuditEntry(
                event_type="sandbox.prepared",
                task_id=task_id,
                session_id=session_id,
                phase=SandboxPhase.PREPARED,
                details={"metadata": dict(metadata or {}), "limits": limits.to_dict()},
            )
        )
        record_timeline_event(
            self.workspace_root,
            TimelineRecord(
                event_type="sandbox.prepared",
                source="sandbox",
                session_id=session_id,
                task_id=task_id,
                unit_id=task_id,
                status=SandboxPhase.PREPARED.value,
                payload={"metadata": dict(metadata or {}), "limits": limits.to_dict()},
            ),
        )
        emit_worker_metric(
            self.workspace_root,
            "worker.sandbox.prepared",
            1.0,
            labels={"task_id": task_id},
            correlation_id=None,
            trace_id=None,
            span_id=None,
        )
        self._save_checkpoint(context)
        return context

    def activate(self, context: TaskSandboxContext, *, metadata: Mapping[str, Any] | None = None) -> TaskSandboxContext:
        active = replace(context, phase=SandboxPhase.ACTIVE)
        self._write_audit(
            SandboxAuditEntry(
                event_type="sandbox.activated",
                task_id=active.task_id,
                session_id=active.session_id,
                phase=SandboxPhase.ACTIVE,
                details=dict(metadata or {}),
            )
        )
        record_timeline_event(
            self.workspace_root,
            TimelineRecord(
                event_type="sandbox.activated",
                source="sandbox",
                session_id=active.session_id,
                task_id=active.task_id,
                unit_id=active.task_id,
                status=SandboxPhase.ACTIVE.value,
                payload=dict(metadata or {}),
            ),
        )
        emit_worker_metric(self.workspace_root, "worker.sandbox.active", 1.0, labels={"task_id": active.task_id})
        self._save_checkpoint(active)
        return active

    def quarantine(
        self,
        context: TaskSandboxContext,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskSandboxContext:
        quarantined = replace(context, phase=SandboxPhase.QUARANTINED, last_reason=reason)
        payload = {"reason": reason, "metadata": dict(metadata or {})}
        (context.quarantine_dir / "quarantine.json").write_text(
            json.dumps(payload, default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_audit(
            SandboxAuditEntry(
                event_type="sandbox.quarantined",
                task_id=quarantined.task_id,
                session_id=quarantined.session_id,
                phase=SandboxPhase.QUARANTINED,
                reason=reason,
                details=dict(metadata or {}),
            )
        )
        record_timeline_event(
            self.workspace_root,
            TimelineRecord(
                event_type="sandbox.quarantined",
                source="sandbox",
                session_id=quarantined.session_id,
                task_id=quarantined.task_id,
                unit_id=quarantined.task_id,
                status=SandboxPhase.QUARANTINED.value,
                payload=payload,
            ),
        )
        emit_worker_metric(
            self.workspace_root,
            "worker.sandbox.quarantined",
            1.0,
            labels={"task_id": quarantined.task_id, "reason": reason},
        )
        self._save_checkpoint(quarantined)
        return quarantined

    def record_crash(
        self,
        context: TaskSandboxContext,
        *,
        reason: str,
        exit_code: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskSandboxContext:
        crashed = replace(context, phase=SandboxPhase.CRASHED, crash_count=context.crash_count + 1, last_reason=reason)
        self._write_audit(
            SandboxAuditEntry(
                event_type="sandbox.crashed",
                task_id=crashed.task_id,
                session_id=crashed.session_id,
                phase=SandboxPhase.CRASHED,
                reason=reason,
                details={"exit_code": exit_code, **dict(metadata or {})},
            )
        )
        record_timeline_event(
            self.workspace_root,
            TimelineRecord(
                event_type="sandbox.crashed",
                source="sandbox",
                session_id=crashed.session_id,
                task_id=crashed.task_id,
                unit_id=crashed.task_id,
                status=SandboxPhase.CRASHED.value,
                payload={"reason": reason, "exit_code": exit_code, **dict(metadata or {})},
            ),
        )
        emit_worker_metric(self.workspace_root, "worker.sandbox.crashed", 1.0, labels={"task_id": crashed.task_id})
        self._save_checkpoint(crashed)
        if sandbox_quarantine_enabled():
            return self.quarantine(crashed, reason=reason, metadata=metadata)
        return crashed

    def recover(self, context: TaskSandboxContext, *, metadata: Mapping[str, Any] | None = None) -> TaskSandboxContext:
        if not sandbox_recovery_enabled():
            return context
        recovering = replace(context, phase=SandboxPhase.RECOVERING)
        self._write_audit(
            SandboxAuditEntry(
                event_type="sandbox.recovering",
                task_id=recovering.task_id,
                session_id=recovering.session_id,
                phase=SandboxPhase.RECOVERING,
                details=dict(metadata or {}),
            )
        )
        record_timeline_event(
            self.workspace_root,
            TimelineRecord(
                event_type="sandbox.recovering",
                source="sandbox",
                session_id=recovering.session_id,
                task_id=recovering.task_id,
                unit_id=recovering.task_id,
                status=SandboxPhase.RECOVERING.value,
                payload=dict(metadata or {}),
            ),
        )
        restored = replace(recovering, phase=SandboxPhase.ACTIVE)
        self._write_audit(
            SandboxAuditEntry(
                event_type="sandbox.recovered",
                task_id=restored.task_id,
                session_id=restored.session_id,
                phase=SandboxPhase.ACTIVE,
                details=dict(metadata or {}),
            )
        )
        record_timeline_event(
            self.workspace_root,
            TimelineRecord(
                event_type="sandbox.recovered",
                source="sandbox",
                session_id=restored.session_id,
                task_id=restored.task_id,
                unit_id=restored.task_id,
                status=SandboxPhase.ACTIVE.value,
                payload=dict(metadata or {}),
            ),
        )
        emit_worker_metric(self.workspace_root, "worker.sandbox.recovered", 1.0, labels={"task_id": restored.task_id})
        self._save_checkpoint(restored)
        return restored

    def release(self, context: TaskSandboxContext, *, metadata: Mapping[str, Any] | None = None) -> TaskSandboxContext:
        released = replace(context, phase=SandboxPhase.RELEASED)
        self._write_audit(
            SandboxAuditEntry(
                event_type="sandbox.released",
                task_id=released.task_id,
                session_id=released.session_id,
                phase=SandboxPhase.RELEASED,
                details=dict(metadata or {}),
            )
        )
        record_timeline_event(
            self.workspace_root,
            TimelineRecord(
                event_type="sandbox.released",
                source="sandbox",
                session_id=released.session_id,
                task_id=released.task_id,
                unit_id=released.task_id,
                status=SandboxPhase.RELEASED.value,
                payload=dict(metadata or {}),
            ),
        )
        emit_worker_metric(self.workspace_root, "worker.sandbox.released", 1.0, labels={"task_id": released.task_id})
        self._save_checkpoint(released)
        return released

    def build_worker_environment(
        self,
        context: TaskSandboxContext,
        *,
        base_environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        env = dict(base_environment or {})
        env["PLODDER_SANDBOX_TASK_ID"] = context.task_id
        env["PLODDER_SANDBOX_SESSION_ID"] = context.session_id
        env["PLODDER_SANDBOX_ROOT"] = str(context.sandbox_root)
        env["PLODDER_SANDBOX_WORKSPACE"] = str(context.workspace_dir)
        env["PLODDER_SANDBOX_PHASE"] = context.phase.value
        env["PLODDER_SANDBOX_CPU_LIMIT"] = str(context.limits.cpu_cores)
        env["PLODDER_SANDBOX_MEMORY_MB"] = str(context.limits.memory_mb)
        env["PLODDER_SANDBOX_TIMEOUT_SECONDS"] = str(context.limits.timeout_seconds)
        env["PLODDER_SANDBOX_PIDS_LIMIT"] = str(context.limits.pids_limit)
        env.update(context.secret_scope.materialize_environment())
        return env

    def build_filesystem_boundary(self, context: TaskSandboxContext) -> tuple[str, ...]:
        if not filesystem_isolation_enabled():
            return (str(context.workspace_root),)
        return (
            str(context.workspace_dir),
            str(context.secrets_dir),
            str(context.quarantine_dir),
        )

    def recover_quarantined_context(self, context: TaskSandboxContext) -> TaskSandboxContext:
        if context.phase not in (SandboxPhase.QUARANTINED, SandboxPhase.CRASHED):
            return context
        return self.recover(context, metadata={"recovered_from": context.phase.value})

    def load_context(self, task_id: str) -> TaskSandboxContext | None:
        checkpoint_id = f"sandbox:*:{task_id}"
        for checkpoint in reversed(self._checkpoint_store.list(task_id)):
            if not checkpoint.checkpoint_id.startswith("sandbox:"):
                continue
            state = dict(checkpoint.state or {})
            phase_raw = str(state.get("phase") or "prepared")
            try:
                phase = SandboxPhase(phase_raw)
            except ValueError:
                phase = SandboxPhase.PREPARED
            root = self._task_root(task_id)
            return TaskSandboxContext(
                task_id=task_id,
                session_id=str(state.get("session_id") or ""),
                workspace_root=self.workspace_root,
                sandbox_root=root,
                workspace_dir=root / "workspace",
                secrets_dir=root / "secrets",
                quarantine_dir=root / "quarantine",
                audit_log_path=self._audit_path(task_id),
                checkpoint_id=str(checkpoint.checkpoint_id),
                phase=phase,
                limits=SandboxResourceLimits(**dict(checkpoint.metadata.get("limits") or {})),
                secret_scope=TaskSecretScope(task_id=task_id),
                crash_count=int(state.get("crash_count") or 0),
                last_reason=state.get("last_reason"),
            )
        return None


def build_default_sandbox_limits(*, cpu_cores: float = 1.0, memory_mb: int = 512, timeout_seconds: int = 300) -> SandboxResourceLimits:
    limits = SandboxResourceLimits(cpu_cores=cpu_cores, memory_mb=memory_mb, timeout_seconds=timeout_seconds)
    limits.validate()
    return limits
