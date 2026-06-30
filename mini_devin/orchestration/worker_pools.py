"""Role-isolated worker pools with per-specialist concurrency limits."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

from .capability_router import SpecialistRole


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, int(default))


def _default_limits() -> dict[SpecialistRole, int]:
    global_default = _int_env("PLODDER_ROLE_POOL_DEFAULT", 2)
    return {
        SpecialistRole.FRONTEND: _int_env("PLODDER_ROLE_POOL_FRONTEND", global_default),
        SpecialistRole.BACKEND: _int_env("PLODDER_ROLE_POOL_BACKEND", global_default),
        SpecialistRole.QA: _int_env("PLODDER_ROLE_POOL_QA", max(1, global_default // 2 or 1)),
        SpecialistRole.DEVOPS: _int_env("PLODDER_ROLE_POOL_DEVOPS", 1),
        SpecialistRole.GENERALIST: _int_env("PLODDER_ROLE_POOL_GENERALIST", global_default),
    }


@dataclass(frozen=True)
class WorkerPoolLease:
    role: SpecialistRole
    unit_id: str
    limit: int
    acquired_at: str
    active_after_acquire: int
    waiting_at_acquire: int

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "unit_id": self.unit_id,
            "limit": self.limit,
            "acquired_at": self.acquired_at,
            "active_after_acquire": self.active_after_acquire,
            "waiting_at_acquire": self.waiting_at_acquire,
        }


class SpecialistWorkerPoolManager:
    """Manages per-role semaphores so specialists run in isolated pools."""

    def __init__(self, limits: dict[SpecialistRole, int] | None = None):
        resolved = dict(_default_limits())
        if limits:
            for role, value in limits.items():
                resolved[role] = max(1, int(value))
        self._limits = resolved
        self._semaphores = {
            role: asyncio.Semaphore(limit) for role, limit in self._limits.items()
        }
        self._active = {role: 0 for role in self._limits}
        self._waiting = {role: 0 for role in self._limits}
        self._active_units = {role: set() for role in self._limits}
        self._log_enabled = (os.environ.get("PLODDER_ROLE_POOL_LOG", "true") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            self._log_interval_seconds = max(
                0.0,
                float((os.environ.get("PLODDER_ROLE_POOL_LOG_INTERVAL_SECONDS") or "15").strip()),
            )
        except ValueError:
            self._log_interval_seconds = 15.0
        self._last_log_monotonic = {role: 0.0 for role in self._limits}
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "SpecialistWorkerPoolManager":
        return cls(limits=None)

    def limit_for(self, role: SpecialistRole) -> int:
        return int(self._limits.get(role, 1))

    @asynccontextmanager
    async def acquire(
        self,
        role: SpecialistRole,
        unit_id: str,
    ) -> AsyncIterator[WorkerPoolLease]:
        if role not in self._semaphores:
            role = SpecialistRole.GENERALIST

        async with self._lock:
            self._waiting[role] += 1

        sem = self._semaphores[role]
        await sem.acquire()

        lease: WorkerPoolLease
        log_line = ""
        async with self._lock:
            self._waiting[role] = max(0, self._waiting[role] - 1)
            self._active[role] += 1
            self._active_units[role].add(unit_id)
            limit = self.limit_for(role)
            lease = WorkerPoolLease(
                role=role,
                unit_id=unit_id,
                limit=limit,
                acquired_at=datetime.now(timezone.utc).isoformat(),
                active_after_acquire=self._active[role],
                waiting_at_acquire=self._waiting[role],
            )
            saturated = self._active[role] >= limit or self._waiting[role] > 0
            if self._log_enabled and saturated:
                now = time.monotonic()
                if (now - self._last_log_monotonic[role]) >= self._log_interval_seconds:
                    self._last_log_monotonic[role] = now
                    log_line = (
                        f"[WorkerPools] saturation role={role.value} "
                        f"active={self._active[role]}/{limit} waiting={self._waiting[role]} unit={unit_id}"
                    )

        if log_line:
            print(log_line)

        try:
            yield lease
        finally:
            async with self._lock:
                self._active[role] = max(0, self._active[role] - 1)
                self._active_units[role].discard(unit_id)
            sem.release()

    def snapshot(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for role in self._limits:
            limit = self._limits[role]
            active = self._active[role]
            waiting = self._waiting[role]
            out[role.value] = {
                "limit": limit,
                "active": active,
                "waiting": waiting,
                "utilization": float(active) / float(limit) if limit > 0 else 0.0,
                "saturated": bool(waiting > 0 or active >= limit),
                "active_units": sorted(self._active_units[role]),
            }
        return out
