"""
E2B cloud sandbox adapter — same surface as :class:`DockerSandbox` for agent integration.

Syncs a bounded subset of ``repo_path`` into ``/workspace`` on sandbox start (one-way snapshot;
host file changes after ``start()`` are not synced to E2B — use Docker for live-mounted workflows).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from .docker_sandbox import SandboxResult, SandboxStatus

_SKIP_PARTS = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "eggs",
    }
)

_MAX_FILES = int(os.environ.get("E2B_SYNC_MAX_FILES", "400"))
_MAX_FILE_BYTES = int(os.environ.get("E2B_SYNC_MAX_FILE_BYTES", str(512 * 1024)))
_BATCH = int(os.environ.get("E2B_SYNC_WRITE_BATCH", "40"))


class E2BSandbox:
    """
    Async E2B sandbox with Docker-compatible methods used by :class:`~mini_devin.orchestrator.agent.Agent`.

    ``is_running()`` is synchronous and reflects successful ``start()`` (agent code is sync here).
    """

    backend = "e2b"

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self._sb = None
        self.status = SandboxStatus.CREATED
        self._running = False
        self.workspace_path = "/workspace"
        self.container_id: str | None = None
        self.config = type("Cfg", (), {"workspace_path": self.workspace_path, "timeout_seconds": 600})()

    def is_running(self) -> bool:
        return self._running

    async def start(self) -> bool:
        if self._running:
            return True
        try:
            from e2b import AsyncSandbox
        except ImportError:
            print("[Sandbox] e2b package not installed; pip install e2b")
            self.status = SandboxStatus.ERROR
            return False

        if not os.environ.get("E2B_API_KEY", "").strip():
            print("[Sandbox] E2B_API_KEY missing")
            self.status = SandboxStatus.ERROR
            return False

        template = os.environ.get("E2B_SANDBOX_TEMPLATE") or None
        timeout_s = int(os.environ.get("E2B_SANDBOX_TIMEOUT", "600"))

        try:
            self._sb = await AsyncSandbox.create(template=template, timeout=timeout_s)
            self.container_id = getattr(self._sb, "sandbox_id", None)
            await self._sb.commands.run("mkdir -p /workspace", cwd="/", timeout=60.0)
            await self._sync_workspace()
            self.status = SandboxStatus.RUNNING
            self._running = True
            return True
        except Exception as e:
            print(f"[Sandbox] E2B start failed: {e}")
            self.status = SandboxStatus.ERROR
            self._sb = None
            self._running = False
            return False

    async def _sync_workspace(self) -> None:
        if not self._sb:
            return
        from e2b.sandbox.filesystem.filesystem import WriteEntry

        root = Path(self.repo_path)
        if not root.is_dir():
            return

        batch: list[WriteEntry] = []
        n = 0
        for path in root.rglob("*"):
            if n >= _MAX_FILES:
                break
            if not path.is_file() or path.is_symlink():
                continue
            try:
                if any(p in _SKIP_PARTS for p in path.parts):
                    continue
                sz = path.stat().st_size
                if sz > _MAX_FILE_BYTES:
                    continue
                rel = path.relative_to(root).as_posix()
                dest = f"{self.workspace_path.rstrip('/')}/{rel}"
                data = path.read_bytes()
            except OSError:
                continue
            batch.append(WriteEntry(path=dest, data=data))
            n += 1
            if len(batch) >= _BATCH:
                await self._sb.files.write_files(batch)
                batch.clear()
        if batch:
            await self._sb.files.write_files(batch)

    async def stop(self) -> bool:
        if not self._sb:
            self._running = False
            self.status = SandboxStatus.STOPPED
            return True
        try:
            await self._sb.kill()
        except Exception as e:
            print(f"[Sandbox] E2B kill: {e}")
        finally:
            self._sb = None
            self._running = False
            self.container_id = None
            self.status = SandboxStatus.STOPPED
        return True

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> SandboxResult:
        if not self._running or not self._sb:
            return SandboxResult(
                stdout="",
                stderr="Sandbox is not running",
                exit_code=-1,
                duration_ms=0,
            )

        wd = working_dir or self.workspace_path
        if wd in (".", ""):
            cwd = self.workspace_path
        elif wd.startswith("/"):
            cwd = wd
        else:
            cwd = f"{self.workspace_path.rstrip('/')}/{wd}".replace("//", "/")

        t0 = datetime.now(timezone.utc)
        to = float(timeout or self.config.timeout_seconds)
        if to <= 0:
            to = 3600.0

        try:
            from e2b.sandbox.commands.command_handle import CommandExitException

            try:
                res = await self._sb.commands.run(cmd=command, cwd=cwd, timeout=to)
            except CommandExitException as e:
                stderr = (e.stderr or "") + (("\n" + e.error) if getattr(e, "error", None) else "")
                dt = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
                return SandboxResult(
                    stdout=e.stdout or "",
                    stderr=stderr.strip(),
                    exit_code=int(e.exit_code),
                    duration_ms=dt,
                    timed_out=False,
                )

            stderr = res.stderr or ""
            if res.error:
                stderr = (stderr + "\n" + res.error).strip() if stderr else str(res.error)
            dt = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
            return SandboxResult(
                stdout=res.stdout or "",
                stderr=stderr,
                exit_code=int(res.exit_code),
                duration_ms=dt,
                timed_out=False,
            )
        except asyncio.TimeoutError:
            dt = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
            return SandboxResult(
                stdout="",
                stderr="Command timed out",
                exit_code=-1,
                duration_ms=dt,
                timed_out=True,
            )
        except Exception as e:
            dt = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
            return SandboxResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_ms=dt,
            )
