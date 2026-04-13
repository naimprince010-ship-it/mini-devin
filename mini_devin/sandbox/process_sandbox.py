"""
Process-based sandbox: run commands via ``subprocess`` (no Docker).

Intended for Railway and other hosts without a Docker daemon. Mirrors the main
ideas of ``SimpleDockerSandbox``: a workspace directory on the host, a random
``SESSION_API_KEY`` injected into the child environment, and a single entry point
to run shell commands.

This is not OS-level isolation; use ``DockerSandbox`` / ``SimpleDockerSandbox``
when you need real containment.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
from typing import Any, Mapping


def _new_session_api_key() -> str:
    return secrets.token_urlsafe(32)


def _resolve_bash() -> str:
    for candidate in ("/bin/bash", "/usr/bin/bash"):
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("bash")
    if found:
        return found
    raise FileNotFoundError(
        "bash is required for ProcessSandbox.exec_bash; install bash or use Docker sandbox."
    )


class ProcessSandbox:
    """Run ``bash -lc`` in a fixed working directory with ``SESSION_API_KEY`` set."""

    WORKSPACE = "/workspace"

    def __init__(
        self,
        project_root: str | None = None,
        *,
        session_api_key: str | None = None,
    ) -> None:
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.session_api_key = session_api_key or _new_session_api_key()
        self._bash = _resolve_bash()
        self._session_env = os.environ.copy()
        self._session_env["SESSION_API_KEY"] = self.session_api_key
        self._session_env.setdefault("SANDBOX_WORKSPACE", self.project_root)

    @property
    def container_id(self) -> str:
        """Opaque id for logging; there is no Docker container."""
        return "process"

    def _cwd_for_exec(self, workdir: str | None) -> str:
        if workdir is None or workdir == self.WORKSPACE:
            return self.project_root
        if workdir.startswith(self.WORKSPACE + "/") or workdir.startswith(self.WORKSPACE + "\\"):
            rel = workdir[len(self.WORKSPACE) :].lstrip("/\\")
            return os.path.abspath(os.path.join(self.project_root, rel))
        return workdir

    def exec_bash(
        self,
        command: str,
        *,
        workdir: str | None = None,
        environment: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> tuple[int, bytes]:
        """
        Run ``command`` through ``bash -lc`` with ``cwd`` under the project root.

        ``environment`` is merged on top of the process environment (including
        ``SESSION_API_KEY``). Extra kwargs are passed to ``subprocess.run`` where
        supported (e.g. ``stdin``). ``text`` defaults to ``False`` so the return
        shape matches ``SimpleDockerSandbox.exec_bash`` (bytes).
        """
        cwd = self._cwd_for_exec(workdir)
        env = self._session_env.copy()
        if environment is not None:
            env.update(dict(environment))

        run_kw: dict[str, Any] = {
            "args": [self._bash, "-lc", command],
            "cwd": cwd,
            "env": env,
            "capture_output": True,
            "timeout": timeout,
        }
        # Allow overrides but keep bytes output by default (docker exec_run style).
        for key in (
            "stdin",
            "stdout",
            "stderr",
            "input",
            "text",
            "encoding",
            "errors",
            "shell",
            "executable",
            "close_fds",
            "preexec_fn",
            "restore_signals",
            "start_new_session",
            "user",
            "group",
            "extra_groups",
            "umask",
            "pipesize",
        ):
            if key in kwargs:
                run_kw[key] = kwargs[key]
        if "text" not in run_kw:
            run_kw["text"] = False

        proc = subprocess.run(**run_kw)
        out_b: bytes
        if run_kw.get("text"):
            so = proc.stdout or ""
            se = proc.stderr or ""
            out_b = (so + se).encode("utf-8", errors="replace")
        else:
            out_b = (proc.stdout or b"") + (proc.stderr or b"")
        return proc.returncode, out_b

    def stop(self, *, remove: bool = True, timeout: int = 10) -> None:
        """No-op for API parity with ``SimpleDockerSandbox``."""

    def __enter__(self) -> ProcessSandbox:
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
