"""
Process-based sandbox: run commands via ``subprocess`` (no Docker).

Stateful **cwd** (and light **export** persistence) follow an OpenHands-style
pattern: each ``bash -lc`` run wraps the user command so ``pwd`` is written to
``.mini_devin/_shell/cwd.txt`` under the workspace (shared with Docker bind mount).

This is not OS-level isolation; use ``DockerSandbox`` / ``SimpleDockerSandbox``
when you need real containment.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from mini_devin.sandbox.runtime_protocol import ExecutionResult
from mini_devin.sandbox.stateful_exec import (
    build_stateful_bash_script,
    maybe_append_exports,
    noninteractive_export_block,
    posix_paths_for_process_sandbox,
)


def _new_session_api_key() -> str:
    return secrets.token_urlsafe(32)


def _resolve_bash() -> str:
    for candidate in ("/bin/bash", "/usr/bin/bash"):
        if os.path.isfile(candidate):
            return candidate
    if os.name == "nt":
        for base in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
        ):
            if not base:
                continue
            git_bash = os.path.join(base, "Git", "bin", "bash.exe")
            if os.path.isfile(git_bash):
                return git_bash
    found = shutil.which("bash")
    if found:
        return found
    raise FileNotFoundError(
        "bash is required for ProcessSandbox.exec_bash; install bash or use Docker sandbox."
    )


def _assume_yes_pipe(user_command: str) -> str:
    """
    When ``MINIDEVIN_ASSUME_YES=1``, prefix ``yes`` for common apt/dpkg prompts.

    Narrow trigger to avoid piping unrelated commands that mention ``read``.
    """
    v = (os.environ.get("MINIDEVIN_ASSUME_YES") or "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return user_command
    stripped = user_command.strip()
    if "apt-get" in stripped or stripped.startswith("apt ") or "dpkg " in stripped:
        return f"yes '' | {user_command}"
    return user_command


class ProcessSandbox:
    """Run ``bash -lc`` in a workspace with optional **stateful cwd** (default: on)."""

    WORKSPACE = "/workspace"

    def __init__(
        self,
        project_root: str | None = None,
        *,
        session_api_key: str | None = None,
        stateful_shell: bool = True,
    ) -> None:
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.session_api_key = session_api_key or _new_session_api_key()
        self._bash = _resolve_bash()
        self._session_env = os.environ.copy()
        self._session_env["SESSION_API_KEY"] = self.session_api_key
        self._session_env.setdefault("SANDBOX_WORKSPACE", self.project_root)
        self._stateful_shell = bool(stateful_shell) and os.name != "nt"

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
        return workdir or self.project_root

    def exec_bash(
        self,
        command: str,
        *,
        workdir: str | None = None,
        environment: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> tuple[int, bytes] | tuple[int, tuple[bytes, bytes]] | ExecutionResult:
        """
        Run ``command`` through ``bash -lc``.

        When ``stateful_shell`` is enabled (default on POSIX), **cd** in ``command``
        persists for the next invocation via ``.mini_devin/_shell/cwd.txt``.

        On timeout, returns :class:`ExecutionResult` with ``timed_out=True`` and
        partial streams; the process group is killed on POSIX.
        """
        cwd = self._cwd_for_exec(workdir)
        env = self._session_env.copy()
        if environment is not None:
            env.update(dict(environment))

        user = _assume_yes_pipe(command)
        maybe_append_exports(self.project_root, user)

        if self._stateful_shell:
            ws, cf, ef = posix_paths_for_process_sandbox(self.project_root)
            if workdir is not None and str(workdir).strip() not in ("", ".", self.WORKSPACE):
                try:
                    Path(cf).write_text(self._cwd_for_exec(workdir), encoding="utf-8")
                except OSError:
                    pass
            body = build_stateful_bash_script(
                noninteractive_export_block() + user,
                workspace_posix=ws,
                cwd_file_posix=cf,
                env_file_posix=ef,
                clamp_under_workspace=True,
            )
        else:
            body = noninteractive_export_block() + user

        run_kw: dict[str, Any] = {
            "args": [self._bash, "-lc", body],
            "cwd": cwd,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "timeout": timeout,
        }
        if os.name != "nt":
            run_kw["start_new_session"] = True

        for key in (
            "stdin",
            "input",
            "text",
            "encoding",
            "errors",
            "shell",
            "executable",
            "close_fds",
            "preexec_fn",
            "restore_signals",
            "user",
            "group",
            "extra_groups",
            "umask",
            "pipesize",
        ):
            if key in kwargs:
                run_kw[key] = kwargs[key]

        try:
            proc = subprocess.run(**run_kw)
        except subprocess.TimeoutExpired as exc:
            out_b = exc.stdout or b"" if isinstance(exc.stdout, bytes) else b""
            err_b = exc.stderr or b"" if isinstance(exc.stderr, bytes) else b""
            if not isinstance(out_b, bytes):
                out_b = b""
            if not isinstance(err_b, bytes):
                err_b = b""
            err_b = err_b + b"\n[process sandbox] Command timed out; process group terminated."
            return ExecutionResult(
                exit_code=-1,
                stdout=out_b,
                stderr=err_b,
                timed_out=True,
            )

        out_b = proc.stdout or b""
        err_b = proc.stderr or b""
        return (proc.returncode or 0, (out_b, err_b))

    def stop(self, *, remove: bool = True, timeout: int = 10) -> None:
        """No-op for API parity with ``SimpleDockerSandbox``."""

    def __enter__(self) -> ProcessSandbox:
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
