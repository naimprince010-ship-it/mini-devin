"""
Plodder ``UnifiedSessionDriver`` sandbox backend using :class:`ProcessSandbox` (host bash, no Docker).

Implements the same operations as ``plodder.sandbox.ExecutionSandbox`` that the task
executor needs: ``run_detected`` and ``run_shell_in_workspace``, returning ``SandboxResult``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from plodder.sandbox.container_manager import plan_container_run

_logger = logging.getLogger(__name__)
from plodder.sandbox.execution_sandbox import SandboxResult
from plodder.sandbox.toolchain_detect import image_for_shell_language, resolve_sql_url_from_env

from mini_devin.sandbox.process_sandbox import ProcessSandbox


def _posix_single_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _argv_to_bash_line(argv: list[str]) -> str:
    return " ".join(_posix_single_quote(a) for a in argv)


class ProcessExecutionSandbox:
    """
    Host-side execution for ``sandbox_run`` / ``sandbox_shell`` (Railway, no Docker).

    Exposes the same image-related attributes as ``ExecutionSandbox`` so
    ``plan_container_run`` and ``container_verify`` can reuse defaults.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        python_image: str = "python:3.11-alpine",
        node_image: str = "node:20-alpine",
        go_image: str = "golang:1.22-alpine",
        rust_image: str = "rust:alpine",
        alpine_image: str = "alpine:3.19",
        cpp_image: str = "gcc:12-bookworm",
        typescript_image: str = "node:22-alpine",
        java_image: str = "eclipse-temurin:21-jdk-alpine",
        php_image: str = "php:8.3-cli-alpine",
        dotnet_image: str = "mcr.microsoft.com/dotnet/sdk:8.0-alpine",
        maven_image: str = "maven:3.9.9-eclipse-temurin-21-alpine",
        gradle_image: str = "gradle:8.10.2-jdk21-alpine",
        composer_image: str = "composer:2",
        postgres_client_image: str = "postgres:16-alpine",
        default_timeout_sec: int = 60,
    ) -> None:
        self._root = Path(workspace_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._ps = ProcessSandbox(str(self._root))
        self.python_image = python_image
        self.node_image = node_image
        self.go_image = go_image
        self.rust_image = rust_image
        self.alpine_image = alpine_image
        self.cpp_image = cpp_image
        self.typescript_image = typescript_image
        self.java_image = java_image
        self.php_image = php_image
        self.dotnet_image = dotnet_image
        self.maven_image = maven_image
        self.gradle_image = gradle_image
        self.composer_image = composer_image
        self.postgres_client_image = postgres_client_image
        self.default_timeout_sec = default_timeout_sec

    def exec_shell(
        self,
        command: str,
        *,
        workdir: str | None = None,
        timeout_sec: int | None = None,
    ) -> SandboxResult:
        """
        Run an arbitrary shell command in the workspace via host bash (Railway / no Docker).

        ``workdir`` is passed to :class:`ProcessSandbox` (``None`` or ``\".\"`` → workspace root).
        """
        t = timeout_sec if timeout_sec is not None else self.default_timeout_sec
        t = max(1, min(300, int(t)))
        code, raw = self._ps.exec_bash(command, workdir=workdir, timeout=float(t))
        stdout = raw.decode("utf-8", errors="replace")
        return SandboxResult(
            stdout=stdout,
            stderr="",
            exit_code=code,
            timed_out=False,
            container_id=self._ps.container_id,
            command=command[:4000],
        )

    @property
    def docker_client(self) -> None:
        """No Docker daemon; ``container_verify`` skips image presence checks."""
        return None

    def _materialize_files(self, files: dict[str, str]) -> None:
        root = self._root
        for rel, content in files.items():
            reln = rel.replace("\\", "/").lstrip("/")
            path = (root / reln).resolve()
            path.relative_to(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")

    def _host_prefix(self) -> str:
        return str(self._root.resolve())

    def _map_arg_paths(self, arg: str) -> str:
        host = self._host_prefix()
        if arg.startswith("/workspace/"):
            return host + arg[len("/workspace") :]
        if arg == "/workspace":
            return host
        if "/workspace/" in arg or arg.endswith("/workspace"):
            return arg.replace("/workspace/", host + "/").replace("/workspace", host)
        return arg

    def _run_mapped_argv(
        self,
        argv: list[str],
        container_env: tuple[tuple[str, str], ...],
        *,
        timeout_sec: int | None,
        network: bool,
        command_label: str,
    ) -> SandboxResult:
        mapped = [self._map_arg_paths(a) for a in argv]
        exports: list[str] = []
        for k, v in container_env:
            exports.append(f"export {k}={_posix_single_quote(v)}")
        export_prefix = (" ".join(exports) + " && ") if exports else ""
        inner = _argv_to_bash_line(mapped)
        cmd = export_prefix + inner if export_prefix else inner
        t = timeout_sec if timeout_sec is not None else self.default_timeout_sec
        _logger.info(
            "[ProcessExecutionSandbox] bash cwd=%s timeout=%s cmd_preview=%s",
            self._root,
            t,
            cmd[:800] + ("…" if len(cmd) > 800 else ""),
        )
        code, raw = self._ps.exec_bash(cmd, timeout=float(t))
        stdout = raw.decode("utf-8", errors="replace")
        stderr_parts: list[str] = []
        if network:
            stderr_parts.append(
                "(process sandbox) network=true ignored — commands run on the host network."
            )
        stderr = "\n".join(stderr_parts) if stderr_parts else ""
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=code,
            timed_out=False,
            container_id=self._ps.container_id,
            command=command_label[:4000],
        )

    def run_detected(
        self,
        files: dict[str, str],
        *,
        entry: str,
        language: str | None = None,
        language_key: str | None = None,
        timeout_sec: int | None = None,
        network: bool = False,
    ) -> SandboxResult:
        self._materialize_files(files)
        planned = plan_container_run(
            entry=entry,
            language_hint=language,
            language_key=language_key,
            python_image=self.python_image,
            node_image=self.node_image,
            go_image=self.go_image,
            rust_image=self.rust_image,
            alpine_image=self.alpine_image,
            cpp_image=self.cpp_image,
            typescript_image=self.typescript_image,
            java_image=self.java_image,
            php_image=self.php_image,
            dotnet_image=self.dotnet_image,
            maven_image=self.maven_image,
            gradle_image=self.gradle_image,
            composer_image=self.composer_image,
            postgres_client_image=self.postgres_client_image,
            sql_url=resolve_sql_url_from_env(),
            docker_client=None,
            prefer_generic_if_image_missing=False,
            auto_pull_missing=False,
            workspace_files=files,
        )
        cmd_label = " ".join(planned.argv)
        result = self._run_mapped_argv(
            list(planned.argv),
            planned.container_env,
            timeout_sec=timeout_sec,
            network=network,
            command_label=cmd_label,
        )
        if planned.notes:
            prefix = "\n".join(planned.notes) + "\n"
            result = SandboxResult(
                stdout=result.stdout,
                stderr=prefix + result.stderr,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                container_id=result.container_id,
                command=result.command,
            )
        return result

    def run_shell_in_workspace(
        self,
        files: dict[str, str],
        argv: list[str],
        *,
        language_hint: str | None = None,
        timeout_sec: int | None = None,
        network: bool = False,
    ) -> SandboxResult:
        self._materialize_files(files)
        image = image_for_shell_language(
            language_hint,
            python_image=self.python_image,
            node_image=self.node_image,
            alpine_image=self.alpine_image,
            java_image=self.java_image,
            php_image=self.php_image,
            dotnet_image=self.dotnet_image,
            postgres_client_image=self.postgres_client_image,
        )
        cmd_label = f"(image={image}) " + " ".join(argv)
        return self._run_mapped_argv(
            list(argv),
            (),
            timeout_sec=timeout_sec,
            network=network,
            command_label=cmd_label,
        )


def use_host_process_terminal_for_tooling() -> bool:
    """
    True when terminal commands should use :meth:`ProcessExecutionSandbox.exec_shell`
    (host bash) instead of Docker.

    Enabled when ``RAILWAY_ENVIRONMENT`` is set, or ``USE_PROCESS_EXECUTION_SANDBOX=1``
    (``true``/``yes``/``on``). Disabled on Windows or when explicitly turned ``off``.
    """
    if os.name == "nt":
        return False
    v = (os.environ.get("USE_PROCESS_EXECUTION_SANDBOX") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return bool((os.environ.get("RAILWAY_ENVIRONMENT") or "").strip())
