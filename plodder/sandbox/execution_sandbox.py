"""
Docker-backed execution sandbox — run untrusted code, capture streams, drive self-heal loop.

Requires: Docker engine available to the daemon (``docker`` CLI or Docker Desktop).
Optional: ``pip install docker`` for the Python SDK path used here.
"""

from __future__ import annotations

import io
import os
import tarfile
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    container_id: str | None
    command: str


class ExecutionSandbox:
    """
    Minimal wrapper: write files → ``docker run`` → collect logs.

    For production: apply CPU/memory limits, read-only rootfs, user namespaces,
    seccomp/AppArmor profiles, and network isolation per your threat model.
    """

    def __init__(
        self,
        *,
        python_image: str = "python:3.11-alpine",
        node_image: str = "node:20-alpine",
        go_image: str = "golang:1.22-alpine",
        rust_image: str = "rust:alpine",
        alpine_image: str = "alpine:3.19",
        cpp_image: str = "gcc:12-bookworm",
        typescript_image: str = "node:22-alpine",
        default_timeout_sec: int = 60,
    ) -> None:
        self.python_image = python_image
        self.node_image = node_image
        self.go_image = go_image
        self.rust_image = rust_image
        self.alpine_image = alpine_image
        self.cpp_image = cpp_image
        self.typescript_image = typescript_image
        self.default_timeout_sec = default_timeout_sec
        try:
            import docker  # type: ignore

            self._docker = docker.from_env()
        except Exception as e:  # pragma: no cover - environment specific
            raise RuntimeError(
                "Docker SDK unavailable. Install with `pip install docker` and ensure Docker is running."
            ) from e

    @property
    def docker_client(self) -> Any:
        """Docker SDK client (``docker.from_env()``) for image checks and pulls."""
        return self._docker

    def _run_container(
        self,
        *,
        image: str,
        command: list[str],
        files: dict[str, str],
        workdir: str = "/workspace",
        timeout_sec: int | None = None,
        network_mode: str = "none",
    ) -> SandboxResult:
        timeout = timeout_sec if timeout_sec is not None else self.default_timeout_sec
        cid = None
        cmd_str = " ".join(command)

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for rel, content in files.items():
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=rel.replace("\\", "/"))
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        tar_bytes = tar_stream.getvalue()

        container = self._docker.containers.create(
            image,
            command=command,
            working_dir=workdir,
            network_mode=network_mode,
            mem_limit="512m",
            detach=True,
        )
        cid = container.id
        container.put_archive(workdir, tar_bytes)
        container.start()

        timed_out = False
        try:
            result = container.wait(timeout=timeout)
            exit_code = int(result.get("StatusCode", -1))
        except Exception:
            timed_out = True
            exit_code = -1
            try:
                container.kill()
            except Exception:
                pass
        out_b = container.logs(stdout=True, stderr=False)
        err_b = container.logs(stdout=False, stderr=True)
        stdout = out_b.decode("utf-8", errors="replace")
        stderr = err_b.decode("utf-8", errors="replace")
        try:
            container.remove(force=True)
        except Exception:
            pass

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            container_id=cid,
            command=cmd_str,
        )

    def run_python(self, code: str, *, timeout_sec: int | None = None) -> SandboxResult:
        """Execute ``code`` as ``python /workspace/main.py`` inside ``python_image``."""
        name = f"main_{uuid.uuid4().hex[:8]}.py"
        return self._run_container(
            image=self.python_image,
            command=["python", f"/workspace/{name}"],
            files={name: code},
            timeout_sec=timeout_sec,
            network_mode="none",
        )

    def run_node(self, code: str, *, timeout_sec: int | None = None) -> SandboxResult:
        """Execute ``code`` as ``node /workspace/main.js``."""
        name = f"main_{uuid.uuid4().hex[:8]}.js"
        return self._run_container(
            image=self.node_image,
            command=["node", f"/workspace/{name}"],
            files={name: code},
            timeout_sec=timeout_sec,
            network_mode="none",
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
        """
        Upload ``files`` to ``/workspace`` and run ``entry`` using an image inferred from
        the file extension (or ``language`` hint: ``python``, ``javascript``, ``typescript``, …).

        When ``language_key`` is set (e.g. RAG doc slug: ``rust``, ``python``), the Docker image
        is chosen from ``plodder.sandbox.container_manager``; missing local images yield stderr
        notes (``docker pull …``) and optionally a generic ``buildpack-deps`` fallback.
        """
        from plodder.sandbox.container_manager import plan_container_run

        auto_pull = os.environ.get("PLODDER_DOCKER_AUTO_PULL", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
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
            docker_client=self._docker,
            prefer_generic_if_image_missing=True,
            auto_pull_missing=auto_pull,
        )
        result = self._run_container(
            image=planned.image,
            command=planned.argv,
            files=files,
            timeout_sec=timeout_sec,
            network_mode="bridge" if network else "none",
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
        """
        Run an arbitrary argv (e.g. ``["sh","-c","npm install && npm test"]``) in a base image
        chosen from ``language_hint`` (``javascript`` → Node, ``python`` → CPython, else Alpine).
        """
        from plodder.sandbox.toolchain_detect import image_for_shell_language

        image = image_for_shell_language(
            language_hint,
            python_image=self.python_image,
            node_image=self.node_image,
            alpine_image=self.alpine_image,
        )
        return self._run_container(
            image=image,
            command=list(argv),
            files=files,
            timeout_sec=timeout_sec,
            network_mode="bridge" if network else "none",
        )

    def run_command(
        self,
        image: str,
        argv: list[str],
        *,
        files: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        network: bool = False,
    ) -> SandboxResult:
        """Generic: arbitrary image + argv, optional file bundle under ``/workspace``."""
        return self._run_container(
            image=image,
            command=argv,
            files=files or {},
            timeout_sec=timeout_sec,
            network_mode="bridge" if network else "none",
        )


def format_stacktrace_for_llm(result: SandboxResult) -> str:
    """Normalize stderr/stdout for the self-healing planner (no mutation)."""
    parts = [f"exit={result.exit_code}", f"timeout={result.timed_out}", f"cmd={result.command}"]
    blob = (result.stderr or "") + ("\n" if result.stderr and result.stdout else "") + (result.stdout or "")
    parts.append("--- output ---\n" + blob.strip())
    return "\n".join(parts)
