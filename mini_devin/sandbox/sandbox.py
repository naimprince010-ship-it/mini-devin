"""
Lightweight Docker sandbox (OpenHands DockerSandboxService–style sketch).

- Starts a long-running container with the host project bound to ``/workspace``.
- Exposes a random ``SESSION_API_KEY`` in the container environment (for parity with
  session-auth patterns; wire your own in-container service if needed).
- Runs shell commands via the Docker SDK ``Container.exec_run`` API.
- Ensures the run image exists: like OpenHands ``pull_if_missing``, missing images are
  **pulled** from a registry, or **built** from ``Dockerfile.sandbox`` for unqualified
  ``plodder-sandbox`` tags.

For full Plodder integration (limits, seccomp, CLI-based flow), use ``docker_sandbox.DockerSandbox``.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docker import DockerClient
    from docker.models.containers import Container

_logger = logging.getLogger(__name__)

# Unqualified plodder-sandbox images are built from Dockerfile.sandbox in this repo.
_PLODDER_IMAGE = re.compile(r"^plodder-sandbox(?::[\w][\w._-]*)?$")


def _new_session_api_key() -> str:
    return secrets.token_urlsafe(32)


def _strip_digest(ref: str) -> str:
    return ref.split("@", 1)[0]


def _is_repo_plodder_sandbox(image: str) -> bool:
    ref = _strip_digest(image)
    if "/" in ref:
        return False
    return bool(_PLODDER_IMAGE.fullmatch(ref))


def _find_dockerfile_sandbox_context() -> Path | None:
    # Prefer cwd when running from a checkout (works if package is installed elsewhere).
    cwd = Path.cwd().resolve()
    if (cwd / "Dockerfile.sandbox").is_file():
        return cwd

    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        dockerfile = candidate / "Dockerfile.sandbox"
        if dockerfile.is_file():
            return candidate
    return None


def _drain_build_logs(build_logs: Any) -> None:
    """Consume the build log iterator (docker-py ``images.build`` second return value)."""
    for _ in build_logs:
        pass


def ensure_sandbox_image(client: DockerClient, image: str) -> None:
    """
    If ``image`` is not present locally, pull it (registry) or build Plodder sandbox
    from ``Dockerfile.sandbox`` (OpenHands-style ``images.get`` / pull / build).
    """
    import docker.errors

    try:
        client.images.get(image)
        return
    except docker.errors.ImageNotFound:
        pass

    if _is_repo_plodder_sandbox(image):
        ctx = _find_dockerfile_sandbox_context()
        if ctx is None:
            raise FileNotFoundError(
                "Dockerfile.sandbox not found while resolving plodder-sandbox image; "
                "clone the mini-devin repo or pass a different ``image=``."
            )
        _logger.info(
            "SimpleDockerSandbox: image %r missing locally; building from %s",
            image,
            ctx / "Dockerfile.sandbox",
        )
        _, build_logs = client.images.build(
            path=str(ctx),
            dockerfile="Dockerfile.sandbox",
            tag=image,
            rm=True,
            forcerm=True,
            decode=True,
        )
        _drain_build_logs(build_logs)
        return

    _logger.info("SimpleDockerSandbox: image %r missing locally; pulling", image)
    client.images.pull(image)


class SimpleDockerSandbox:
    """
    Minimal sandbox: one container, ``/workspace`` mount, ``exec_run`` for bash.

    Default image is ``plodder-sandbox:latest``. If it (or another unqualified
    ``plodder-sandbox:*`` tag) is missing, it is built from ``Dockerfile.sandbox``.
    Other images are pulled when missing (OpenHands-style).
    """

    WORKSPACE = "/workspace"

    def __init__(
        self,
        project_root: str | None = None,
        *,
        image: str = "plodder-sandbox:latest",
        session_api_key: str | None = None,
    ) -> None:
        import docker

        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.session_api_key = session_api_key or _new_session_api_key()
        self._client = docker.from_env()
        ensure_sandbox_image(self._client, image)
        self._container: Container = self._client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=False,
            command=["sleep", "infinity"],
            environment={"SESSION_API_KEY": self.session_api_key},
            working_dir=self.WORKSPACE,
            volumes={self.project_root: {"bind": self.WORKSPACE, "mode": "rw"}},
        )

    @property
    def container_id(self) -> str:
        return self._container.id

    def exec_bash(
        self,
        command: str,
        *,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
        demux: bool = False,
        **kwargs: Any,
    ) -> tuple[int, bytes | tuple[bytes, bytes]]:
        """
        Run ``command`` through ``/bin/bash -lc`` inside the container (``exec_run``).

        Extra kwargs are forwarded to ``Container.exec_run`` (e.g. ``user``, ``privileged``).
        """
        cmd = ["/bin/bash", "-lc", command]
        return self._container.exec_run(
            cmd,
            workdir=workdir or self.WORKSPACE,
            environment=environment,
            demux=demux,
            **kwargs,
        )

    def stop(self, *, remove: bool = True, timeout: int = 10) -> None:
        self._container.stop(timeout=timeout)
        if remove:
            self._container.remove()

    def __enter__(self) -> SimpleDockerSandbox:
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
