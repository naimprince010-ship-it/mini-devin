"""
Create code-execution sandboxes (Docker or E2B) from environment configuration.

Environment:
- ``SANDBOX_BACKEND``: ``docker`` (default) or ``e2b``
- ``E2B_API_KEY``: required when backend is ``e2b``
- ``E2B_SANDBOX_TEMPLATE``: optional E2B template id
"""

from __future__ import annotations

import os
from typing import Any


def get_sandbox_backend() -> str:
    return os.environ.get("SANDBOX_BACKEND", "docker").strip().lower()


def create_execution_sandbox(
    repo_path: str,
    memory_limit: str = "2g",
    cpu_limit: float = 2.0,
    network_enabled: bool = False,
    security_level: Any = None,
    mount_allowlist: list[str] | None = None,
) -> Any:
    """
    Return a sandbox compatible with :class:`DockerSandbox` (``is_running``, ``start``,
    ``execute``, ``stop``, ``container_id``).

    Falls back to Docker if ``e2b`` is requested but ``E2B_API_KEY`` is missing.
    """
    backend = get_sandbox_backend()
    if backend == "e2b":
        key = os.environ.get("E2B_API_KEY", "").strip()
        if not key:
            print("[Sandbox] SANDBOX_BACKEND=e2b but E2B_API_KEY is empty; using docker")
        else:
            from .e2b_sandbox import E2BSandbox

            return E2BSandbox(repo_path)

    from .docker_sandbox import SecurityLevel, create_sandbox

    level = security_level if security_level is not None else SecurityLevel.STANDARD
    return create_sandbox(
        repo_path=repo_path,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        network_enabled=network_enabled,
        security_level=level,
        mount_allowlist=mount_allowlist,
    )
