"""
Stable workspace paths for Railway and local dev.

Set ``PLODDER_AGENT_WORKSPACE_ROOT`` to a persistent volume mount so session files
survive redeploys. Each session uses a unique ``workspace_id`` subdirectory.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_agent_workspaces_root() -> str:
    """
    Root directory for per-session agent workspaces.

    - If ``PLODDER_AGENT_WORKSPACE_ROOT`` is set, use it (Railway volume, etc.).
    - Otherwise mirror ``mini_devin.api.app`` layout: sibling ``agent-workspace``
      of the ``mini_devin`` package checkout.
    """
    override = (os.environ.get("PLODDER_AGENT_WORKSPACE_ROOT") or "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))

    env_root = (os.environ.get("PLODDER_REPO_ROOT") or os.environ.get("MINI_DEVIN_REPO_ROOT") or "").strip()
    if env_root:
        abs_env = os.path.abspath(os.path.expanduser(env_root))
        return os.path.abspath(os.path.join(os.path.dirname(abs_env), "agent-workspace"))

    # Match ``mini_devin.api.app`` layout: sibling ``agent-workspace`` of the repo checkout.
    here = Path(__file__).resolve()
    mini_devin_pkg = here.parents[1]
    p = mini_devin_pkg.parent
    for _ in range(16):
        if (p / "pyproject.toml").is_file() and (p / "mini_devin").is_dir():
            return os.path.abspath(str((p.parent / "agent-workspace")))
        parent = p.parent
        if parent == p:
            break
        p = parent

    # Dev fallback: sibling of inferred repo folder (…/repo/mini_devin → …/agent-workspace)
    return os.path.abspath(str((mini_devin_pkg.parent.parent / "agent-workspace")))


def resolve_session_workspace_directory(
    workspace_id: str | None,
    stored_working_directory: str | None,
) -> str:
    """
    Return the absolute workspace directory for a session.

    When ``workspace_id`` is set, the path is always
    ``{get_agent_workspaces_root()}/{workspace_id}`` (directory is created).
    Otherwise returns ``stored_working_directory`` (legacy custom paths).
    """
    wid = (workspace_id or "").strip()
    if wid:
        root = get_agent_workspaces_root()
        path = os.path.join(root, wid)
        os.makedirs(path, exist_ok=True)
        return os.path.abspath(path)
    return os.path.abspath(os.path.expanduser(stored_working_directory or "."))
