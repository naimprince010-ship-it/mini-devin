"""
On new workspace / task start: index code, capture README into working memory.

This gives a lightweight "self-learning on repo open" behaviour without training weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


README_NAMES = (
    "README.md",
    "README.MD",
    "readme.md",
    "Readme.md",
    "README.rst",
)


def _read_readme(workspace: Path) -> tuple[str, str]:
    for name in README_NAMES:
        p = workspace / name
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="ignore")
            return name, text
    return "", ""


def run_workspace_bootstrap(agent: Any) -> dict[str, Any]:
    """
    Index workspace for retrieval and stash README summary in working memory.

    Returns small stats dict for logging.
    """
    stats: dict[str, Any] = {
        "indexed": False,
        "readme_chars": 0,
        "readme_file": "",
        "already_git_checkout": False,
    }
    wd_raw = getattr(agent, "working_directory", None) or ""
    workspace = Path(wd_raw)
    if not workspace.is_dir():
        return stats

    git_meta = workspace / ".git"
    if git_meta.exists():
        stats["already_git_checkout"] = True
        try:
            agent.add_to_memory(
                "## Workspace is already a Git repository\n"
                "The session working directory is a checkout (`.git` is present). "
                "**Do not `git clone` into `.`** — explore with `editor` or run commands from this root. "
                "Only clone elsewhere if the user explicitly asked for a **nested** copy in a new subfolder.",
                item_type="constraint",
                priority="critical",
            )
        except Exception:
            pass

    try:
        agent.index_workspace(force=False)
        stats["indexed"] = True
    except Exception:
        pass

    name, text = _read_readme(workspace)
    if text:
        cap = 12_000
        snippet = text[:cap]
        if len(text) > cap:
            snippet += "\n\n[README truncated…]"
        block = (
            f"## Repository overview (auto from `{name}`)\n\n{snippet}"
        )
        try:
            agent.add_to_memory(block, item_type="context", priority="high")
        except Exception:
            pass
        stats["readme_chars"] = len(text)
        stats["readme_file"] = name

    # Optional: one-line stack hint from package.json / pyproject.toml
    hints: list[str] = []
    pj = workspace / "package.json"
    if pj.is_file():
        hints.append("Found `package.json` (Node frontend or tooling likely).")
    pyproject = workspace / "pyproject.toml"
    if pyproject.is_file():
        hints.append("Found `pyproject.toml` (Python/Poetry project likely).")
    if hints:
        try:
            agent.add_to_memory("\n".join(hints), item_type="context", priority="medium")
        except Exception:
            pass

    return stats
