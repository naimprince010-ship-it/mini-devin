"""
Load repo-root ``skills/*.md`` playbooks and format them for LLM prompt injection.

No content is copied from OpenHands; this follows the common “skills folder + prompt block” pattern.

Resolution order for repository root:
1. ``PLODDER_REPO_ROOT`` / ``MINI_DEVIN_REPO_ROOT`` if set
2. Walk parents from ``mini_devin/skills/playbook.py`` until ``skills/`` exists alongside ``pyproject.toml``
3. ``Path.cwd()``
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$", re.I)


def discover_repo_root() -> Path:
    env = (os.environ.get("PLODDER_REPO_ROOT") or os.environ.get("MINI_DEVIN_REPO_ROOT") or "").strip()
    if env:
        p = Path(env).resolve()
        if (p / "skills").is_dir():
            return p

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "skills").is_dir() and (parent / "pyproject.toml").is_file():
            return parent

    cwd = Path.cwd().resolve()
    if (cwd / "skills").is_dir():
        return cwd
    return cwd


def load_playbook_markdown(repo_root: str | Path, tag: str) -> str | None:
    """Return file text for ``skills/<tag>.md`` or None if missing."""
    if not _TAG_RE.match(tag.strip()):
        return None
    path = Path(repo_root).resolve() / "skills" / f"{tag.strip()}.md"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def format_playbooks_for_prompt(
    repo_root: str | Path,
    tags: Sequence[str],
    *,
    max_chars_per_playbook: int = 12_000,
) -> str:
    """
    Build a single markdown block for orchestrator / agent system or task prompts.

    Skips missing tags silently (caller controls which tags are active).
    """
    root = Path(repo_root).resolve()
    parts: list[str] = []
    for raw in tags:
        tag = str(raw).strip()
        if not tag:
            continue
        body = load_playbook_markdown(root, tag)
        if not body:
            continue
        if len(body) > max_chars_per_playbook:
            body = body[:max_chars_per_playbook] + "\n\n…(truncated)…"
        parts.append(f"### Playbook: `{tag}`\n\n{body}")
    if not parts:
        return ""
    return "## Active playbooks\n\n" + "\n\n---\n\n".join(parts) + "\n"


def playbook_tags_from_env() -> list[str]:
    raw = (os.environ.get("PLODDER_PLAYBOOK_TAGS") or "").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]
