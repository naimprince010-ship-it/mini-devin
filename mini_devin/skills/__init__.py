"""Repo-root playbooks (``skills/*.md``) for orchestrator prompt injection."""

from .playbook import (
    discover_repo_root,
    format_playbooks_for_prompt,
    load_playbook_markdown,
    playbook_tags_from_env,
)

__all__ = [
    "discover_repo_root",
    "format_playbooks_for_prompt",
    "load_playbook_markdown",
    "playbook_tags_from_env",
]
