"""Sanity checks for GitHub integration helpers."""

from mini_devin.integrations.github import _sanitize_branch_fragment
from mini_devin.tools.github import _resolve_base_branch


def test_sanitize_branch_fragment():
    assert _sanitize_branch_fragment("Fix login!!!  ") == "fix-login"
    assert _sanitize_branch_fragment("   ") == "task"


def test_resolve_base_branch():
    assert _resolve_base_branch(None) is None
    assert _resolve_base_branch("") is None
    assert _resolve_base_branch("  default  ") is None
    assert _resolve_base_branch("main") == "main"
