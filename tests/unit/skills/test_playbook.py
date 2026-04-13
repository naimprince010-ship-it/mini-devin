"""Tests for repo-root playbook loading."""

from pathlib import Path

from mini_devin.skills.playbook import format_playbooks_for_prompt, load_playbook_markdown


def test_load_code_review_from_repo_root():
    root = Path(__file__).resolve().parents[3]
    text = load_playbook_markdown(root, "code_review")
    assert text
    assert "Code review" in text or "review" in text.lower()


def test_format_playbooks_includes_both():
    root = Path(__file__).resolve().parents[3]
    block = format_playbooks_for_prompt(root, ["code_review", "refactor", "missing_tag_xyz"])
    assert "code_review" in block
    assert "refactor" in block
    assert "missing_tag_xyz" not in block
