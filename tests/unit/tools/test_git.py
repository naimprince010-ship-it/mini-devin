"""Unit tests for the git tool."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from mini_devin.schemas.tools import ToolStatus
from mini_devin.tools.git import GitTool


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-m", "init")
    _git(tmp_path, "branch", "-M", "main")
    return tmp_path


class TestGitTool:
    def test_status_returns_branch(self, repo: Path) -> None:
        tool = GitTool()
        result = asyncio.run(tool.execute({"action": "status", "repo_path": str(repo)}))
        assert result.status == ToolStatus.SUCCESS
        assert result.branch_name == "main"

    def test_checkout_branch_create(self, repo: Path) -> None:
        tool = GitTool()
        result = asyncio.run(
            tool.execute(
                {
                    "action": "checkout_branch",
                    "repo_path": str(repo),
                    "branch_name": "feature/test-branch",
                    "create": True,
                }
            )
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.branch_name == "feature/test-branch"

    def test_commit_stages_and_commits_changes(self, repo: Path) -> None:
        tool = GitTool()
        (repo / "README.md").write_text("updated\n", encoding="utf-8")
        result = asyncio.run(
            tool.execute(
                {
                    "action": "commit",
                    "repo_path": str(repo),
                    "commit_message": "feat: update readme",
                }
            )
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.commit_sha

    def test_push_blocks_default_branch_by_policy(self, repo: Path) -> None:
        tool = GitTool()

        async def fake_default_branch(repo_path: str) -> str:
            return "main"

        tool._origin_default_branch = fake_default_branch  # type: ignore[method-assign]
        result = asyncio.run(tool.execute({"action": "push", "repo_path": str(repo)}))
        assert result.status == ToolStatus.BLOCKED
