"""Unit tests for the GitHub tool."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from mini_devin.schemas.tools import ToolStatus
from mini_devin.tools.github import GitHubTool


class StubBackend:
    async def get_repository_context(self, issue_limit: int = 20):
        return {"default_branch": "main", "labels": [{"name": "bug"}], "issue_limit": issue_limit}

    async def get_issue(self, issue_number: int, include_comments: bool = True):
        return {
            "number": issue_number,
            "title": "Fix login bug",
            "comments": [{"body": "Please handle edge cases"}] if include_comments else [],
        }

    async def list_issues(self, state: str = "open", limit: int = 20, include_comments: bool = False):
        return [{"number": 1, "state": state, "comments": [] if not include_comments else [{}]}][:limit]

    async def list_pull_requests(self, state: str = "open", limit: int = 20):
        return [{"number": 7, "state": state}][:limit]

    async def get_pull_request(self, pr_number: int, include_comments: bool = True):
        return {
            "number": pr_number,
            "html_url": f"https://github.com/acme/repo/pull/{pr_number}",
            "comments": [{"body": "looks good"}] if include_comments else [],
        }

    async def create_pull_request(
        self,
        title: str,
        description: str,
        head_branch: str,
        base_branch: str | None = None,
        draft: bool = False,
        assignees: list[str] | None = None,
    ):
        return SimpleNamespace(number=42, html_url="https://github.com/acme/repo/pull/42")

    async def get_pr_status(self, pr_number: int):
        return {"html_url": f"https://github.com/acme/repo/pull/{pr_number}", "mergeable": True}


@pytest.fixture
def github_tool() -> GitHubTool:
    tool = GitHubTool()
    tool.initialized = True
    backend = StubBackend()
    tool._host_backend = lambda: backend  # type: ignore[method-assign]
    return tool


class TestGitHubTool:
    def test_get_repo_context(self, github_tool: GitHubTool) -> None:
        result = asyncio.run(github_tool.execute({"action": "get_repo_context"}))
        assert result.status == ToolStatus.SUCCESS
        assert result.data["default_branch"] == "main"

    def test_get_issue(self, github_tool: GitHubTool) -> None:
        result = asyncio.run(github_tool.execute({"action": "get_issue", "issue_number": 123}))
        assert result.status == ToolStatus.SUCCESS
        assert result.data["number"] == 123
        assert result.data["comments"]

    def test_get_pull_request(self, github_tool: GitHubTool) -> None:
        result = asyncio.run(github_tool.execute({"action": "get_pull_request", "pr_number": 7}))
        assert result.status == ToolStatus.SUCCESS
        assert result.pr_url == "https://github.com/acme/repo/pull/7"

    def test_create_pr_autogenerates_description(self, github_tool: GitHubTool, tmp_path) -> None:
        result = asyncio.run(
            github_tool.execute(
                {
                    "action": "create_pr",
                    "repo_path": str(tmp_path),
                    "branch_name": "plodder/fix-login",
                    "pr_title": "Fix login bug",
                    "task_description": "Fix the login edge case",
                    "linked_issues": [123],
                }
            )
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.pr_url == "https://github.com/acme/repo/pull/42"
        assert "Closes #123" in json.dumps(result.data)
