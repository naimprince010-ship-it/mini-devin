"""
GitHub Tool for Plodder

This tool provides access to GitHub integration features like branch creation,
commits, and pull requests.
"""

import json
import os
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import Field

from ..core.tool_interface import BaseTool
from ..integrations.github import GitHubIntegration, create_automated_pr_workflow
from ..integrations.gitlab_hosting import GitLabProjectIntegration
from ..integrations.remote_parser import parse_git_remote_url
from ..schemas.tools import BaseToolInput, BaseToolOutput, ToolStatus


def _resolve_base_branch(base: Optional[str]) -> Optional[str]:
    """``None`` / empty / ``default`` → let GitHub integration use the repo default branch."""
    if base is None:
        return None
    t = str(base).strip()
    if not t or t.lower() == "default":
        return None
    return t


class GitHubAction(str, Enum):
    """Available GitHub actions."""

    CREATE_BRANCH = "create_branch"
    COMMIT = "commit"
    CREATE_PR = "create_pr"
    AUTOMATED_WORKFLOW = "automated_workflow"
    GET_PR_STATUS = "get_pr_status"
    MERGE_PR = "merge_pr"


class GitHubToolInput(BaseToolInput):
    """Input for GitHub tool."""

    action: GitHubAction = Field(description="The GitHub action to perform")

    # Common/Branch parameters
    branch_name: Optional[str] = Field(default=None, description="Name of the branch")
    base_branch: Optional[str] = Field(
        default=None,
        description='Base branch (PR target / branch-off point). Omit or use "default" for the GitHub repo default (e.g. main or master).',
    )
    
    # Commit parameters
    commit_message: Optional[str] = Field(default=None, description="Commit message")
    files: Optional[List[str]] = Field(default=None, description="List of files to commit. If empty, commits all changes.")
    
    # PR parameters
    pr_title: Optional[str] = Field(default=None, description="Pull Request title")
    pr_description: Optional[str] = Field(default=None, description="Pull Request description")
    
    # Automated Workflow parameters
    task_description: Optional[str] = Field(default=None, description="Task description for automated workflow")

    pr_number: Optional[int] = Field(default=None, description="Pull request number (get_pr_status, merge_pr)")
    merge_method: Optional[str] = Field(
        default="squash",
        description="For merge_pr: squash | merge | rebase (repo rules permitting)",
    )
    draft: bool = Field(default=False, description="For create_pr: open as draft PR")
    assignees: Optional[List[str]] = Field(default=None, description="GitHub usernames to assign after create_pr")
    linked_issues: Optional[List[int]] = Field(
        default=None,
        description="Issue numbers to link (appends Closes #n lines to PR body)",
    )

    repo_path: str = Field(default=".", description="Local repository path")

class GitHubToolOutput(BaseToolOutput):
    """Output for GitHub tool."""
    success: bool = Field(description="Whether the action succeeded")
    message: str = Field(description="Result message")
    pr_url: Optional[str] = Field(default=None, description="Pull request URL, if applicable")

class GitHubTool(BaseTool[GitHubToolInput, GitHubToolOutput]):
    """Tool for GitHub operations."""
    
    name = "github"
    description = """Perform GitHub or GitLab operations (token + matching remote):
- GitHub: ``GITHUB_TOKEN`` / ``GH_TOKEN`` and ``github.com`` remote
- GitLab: ``GITLAB_TOKEN`` and ``gitlab.com`` or ``GITLAB_API_URL`` / ``GITLAB_HOST`` for self-hosted
- create_branch: Create and push a new branch from the default branch (``base_branch``: ``default``)
- commit: Commit and push on the current branch
- create_pr: Open a PR/MR (``base_branch``: ``default``)
- automated_workflow: Branch -> commit -> PR/MR in one shot
- get_pr_status: JSON status for a PR/MR number
- merge_pr: Merge (squash by default). Bitbucket remotes are not supported."""
    
    input_schema = GitHubToolInput
    output_schema = GitHubToolOutput
    
    def __init__(self, pb_token: Optional[str] = None, repo_name: Optional[str] = None):
        super().__init__()
        self.github_integration = GitHubIntegration(pb_token)
        self.repo_name = repo_name
        self.initialized = False
        self._gitlab: Optional[GitLabProjectIntegration] = None
        self._init_error: str = ""

    def _host_backend(self) -> Union[GitHubIntegration, GitLabProjectIntegration]:
        if self._gitlab is not None:
            return self._gitlab
        return self.github_integration

    async def _init_if_needed(self, repo_path: str):
        if self.initialized:
            return
        import subprocess

        url = ""
        try:
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            url = (result.stdout or "").strip()
        except Exception:
            url = ""

        parsed = parse_git_remote_url(url) if url else None
        if parsed is None and self.repo_name:
            parsed = parse_git_remote_url(f"https://github.com/{self.repo_name}.git")

        if parsed is None:
            self._init_error = "Could not parse remote.origin.url"
            return

        if parsed.platform == "bitbucket":
            self._init_error = "Bitbucket API is not supported by this tool; use GitHub or GitLab."
            return

        if parsed.platform == "unknown":
            self._init_error = (
                f"Unsupported git host {parsed.web_host!r}. "
                "Use github.com, gitlab.com, or set GITLAB_HOST / GITLAB_API_URL for self-hosted GitLab."
            )
            return

        if parsed.platform == "gitlab":
            gl_tok = (os.getenv("GITLAB_TOKEN") or "").strip() or (
                self.github_integration.github_token or ""
            ).strip()
            self._gitlab = GitLabProjectIntegration(
                gl_tok or None,
                parsed.path_with_namespace,
                parsed.api_v4_base,
            )
            self.initialized = await self._gitlab.initialize(repo_path)
            if not self.initialized:
                self._init_error = "GitLab init failed (token, path, or local repo)."
            return

        # GitHub
        self.repo_name = self.repo_name or parsed.path_with_namespace
        if self.repo_name:
            self.initialized = await self.github_integration.initialize(repo_path, self.repo_name)
            if not self.initialized:
                self._init_error = "GitHub init failed (token, PyGithub, or local repo)."
    
    async def _execute(self, input_data: GitHubToolInput) -> GitHubToolOutput:
        try:
            await self._init_if_needed(input_data.repo_path)
            
            if not self.initialized:
                msg = self._init_error or "Failed to initialize git hosting (token, remote, or local repo)."
                return GitHubToolOutput(
                    status=ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=False,
                    message=msg,
                )

            backend = self._host_backend()

            if input_data.action == GitHubAction.CREATE_BRANCH:
                if not input_data.branch_name:
                    raise ValueError("branch_name is required")
                success = await backend.create_feature_branch(
                    input_data.branch_name,
                    _resolve_base_branch(input_data.base_branch),
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if success else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=success,
                    message=f"Branch {input_data.branch_name} created." if success else "Failed to create branch."
                )
                
            elif input_data.action == GitHubAction.COMMIT:
                if not input_data.commit_message:
                    raise ValueError("commit_message is required")
                success = await backend.commit_changes(
                    input_data.commit_message, input_data.files
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if success else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=success,
                    message="Changes committed" if success else "Failed to commit"
                )
                
            elif input_data.action == GitHubAction.CREATE_PR:
                if not input_data.branch_name or not input_data.pr_title or not input_data.pr_description:
                    raise ValueError("branch_name, pr_title, pr_description are required")
                body = input_data.pr_description
                if input_data.linked_issues:
                    body += "\n\n" + "\n".join(f"Closes #{n}" for n in input_data.linked_issues)
                pr = await self.github_integration.create_pull_request(
                    title=input_data.pr_title,
                    description=body,
                    head_branch=input_data.branch_name,
                    base_branch=_resolve_base_branch(input_data.base_branch),
                    draft=input_data.draft,
                    assignees=input_data.assignees,
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if pr else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=bool(pr),
                    message=f"Created PR #{pr.number}" if pr else "Failed to create PR",
                    pr_url=pr.html_url if pr else None
                )
                
            elif input_data.action == GitHubAction.AUTOMATED_WORKFLOW:
                if not input_data.task_description:
                    raise ValueError("task_description is required")
                pr = await create_automated_pr_workflow(
                    hosting=backend,
                    task_description=input_data.task_description,
                    changes_made=input_data.files or ["Changed files"],
                    branch_name=input_data.branch_name,
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if pr else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=bool(pr),
                    message=f"Automated workflow complete, PR #{pr.number}" if pr else "Automated workflow failed",
                    pr_url=pr.html_url if pr else None
                )

            elif input_data.action == GitHubAction.GET_PR_STATUS:
                if input_data.pr_number is None:
                    raise ValueError("pr_number is required")
                status = await backend.get_pr_status(input_data.pr_number)
                if not status:
                    return GitHubToolOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message="Failed to load PR status (check pr_number and token).",
                    )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS,
                    execution_time_ms=0,
                    success=True,
                    message=json.dumps(status, indent=2, default=str),
                    pr_url=status.get("html_url"),
                )

            elif input_data.action == GitHubAction.MERGE_PR:
                if input_data.pr_number is None:
                    raise ValueError("pr_number is required")
                method = (input_data.merge_method or "squash").lower()
                if method not in ("squash", "merge", "rebase"):
                    raise ValueError("merge_method must be squash, merge, or rebase")
                ok = await backend.merge_pull_request(input_data.pr_number, merge_method=method)
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if ok else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=ok,
                    message=f"Merged PR #{input_data.pr_number} ({method})" if ok else f"Could not merge PR #{input_data.pr_number}",
                )

            return GitHubToolOutput(
                status=ToolStatus.FAILURE,
                execution_time_ms=0,
                success=False,
                message=f"Unknown action {input_data.action}"
            )

        except Exception as e:
            return GitHubToolOutput(
                status=ToolStatus.FAILURE,
                execution_time_ms=0,
                success=False,
                message=f"Error executing GitHub tool: {str(e)}"
            )

def create_github_tool() -> BaseTool:
    """Create a new github tool instance."""
    return GitHubTool()
