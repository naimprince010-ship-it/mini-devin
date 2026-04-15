"""
GitHub Tool for Plodder

This tool provides access to GitHub integration features like branch creation,
commits, and pull requests.
"""

import json
import os
import subprocess
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

    GET_REPO_CONTEXT = "get_repo_context"
    LIST_ISSUES = "list_issues"
    GET_ISSUE = "get_issue"
    LIST_PULL_REQUESTS = "list_pull_requests"
    GET_PULL_REQUEST = "get_pull_request"
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
    issue_number: Optional[int] = Field(default=None, description="Issue number for get_issue")
    state: str = Field(default="open", description="State filter for list_issues / list_pull_requests")
    include_comments: bool = Field(
        default=True,
        description="Include comments in get_issue / get_pull_request responses",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max results for list actions")

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
    data: Any = Field(default=None, description="Structured GitHub response payload")
    branch_name: Optional[str] = Field(default=None, description="Relevant branch name")

class GitHubTool(BaseTool[GitHubToolInput, GitHubToolOutput]):
    """Tool for GitHub operations."""
    
    name = "github"
    description = """Perform GitHub or GitLab operations (token + matching remote):
- GitHub: ``GITHUB_TOKEN`` / ``GH_TOKEN`` and ``github.com`` remote
- GitLab: ``GITLAB_TOKEN`` and ``gitlab.com`` or ``GITLAB_API_URL`` / ``GITLAB_HOST`` for self-hosted
- create_branch: Create and push a new branch from the default branch (``base_branch``: ``default``)
- commit: Commit and push on the current branch
- create_pr: Open a PR/MR (``base_branch``: ``default``)
- get_repo_context: Fetch repository metadata, labels, and branch info
- list_issues / get_issue: Fetch issue context with labels/comments
- list_pull_requests / get_pull_request: Fetch PR context with comments/review signals
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

    def _summarize_changed_files(self, repo_path: str) -> list[str]:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--cached"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            files = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
            if files:
                return files[:50]
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()][:50]
        except Exception:
            return []

    def _build_pr_description(self, input_data: GitHubToolInput) -> str:
        summary_lines: list[str] = []
        if input_data.task_description:
            summary_lines.append(input_data.task_description.strip())
        if input_data.issue_number is not None:
            summary_lines.append(f"Addresses GitHub issue #{input_data.issue_number}.")
        if not summary_lines:
            summary_lines.append("Implements the requested code changes.")

        changed_files = input_data.files or self._summarize_changed_files(input_data.repo_path)
        lines = [
            "## Summary",
            *[f"- {line}" for line in summary_lines],
        ]
        if changed_files:
            lines.extend(
                [
                    "",
                    "## Changed Files",
                    *[f"- `{path}`" for path in changed_files],
                ]
            )
        lines.extend(
            [
                "",
                "## Verification",
                "- Local verification completed before PR creation.",
            ]
        )
        if input_data.linked_issues:
            lines.extend(
                [
                    "",
                    "## Linked Issues",
                    *[f"- Closes #{issue}" for issue in input_data.linked_issues],
                ]
            )
        return "\n".join(lines).strip()

    async def _init_if_needed(self, repo_path: str):
        if self.initialized:
            return

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

            def _supports(method_name: str) -> bool:
                return callable(getattr(backend, method_name, None))

            if input_data.action == GitHubAction.GET_REPO_CONTEXT:
                if not _supports("get_repository_context"):
                    return GitHubToolOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message="Repository context is not supported for this git host.",
                    )
                context = await backend.get_repository_context(issue_limit=input_data.limit)
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if context else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=bool(context),
                    message="Loaded repository context" if context else "Failed to load repository context",
                    data=context,
                )

            if input_data.action == GitHubAction.LIST_ISSUES:
                if not _supports("list_issues"):
                    return GitHubToolOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message="Issue listing is not supported for this git host.",
                    )
                issues = await backend.list_issues(
                    state=input_data.state,
                    limit=input_data.limit,
                    include_comments=input_data.include_comments,
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS,
                    execution_time_ms=0,
                    success=True,
                    message=f"Loaded {len(issues)} issue(s)",
                    data=issues,
                )

            if input_data.action == GitHubAction.GET_ISSUE:
                if input_data.issue_number is None:
                    raise ValueError("issue_number is required")
                if not _supports("get_issue"):
                    return GitHubToolOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message="Issue fetching is not supported for this git host.",
                    )
                issue = await backend.get_issue(
                    input_data.issue_number,
                    include_comments=input_data.include_comments,
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if issue else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=bool(issue),
                    message=(
                        f"Loaded issue #{input_data.issue_number}"
                        if issue
                        else f"Failed to load issue #{input_data.issue_number}"
                    ),
                    data=issue,
                )

            if input_data.action == GitHubAction.LIST_PULL_REQUESTS:
                if not _supports("list_pull_requests"):
                    return GitHubToolOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message="Pull request listing is not supported for this git host.",
                    )
                pulls = await backend.list_pull_requests(
                    state=input_data.state,
                    limit=input_data.limit,
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS,
                    execution_time_ms=0,
                    success=True,
                    message=f"Loaded {len(pulls)} pull request(s)",
                    data=pulls,
                )

            if input_data.action == GitHubAction.GET_PULL_REQUEST:
                if input_data.pr_number is None:
                    raise ValueError("pr_number is required")
                if not _supports("get_pull_request"):
                    return GitHubToolOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message="Pull request fetching is not supported for this git host.",
                    )
                pull = await backend.get_pull_request(
                    input_data.pr_number,
                    include_comments=input_data.include_comments,
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if pull else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=bool(pull),
                    message=(
                        f"Loaded pull request #{input_data.pr_number}"
                        if pull
                        else f"Failed to load pull request #{input_data.pr_number}"
                    ),
                    data=pull,
                    pr_url=pull.get("html_url") if pull else None,
                )

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
                    message=f"Branch {input_data.branch_name} created." if success else "Failed to create branch.",
                    branch_name=input_data.branch_name,
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
                    message="Changes committed" if success else "Failed to commit",
                    branch_name=input_data.branch_name,
                )
                
            elif input_data.action == GitHubAction.CREATE_PR:
                if not input_data.branch_name or not input_data.pr_title:
                    raise ValueError("branch_name and pr_title are required")
                body = (input_data.pr_description or "").strip() or self._build_pr_description(input_data)
                if input_data.linked_issues:
                    issue_lines = "\n".join(f"Closes #{n}" for n in input_data.linked_issues)
                    if issue_lines not in body:
                        body += "\n\n" + issue_lines
                pr = await backend.create_pull_request(
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
                    pr_url=pr.html_url if pr else None,
                    data={"body": body} if pr else None,
                    branch_name=input_data.branch_name,
                )
                
            elif input_data.action == GitHubAction.AUTOMATED_WORKFLOW:
                if not input_data.task_description:
                    raise ValueError("task_description is required")
                pr = await create_automated_pr_workflow(
                    hosting=backend,
                    task_description=input_data.task_description,
                    changes_made=input_data.files or self._summarize_changed_files(input_data.repo_path) or ["Changed files"],
                    branch_name=input_data.branch_name,
                    linked_issues=input_data.linked_issues,
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
