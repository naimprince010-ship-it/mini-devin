"""
Structured git workflow tool for Plodder.

This tool exposes common git workflow actions as explicit agent-callable
operations so the model does not have to shell out for every branch/add/commit
step when a PR-based workflow is desired.
"""

from __future__ import annotations

import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field

from ..core.tool_interface import BaseTool
from ..schemas.tools import BaseToolInput, BaseToolOutput, ToolStatus


def _is_truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


class GitAction(str, Enum):
    STATUS = "status"
    CHECKOUT_BRANCH = "checkout_branch"
    ADD = "add"
    COMMIT = "commit"
    PUSH = "push"
    DIFF = "diff"


class GitToolInput(BaseToolInput):
    action: GitAction = Field(description="Git action to perform")
    repo_path: str = Field(default=".", description="Local repository path")
    branch_name: Optional[str] = Field(
        default=None,
        description="Branch name for checkout_branch / push",
    )
    create: bool = Field(
        default=False,
        description="For checkout_branch: create the branch with `git checkout -b`",
    )
    base_branch: Optional[str] = Field(
        default=None,
        description="Optional starting point for checkout_branch when create=true",
    )
    files: list[str] = Field(
        default_factory=list,
        description="Files for add / commit. Empty means all changes.",
    )
    commit_message: Optional[str] = Field(
        default=None,
        description="Commit message for commit",
    )
    remote: str = Field(default="origin", description="Remote name for push")
    set_upstream: bool = Field(
        default=True,
        description="For push: add -u when pushing a branch for the first time",
    )
    allow_default_branch_push: bool = Field(
        default=False,
        description="Allow push to origin default branch. Otherwise blocked unless env override is set.",
    )
    staged_only: bool = Field(
        default=False,
        description="For diff: show only staged changes",
    )


class GitToolOutput(BaseToolOutput):
    success: bool = Field(description="Whether the action succeeded")
    message: str = Field(description="Human readable result message")
    stdout: str = Field(default="", description="Git stdout")
    stderr: str = Field(default="", description="Git stderr")
    branch_name: Optional[str] = Field(default=None, description="Active or target branch")
    commit_sha: Optional[str] = Field(default=None, description="Commit SHA when relevant")


class GitTool(BaseTool[GitToolInput, GitToolOutput]):
    name = "git"
    description = (
        "Perform structured git workflow actions: inspect status/diff, checkout a new "
        "branch, add files, commit changes, and push safely."
    )
    input_schema = GitToolInput
    output_schema = GitToolOutput

    async def _run_git(self, repo_path: str, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (
            proc.returncode,
            (stdout or b"").decode(errors="replace"),
            (stderr or b"").decode(errors="replace"),
        )

    async def _current_branch(self, repo_path: str) -> Optional[str]:
        code, stdout, _ = await self._run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
        if code != 0:
            return None
        branch = stdout.strip()
        return branch or None

    async def _origin_default_branch(self, repo_path: str) -> Optional[str]:
        code, stdout, _ = await self._run_git(
            repo_path, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"
        )
        if code == 0 and stdout.strip():
            full = stdout.strip()
            return full.rsplit("/", 1)[-1] if "/" in full else full
        code, stdout, _ = await self._run_git(repo_path, "remote", "show", "origin")
        if code != 0:
            return None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("HEAD branch:"):
                value = line.split(":", 1)[1].strip()
                return value or None
        return None

    async def _resolve_head_sha(self, repo_path: str) -> Optional[str]:
        code, stdout, _ = await self._run_git(repo_path, "rev-parse", "HEAD")
        if code != 0:
            return None
        sha = stdout.strip()
        return sha or None

    async def _ensure_repo(self, repo_path: str) -> Optional[str]:
        abs_repo = str(Path(repo_path).resolve())
        if not Path(abs_repo).exists():
            return None
        code, _, _ = await self._run_git(abs_repo, "rev-parse", "--is-inside-work-tree")
        if code != 0:
            return None
        return abs_repo

    async def _execute(self, input_data: GitToolInput) -> GitToolOutput:
        repo_path = await self._ensure_repo(input_data.repo_path)
        if repo_path is None:
            return GitToolOutput(
                status=ToolStatus.FAILURE,
                execution_time_ms=0,
                success=False,
                message="repo_path is not a git repository",
                stderr=f"Invalid git repository: {input_data.repo_path}",
            )

        if input_data.action == GitAction.STATUS:
            code, stdout, stderr = await self._run_git(repo_path, "status", "--short", "--branch")
            branch = await self._current_branch(repo_path)
            return GitToolOutput(
                status=ToolStatus.SUCCESS if code == 0 else ToolStatus.FAILURE,
                execution_time_ms=0,
                success=code == 0,
                message="Git status loaded" if code == 0 else "Failed to load git status",
                stdout=stdout,
                stderr=stderr,
                branch_name=branch,
            )

        if input_data.action == GitAction.CHECKOUT_BRANCH:
            if not input_data.branch_name:
                raise ValueError("branch_name is required")
            args = ["checkout"]
            if input_data.create:
                args.append("-b")
            args.append(input_data.branch_name)
            if input_data.create and input_data.base_branch:
                args.append(input_data.base_branch)
            code, stdout, stderr = await self._run_git(repo_path, *args)
            branch = await self._current_branch(repo_path)
            return GitToolOutput(
                status=ToolStatus.SUCCESS if code == 0 else ToolStatus.FAILURE,
                execution_time_ms=0,
                success=code == 0,
                message=(
                    f"Checked out branch {branch or input_data.branch_name}"
                    if code == 0
                    else f"Failed to checkout branch {input_data.branch_name}"
                ),
                stdout=stdout,
                stderr=stderr,
                branch_name=branch or input_data.branch_name,
            )

        if input_data.action == GitAction.ADD:
            args = ["add"]
            args.extend(input_data.files or ["-A"])
            code, stdout, stderr = await self._run_git(repo_path, *args)
            return GitToolOutput(
                status=ToolStatus.SUCCESS if code == 0 else ToolStatus.FAILURE,
                execution_time_ms=0,
                success=code == 0,
                message="Added changes to index" if code == 0 else "Failed to add changes",
                stdout=stdout,
                stderr=stderr,
                branch_name=await self._current_branch(repo_path),
            )

        if input_data.action == GitAction.COMMIT:
            if not input_data.commit_message:
                raise ValueError("commit_message is required")
            add_args = ["add"]
            add_args.extend(input_data.files or ["-A"])
            add_code, add_stdout, add_stderr = await self._run_git(repo_path, *add_args)
            if add_code != 0:
                return GitToolOutput(
                    status=ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=False,
                    message="Failed to stage changes before commit",
                    stdout=add_stdout,
                    stderr=add_stderr,
                    branch_name=await self._current_branch(repo_path),
                )
            code, stdout, stderr = await self._run_git(
                repo_path, "commit", "-m", input_data.commit_message
            )
            if code != 0 and "nothing to commit" in f"{stdout}\n{stderr}".lower():
                return GitToolOutput(
                    status=ToolStatus.SUCCESS,
                    execution_time_ms=0,
                    success=True,
                    message="Nothing to commit",
                    stdout=stdout,
                    stderr=stderr,
                    branch_name=await self._current_branch(repo_path),
                    commit_sha=await self._resolve_head_sha(repo_path),
                )
            return GitToolOutput(
                status=ToolStatus.SUCCESS if code == 0 else ToolStatus.FAILURE,
                execution_time_ms=0,
                success=code == 0,
                message="Committed changes" if code == 0 else "Failed to commit changes",
                stdout=stdout,
                stderr=stderr,
                branch_name=await self._current_branch(repo_path),
                commit_sha=await self._resolve_head_sha(repo_path) if code == 0 else None,
            )

        if input_data.action == GitAction.PUSH:
            branch = input_data.branch_name or await self._current_branch(repo_path)
            if not branch:
                return GitToolOutput(
                    status=ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=False,
                    message="Could not determine current branch",
                )
            default_branch = await self._origin_default_branch(repo_path)
            allow_default = input_data.allow_default_branch_push or _is_truthy_env(
                "PLODDER_GIT_PUSH_DEFAULT_BRANCH"
            )
            if default_branch and branch == default_branch and not allow_default:
                return GitToolOutput(
                    status=ToolStatus.BLOCKED,
                    execution_time_ms=0,
                    success=False,
                    message=(
                        "Push to the remote default branch is blocked by policy. "
                        "Create a feature branch or allow it explicitly."
                    ),
                    branch_name=branch,
                )
            args = ["push"]
            if input_data.set_upstream:
                args.append("-u")
            args.extend([input_data.remote, branch])
            code, stdout, stderr = await self._run_git(repo_path, *args)
            return GitToolOutput(
                status=ToolStatus.SUCCESS if code == 0 else ToolStatus.FAILURE,
                execution_time_ms=0,
                success=code == 0,
                message=f"Pushed branch {branch}" if code == 0 else f"Failed to push branch {branch}",
                stdout=stdout,
                stderr=stderr,
                branch_name=branch,
            )

        if input_data.action == GitAction.DIFF:
            args = ["diff"]
            if input_data.staged_only:
                args.append("--cached")
            code, stdout, stderr = await self._run_git(repo_path, *args)
            return GitToolOutput(
                status=ToolStatus.SUCCESS if code == 0 else ToolStatus.FAILURE,
                execution_time_ms=0,
                success=code == 0,
                message="Git diff loaded" if code == 0 else "Failed to load git diff",
                stdout=stdout,
                stderr=stderr,
                branch_name=await self._current_branch(repo_path),
            )

        return GitToolOutput(
            status=ToolStatus.FAILURE,
            execution_time_ms=0,
            success=False,
            message=f"Unsupported git action: {input_data.action}",
        )


def create_git_tool() -> BaseTool:
    return GitTool()
