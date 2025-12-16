"""
Git Manager for Mini-Devin

This module provides git operations for diff inspection, checkpoint creation,
and rollback capabilities to support safe agent operations.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class GitStatus(str, Enum):
    """Status of git operations."""
    SUCCESS = "success"
    FAILURE = "failure"
    NO_REPO = "no_repo"
    DIRTY = "dirty"
    CONFLICT = "conflict"


@dataclass
class GitDiff:
    """Represents a git diff."""
    files_changed: list[str]
    insertions: int
    deletions: int
    diff_text: str
    is_empty: bool


@dataclass
class GitCheckpoint:
    """A checkpoint for rollback purposes."""
    checkpoint_id: str
    commit_hash: str
    branch_name: str
    message: str
    created_at: datetime
    files_at_checkpoint: list[str]


@dataclass
class GitOperationResult:
    """Result of a git operation."""
    status: GitStatus
    message: str
    data: dict[str, Any] | None = None
    stdout: str = ""
    stderr: str = ""


class GitManager:
    """
    Manages git operations for the agent.
    
    Provides capabilities for:
    - Diff inspection (see what changed)
    - Checkpoint creation (save state for rollback)
    - Rollback (revert to previous state)
    - Branch management
    - Commit operations
    """
    
    def __init__(
        self,
        working_directory: str,
        verbose: bool = True,
    ):
        self.working_directory = working_directory
        self.verbose = verbose
        self._checkpoints: dict[str, GitCheckpoint] = {}
    
    async def is_git_repo(self) -> bool:
        """Check if the working directory is a git repository."""
        result = await self._run_git_command("rev-parse --is-inside-work-tree")
        return result.status == GitStatus.SUCCESS and "true" in result.stdout.lower()
    
    async def init_repo(self) -> GitOperationResult:
        """Initialize a new git repository."""
        return await self._run_git_command("init")
    
    async def get_current_branch(self) -> str | None:
        """Get the current branch name."""
        result = await self._run_git_command("rev-parse --abbrev-ref HEAD")
        if result.status == GitStatus.SUCCESS:
            return result.stdout.strip()
        return None
    
    async def get_current_commit(self) -> str | None:
        """Get the current commit hash."""
        result = await self._run_git_command("rev-parse HEAD")
        if result.status == GitStatus.SUCCESS:
            return result.stdout.strip()
        return None
    
    async def get_status(self) -> GitOperationResult:
        """Get git status."""
        result = await self._run_git_command("status --porcelain")
        
        if result.status == GitStatus.SUCCESS:
            is_clean = not result.stdout.strip()
            result.data = {
                "is_clean": is_clean,
                "changed_files": self._parse_status_output(result.stdout),
            }
        
        return result
    
    async def get_diff(
        self,
        staged: bool = False,
        commit: str | None = None,
    ) -> GitDiff:
        """Get the current diff."""
        if commit:
            cmd = f"diff {commit}"
        elif staged:
            cmd = "diff --staged"
        else:
            cmd = "diff"
        
        result = await self._run_git_command(cmd)
        
        if result.status != GitStatus.SUCCESS:
            return GitDiff(
                files_changed=[],
                insertions=0,
                deletions=0,
                diff_text="",
                is_empty=True,
            )
        
        diff_text = result.stdout
        
        # Get diff stats
        stat_cmd = f"{cmd} --stat"
        stat_result = await self._run_git_command(stat_cmd)
        
        files_changed = []
        insertions = 0
        deletions = 0
        
        if stat_result.status == GitStatus.SUCCESS:
            files_changed, insertions, deletions = self._parse_diff_stat(
                stat_result.stdout
            )
        
        return GitDiff(
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions,
            diff_text=diff_text,
            is_empty=not diff_text.strip(),
        )
    
    async def get_file_diff(self, file_path: str) -> str:
        """Get diff for a specific file."""
        result = await self._run_git_command(f"diff -- {file_path}")
        return result.stdout if result.status == GitStatus.SUCCESS else ""
    
    async def create_checkpoint(
        self,
        checkpoint_id: str,
        message: str = "Agent checkpoint",
    ) -> GitCheckpoint | None:
        """
        Create a checkpoint for potential rollback.
        
        This stages and commits all current changes, creating a point
        we can roll back to if needed.
        """
        # Get current state
        branch = await self.get_current_branch()
        if not branch:
            return None
        
        # Stage all changes
        await self._run_git_command("add -A")
        
        # Get list of staged files
        status_result = await self._run_git_command("diff --staged --name-only")
        files = status_result.stdout.strip().split("\n") if status_result.stdout.strip() else []
        
        # Commit with checkpoint message
        commit_msg = f"[checkpoint:{checkpoint_id}] {message}"
        commit_result = await self._run_git_command(f'commit -m "{commit_msg}" --allow-empty')
        
        if commit_result.status != GitStatus.SUCCESS:
            # If nothing to commit, get current commit
            pass
        
        # Get the commit hash
        commit_hash = await self.get_current_commit()
        if not commit_hash:
            return None
        
        checkpoint = GitCheckpoint(
            checkpoint_id=checkpoint_id,
            commit_hash=commit_hash,
            branch_name=branch,
            message=message,
            created_at=datetime.utcnow(),
            files_at_checkpoint=files,
        )
        
        self._checkpoints[checkpoint_id] = checkpoint
        
        if self.verbose:
            print(f"[Git] Created checkpoint '{checkpoint_id}' at {commit_hash[:8]}")
        
        return checkpoint
    
    async def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        hard: bool = False,
    ) -> GitOperationResult:
        """
        Rollback to a previous checkpoint.
        
        Args:
            checkpoint_id: The checkpoint to roll back to
            hard: If True, discard all changes. If False, keep changes as unstaged.
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return GitOperationResult(
                status=GitStatus.FAILURE,
                message=f"Checkpoint not found: {checkpoint_id}",
            )
        
        # Reset to the checkpoint commit
        reset_type = "--hard" if hard else "--soft"
        result = await self._run_git_command(f"reset {reset_type} {checkpoint.commit_hash}")
        
        if result.status == GitStatus.SUCCESS:
            if self.verbose:
                print(f"[Git] Rolled back to checkpoint '{checkpoint_id}'")
        
        return result
    
    async def rollback_file(self, file_path: str) -> GitOperationResult:
        """Rollback a specific file to its last committed state."""
        return await self._run_git_command(f"checkout -- {file_path}")
    
    async def rollback_last_commit(self, soft: bool = True) -> GitOperationResult:
        """
        Rollback the last commit.
        
        Args:
            soft: If True, keep changes staged. If False, discard changes.
        """
        reset_type = "--soft" if soft else "--hard"
        return await self._run_git_command(f"reset {reset_type} HEAD~1")
    
    async def stage_files(self, files: list[str] | None = None) -> GitOperationResult:
        """Stage files for commit."""
        if files:
            files_str = " ".join(f'"{f}"' for f in files)
            return await self._run_git_command(f"add {files_str}")
        else:
            return await self._run_git_command("add -A")
    
    async def commit(
        self,
        message: str,
        files: list[str] | None = None,
    ) -> GitOperationResult:
        """Create a commit."""
        if files:
            await self.stage_files(files)
        
        # Escape quotes in message
        safe_message = message.replace('"', '\\"')
        return await self._run_git_command(f'commit -m "{safe_message}"')
    
    async def create_branch(
        self,
        branch_name: str,
        checkout: bool = True,
    ) -> GitOperationResult:
        """Create a new branch."""
        if checkout:
            return await self._run_git_command(f"checkout -b {branch_name}")
        else:
            return await self._run_git_command(f"branch {branch_name}")
    
    async def checkout_branch(self, branch_name: str) -> GitOperationResult:
        """Checkout an existing branch."""
        return await self._run_git_command(f"checkout {branch_name}")
    
    async def stash_changes(self, message: str | None = None) -> GitOperationResult:
        """Stash current changes."""
        if message:
            return await self._run_git_command(f'stash push -m "{message}"')
        else:
            return await self._run_git_command("stash")
    
    async def pop_stash(self) -> GitOperationResult:
        """Pop the most recent stash."""
        return await self._run_git_command("stash pop")
    
    async def get_log(
        self,
        count: int = 10,
        oneline: bool = True,
    ) -> list[dict[str, str]]:
        """Get recent commit log."""
        format_str = "%H|%s|%an|%ai" if not oneline else "%h|%s"
        result = await self._run_git_command(f'log -{count} --format="{format_str}"')
        
        if result.status != GitStatus.SUCCESS:
            return []
        
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if oneline and len(parts) >= 2:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                })
            elif len(parts) >= 4:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })
        
        return commits
    
    async def get_changed_files_since(self, commit: str) -> list[str]:
        """Get list of files changed since a commit."""
        result = await self._run_git_command(f"diff --name-only {commit}")
        if result.status == GitStatus.SUCCESS:
            return [f for f in result.stdout.strip().split("\n") if f]
        return []
    
    async def has_conflicts(self) -> bool:
        """Check if there are merge conflicts."""
        result = await self._run_git_command("diff --check")
        return result.status != GitStatus.SUCCESS
    
    async def _run_git_command(self, command: str) -> GitOperationResult:
        """Run a git command."""
        full_command = f"git {command}"
        
        try:
            process = await asyncio.create_subprocess_shell(
                full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
            )
            
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=60,
            )
            
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            
            if process.returncode == 0:
                return GitOperationResult(
                    status=GitStatus.SUCCESS,
                    message="Command succeeded",
                    stdout=stdout,
                    stderr=stderr,
                )
            else:
                # Check for specific error conditions
                if "not a git repository" in stderr.lower():
                    return GitOperationResult(
                        status=GitStatus.NO_REPO,
                        message="Not a git repository",
                        stdout=stdout,
                        stderr=stderr,
                    )
                elif "conflict" in stderr.lower():
                    return GitOperationResult(
                        status=GitStatus.CONFLICT,
                        message="Merge conflict detected",
                        stdout=stdout,
                        stderr=stderr,
                    )
                else:
                    return GitOperationResult(
                        status=GitStatus.FAILURE,
                        message=f"Command failed: {stderr or stdout}",
                        stdout=stdout,
                        stderr=stderr,
                    )
                    
        except asyncio.TimeoutError:
            return GitOperationResult(
                status=GitStatus.FAILURE,
                message="Command timed out",
            )
        except Exception as e:
            return GitOperationResult(
                status=GitStatus.FAILURE,
                message=str(e),
            )
    
    def _parse_status_output(self, output: str) -> list[dict[str, str]]:
        """Parse git status --porcelain output."""
        files = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            status = line[:2]
            file_path = line[3:]
            files.append({
                "status": status.strip(),
                "path": file_path,
            })
        return files
    
    def _parse_diff_stat(self, output: str) -> tuple[list[str], int, int]:
        """Parse git diff --stat output."""
        files = []
        insertions = 0
        deletions = 0
        
        lines = output.strip().split("\n")
        for line in lines:
            # Match file lines like: " file.py | 10 ++++----"
            file_match = re.match(r"\s*(.+?)\s*\|\s*\d+", line)
            if file_match:
                files.append(file_match.group(1).strip())
            
            # Match summary line like: " 3 files changed, 10 insertions(+), 5 deletions(-)"
            summary_match = re.search(
                r"(\d+) insertion.*?(\d+) deletion",
                line,
            )
            if summary_match:
                insertions = int(summary_match.group(1))
                deletions = int(summary_match.group(2))
            else:
                # Try just insertions
                ins_match = re.search(r"(\d+) insertion", line)
                if ins_match:
                    insertions = int(ins_match.group(1))
                # Try just deletions
                del_match = re.search(r"(\d+) deletion", line)
                if del_match:
                    deletions = int(del_match.group(1))
        
        return files, insertions, deletions
    
    def get_checkpoint(self, checkpoint_id: str) -> GitCheckpoint | None:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self) -> list[GitCheckpoint]:
        """List all checkpoints."""
        return list(self._checkpoints.values())


def create_git_manager(
    working_directory: str,
    verbose: bool = True,
) -> GitManager:
    """Create a git manager with default settings."""
    return GitManager(
        working_directory=working_directory,
        verbose=verbose,
    )
