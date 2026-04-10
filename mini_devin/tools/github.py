"""
GitHub Tool for Mini-Devin

This tool provides access to GitHub integration features like branch creation,
commits, and pull requests.
"""

from typing import Any, List, Optional
from pydantic import Field
import asyncio
from enum import Enum

from ..core.tool_interface import BaseTool
from ..schemas.tools import BaseToolInput, BaseToolOutput, ToolStatus
from ..integrations.github import GitHubIntegration, create_automated_pr_workflow

class GitHubAction(str, Enum):
    """Available GitHub actions."""
    CREATE_BRANCH = "create_branch"
    COMMIT = "commit"
    CREATE_PR = "create_pr"
    AUTOMATED_WORKFLOW = "automated_workflow"

class GitHubToolInput(BaseToolInput):
    """Input for GitHub tool."""
    action: GitHubAction = Field(description="The GitHub action to perform")
    
    # Common/Branch parameters
    branch_name: Optional[str] = Field(default=None, description="Name of the branch")
    base_branch: str = Field(default="main", description="Base branch name")
    
    # Commit parameters
    commit_message: Optional[str] = Field(default=None, description="Commit message")
    files: Optional[List[str]] = Field(default=None, description="List of files to commit. If empty, commits all changes.")
    
    # PR parameters
    pr_title: Optional[str] = Field(default=None, description="Pull Request title")
    pr_description: Optional[str] = Field(default=None, description="Pull Request description")
    
    # Automated Workflow parameters
    task_description: Optional[str] = Field(default=None, description="Task description for automated workflow")
    
    repo_path: str = Field(default=".", description="Local repository path")

class GitHubToolOutput(BaseToolOutput):
    """Output for GitHub tool."""
    success: bool = Field(description="Whether the action succeeded")
    message: str = Field(description="Result message")
    pr_url: Optional[str] = Field(default=None, description="Pull request URL, if applicable")

class GitHubTool(BaseTool[GitHubToolInput, GitHubToolOutput]):
    """Tool for GitHub operations."""
    
    name = "github"
    description = """Perform GitHub operations:
- create_branch: Create and push a new feature branch
- commit: Commit and push changes
- create_pr: Create a pull request
- automated_workflow: Complete automated PR flow (branch -> commit -> PR)"""
    
    input_schema = GitHubToolInput
    output_schema = GitHubToolOutput
    
    def __init__(self, pb_token: Optional[str] = None, repo_name: Optional[str] = None):
        super().__init__()
        self.github_integration = GitHubIntegration(pb_token)
        self.repo_name = repo_name
        self.initialized = False
        
    async def _init_if_needed(self, repo_path: str):
        if not self.initialized:
            # We need repo_name, assume from path or explicit parameter. For simplicity if not provided:
            # Let's extract from .git config or require it. For autonomous agents, often we pass repo_name.
            import subprocess
            if not self.repo_name:
                try:
                    result = subprocess.run(["git", "config", "--get", "remote.origin.url"], 
                                           cwd=repo_path, capture_output=True, text=True)
                    url = result.stdout.strip()
                    # Parse github.com:user/repo.git or https://github.com/user/repo.git
                    if "github.com" in url:
                        parts = url.split("github.com")[-1].replace(":", "/").strip("/")
                        if parts.endswith(".git"):
                            parts = parts[:-4]
                        self.repo_name = parts
                except:
                    pass
            
            if self.repo_name:
                self.initialized = await self.github_integration.initialize(repo_path, self.repo_name)
    
    async def _execute(self, input_data: GitHubToolInput) -> GitHubToolOutput:
        try:
            await self._init_if_needed(input_data.repo_path)
            
            if not self.initialized:
                return GitHubToolOutput(
                    status=ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=False,
                    message="Failed to initialize GitHub integration (maybe no token or not a git repo)."
                )

            if input_data.action == GitHubAction.CREATE_BRANCH:
                if not input_data.branch_name:
                    raise ValueError("branch_name is required")
                success = await self.github_integration.create_feature_branch(
                    input_data.branch_name, input_data.base_branch
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
                success = await self.github_integration.commit_changes(
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
                pr = await self.github_integration.create_pull_request(
                    title=input_data.pr_title,
                    description=input_data.pr_description,
                    head_branch=input_data.branch_name,
                    base_branch=input_data.base_branch
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
                    github_integration=self.github_integration,
                    task_description=input_data.task_description,
                    changes_made=input_data.files or ["Changed files"],
                    branch_name=input_data.branch_name
                )
                return GitHubToolOutput(
                    status=ToolStatus.SUCCESS if pr else ToolStatus.FAILURE,
                    execution_time_ms=0,
                    success=bool(pr),
                    message=f"Automated workflow complete, PR #{pr.number}" if pr else "Automated workflow failed",
                    pr_url=pr.html_url if pr else None
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
