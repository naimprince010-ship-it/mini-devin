"""
GitHub Integration for Plodder
Handles PR creation, branch management, and commits
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from github import Github, GithubException
    from github.Repository import Repository
    from github.PullRequest import PullRequest
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    Github = None

try:
    from git import Repo, InvalidGitRepositoryError
    GIT_AVAILABLE = True
except Exception:
    GIT_AVAILABLE = False
    Repo = None
    InvalidGitRepositoryError = Exception

logger = logging.getLogger(__name__)


def _sanitize_branch_fragment(text: str, max_len: int = 48) -> str:
    """Make a string safe for use inside a git branch name segment."""
    s = text.lower().strip().replace(" ", "-")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return (s[:max_len] or "task").strip("-") or "task"


class GitHubIntegration:
    """GitHub integration for autonomous development workflow"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not found in environment")
        
        self.github = None
        self.repo = None
        self.local_repo = None
        
    async def initialize(self, repo_path: str, repo_name: str) -> bool:
        """Initialize GitHub connection and repository"""
        try:
            if not self.github_token:
                raise ValueError("GitHub token required")
            if not GITHUB_AVAILABLE:
                logger.warning("PyGithub not available — GitHub integration disabled")
                return False
                
            self.github = Github(self.github_token)
            self.repo = self.github.get_repo(repo_name)
            
            # Initialize local git repository
            if GIT_AVAILABLE:
                try:
                    self.local_repo = Repo(repo_path)
                except InvalidGitRepositoryError:
                    logger.error(f"Not a git repository: {repo_path}")
                    return False
            else:
                logger.warning("GitPython not available — local git ops disabled")
                
            logger.info(f"Connected to GitHub repo: {repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GitHub integration: {e}")
            return False
    
    async def create_feature_branch(self, branch_name: str, base_branch: str | None = None) -> bool:
        """Create a new feature branch from the repo default branch (or ``base_branch`` when set)."""
        stashed = False
        try:
            base = (base_branch or "").strip() if base_branch else ""
            if not base or base.lower() == "default":
                base = self.repo.default_branch if self.repo else "main"
            if self.local_repo.is_dirty(untracked_files=True):
                self.local_repo.git.stash("push", "-u", "-m", "plodder: auto-stash before create_feature_branch")
                stashed = True
            # Checkout base branch and pull latest
            self.local_repo.git.checkout(base)
            self.local_repo.remotes.origin.pull(base)
            
            # Create and checkout new branch
            self.local_repo.git.checkout("-b", branch_name)

            # Push branch to remote
            self.local_repo.git.push("-u", "origin", branch_name)

            if stashed:
                try:
                    self.local_repo.git.stash("pop")
                except Exception as pop_e:
                    logger.warning(
                        "Stash pop failed after creating branch %s — apply stash manually: %s",
                        branch_name,
                        pop_e,
                    )

            logger.info(f"Created and pushed branch: {branch_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create branch {branch_name}: {e}")
            return False
    
    async def commit_changes(self, message: str, files: Optional[List[str]] = None) -> bool:
        """Commit changes with optional specific files"""
        try:
            if files:
                # Add specific files
                for file_path in files:
                    self.local_repo.index.add([file_path])
            else:
                # Add all changes
                self.local_repo.git.add('-A')
            
            # Check if there are changes to commit
            if not self.local_repo.is_dirty(untracked_files=True):
                logger.info("No changes to commit")
                return True
            
            # Commit changes
            self.local_repo.index.commit(message)
            
            # Push to current branch
            current_branch = self.local_repo.active_branch.name
            self.local_repo.git.push('origin', current_branch)
            
            logger.info(f"Committed and pushed: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to commit changes: {e}")
            return False
    
    async def create_pull_request(
        self,
        title: str,
        description: str,
        head_branch: str,
        base_branch: str | None = None,
        labels: Optional[List[str]] = None,
        reviewers: Optional[List[str]] = None,
        draft: bool = False,
        assignees: Optional[List[str]] = None,
    ) -> Optional[PullRequest]:
        """Create a pull request (``base_branch`` defaults to the GitHub repo default branch)."""
        try:
            base = (base_branch or "").strip() if base_branch else ""
            if not base or base.lower() == "default":
                base = self.repo.default_branch if self.repo else "main"
            # GitHub API expects ``owner:branch`` when filtering pulls by head (same-repo).
            head_label = head_branch if ":" in head_branch else f"{self.repo.owner.login}:{head_branch}"
            pulls = self.repo.get_pulls(state="open", head=head_label)
            if pulls.totalCount > 0:
                logger.info(f"PR already exists for branch {head_branch}")
                return pulls[0]

            # Create new PR (draft / assignees depend on PyGithub + token scopes)
            if draft:
                try:
                    pr = self.repo.create_pull(
                        title=title,
                        body=description,
                        head=head_branch,
                        base=base,
                        draft=True,
                    )
                except TypeError:
                    logger.warning("PyGithub create_pull(draft=...) not supported; opening non-draft PR")
                    pr = self.repo.create_pull(
                        title=title,
                        body=description,
                        head=head_branch,
                        base=base,
                    )
            else:
                pr = self.repo.create_pull(
                    title=title,
                    body=description,
                    head=head_branch,
                    base=base,
                )

            if labels:
                pr.add_to_labels(*labels)

            if reviewers:
                pr.create_review_request(reviewers)

            if assignees:
                try:
                    pr.add_to_assignees(*assignees)
                except GithubException as ae:
                    logger.warning("Could not set assignees: %s", ae)

            logger.info(f"Created PR #{pr.number}: {title}")
            return pr
            
        except GithubException as e:
            logger.error(f"Failed to create PR: {e}")
            return None
    
    async def merge_pull_request(self, pr_number: int, merge_method: str = "squash") -> bool:
        """Merge a pull request"""
        try:
            pr = self.repo.get_pull(pr_number)

            if pr.mergeable is None:
                logger.warning(f"PR #{pr_number}: mergeability not computed yet — retry shortly")
                return False
            if pr.mergeable is False:
                logger.warning(f"PR #{pr_number} is not mergeable (conflicts or checks)")
                return False

            pr.merge(merge_method=merge_method)
            logger.info(f"Merged PR #{pr_number}")
            return True
                
        except GithubException as e:
            logger.error(f"Failed to merge PR #{pr_number}: {e}")
            return False
    
    async def get_pr_status(self, pr_number: int) -> Dict[str, Any]:
        """Get PR status and check results"""
        try:
            pr = self.repo.get_pull(pr_number)

            head_commit_obj = self.repo.get_commit(pr.head.sha)

            status = {
                "pr_number": pr_number,
                "html_url": pr.html_url,
                "state": pr.state,
                "mergeable": pr.mergeable,
                "mergeable_state": pr.mergeable_state,
                "title": pr.title,
                "description": pr.body,
                "head_branch": pr.head.ref,
                "base_branch": pr.base.ref,
                "review_status": self._get_review_status(pr),
                "ci_status": None,
            }

            combined_status = head_commit_obj.get_combined_status()
            status["ci_status"] = {
                "state": combined_status.state,
                "total_count": combined_status.total_count,
                "statuses": [
                    {
                        "context": s.context,
                        "state": s.state,
                        "description": s.description,
                    }
                    for s in combined_status.statuses
                ],
            }

            check_runs: list[dict[str, Any]] = []
            try:
                crs = head_commit_obj.get_check_runs()
                for cr in crs[:40]:
                    check_runs.append(
                        {
                            "name": cr.name,
                            "status": cr.status,
                            "conclusion": cr.conclusion,
                            "html_url": cr.html_url,
                        }
                    )
            except Exception as cre:
                logger.debug("check runs unavailable: %s", cre)
            status["check_runs"] = check_runs

            return status
            
        except GithubException as e:
            logger.error(f"Failed to get PR status: {e}")
            return {}
    
    def _get_review_status(self, pr: PullRequest) -> Dict[str, Any]:
        """Get review status of PR"""
        reviews = pr.get_reviews()
        approved = 0
        changes_requested = 0
        
        for review in reviews:
            if review.state == "APPROVED":
                approved += 1
            elif review.state == "CHANGES_REQUESTED":
                changes_requested += 1
        
        return {
            "approved": approved,
            "changes_requested": changes_requested,
            "total_reviews": reviews.totalCount
        }
    
    async def add_comment_to_pr(self, pr_number: int, comment: str) -> bool:
        """Add comment to PR"""
        try:
            pr = self.repo.get_pull(pr_number)
            pr.create_issue_comment(comment)
            logger.info(f"Added comment to PR #{pr_number}")
            return True
            
        except GithubException as e:
            logger.error(f"Failed to add comment to PR #{pr_number}: {e}")
            return False
    
    async def get_repository_info(self) -> Dict[str, Any]:
        """Get repository information"""
        try:
            if not self.repo:
                return {}
            
            return {
                "name": self.repo.name,
                "full_name": self.repo.full_name,
                "description": self.repo.description,
                "language": self.repo.language,
                "stars": self.repo.stargazers_count,
                "forks": self.repo.forks_count,
                "open_issues": self.repo.open_issues_count,
                "default_branch": self.repo.default_branch,
                "is_private": self.repo.private,
                "created_at": self.repo.created_at.isoformat(),
                "updated_at": self.repo.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return {}

# Example usage and helper functions
async def create_automated_pr_workflow(
    hosting: Any,
    task_description: str,
    changes_made: List[str],
    branch_name: Optional[str] = None,
) -> Any:
    """Complete automated PR/MR creation workflow (GitHub or GitLab hosting backend)."""

    if not branch_name:
        slug = _sanitize_branch_fragment(task_description, max_len=40)
        branch_name = f"plodder/{slug}"

    success = await hosting.create_feature_branch(branch_name)
    if not success:
        return None

    commit_message = f"feat: {task_description}"
    success = await hosting.commit_changes(commit_message, changes_made)
    if not success:
        return None

    pr_title = f"Automated: {task_description}"
    pr_description = f"""
## Automated Pull Request

**Task:** {task_description}

**Changes Made:**
{chr(10).join(f"- {change}" for change in changes_made)}

**Generated by:** Plodder AI Assistant

---
*This PR was created automatically by Plodder. Please review the changes before merging.*
"""

    labels_kw: dict[str, Any] = {}
    if hasattr(hosting, "repo") and getattr(hosting, "repo", None) is not None:
        labels_kw["labels"] = ["automated", "plodder"]

    pr = await hosting.create_pull_request(
        title=pr_title,
        description=pr_description,
        head_branch=branch_name,
        **labels_kw,
    )

    return pr
