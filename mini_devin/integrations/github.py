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
    from github.Issue import Issue
    from github.Repository import Repository
    from github.PullRequest import PullRequest
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    Github = None
    Issue = None

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


def _iter_paginated_items(items: Any, limit: int) -> list[Any]:
    """Convert a PyGithub PaginatedList into a bounded plain list."""
    out: list[Any] = []
    for idx, item in enumerate(items):
        if idx >= limit:
            break
        out.append(item)
    return out


def _serialize_label(label: Any) -> dict[str, Any]:
    return {
        "name": getattr(label, "name", ""),
        "color": getattr(label, "color", ""),
        "description": getattr(label, "description", None),
    }


def _serialize_issue_comment(comment: Any) -> dict[str, Any]:
    user = getattr(comment, "user", None)
    return {
        "id": getattr(comment, "id", None),
        "author": getattr(user, "login", None),
        "body": getattr(comment, "body", ""),
        "created_at": (
            comment.created_at.isoformat() if getattr(comment, "created_at", None) else None
        ),
        "updated_at": (
            comment.updated_at.isoformat() if getattr(comment, "updated_at", None) else None
        ),
        "html_url": getattr(comment, "html_url", None),
    }


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

    async def get_repository_context(self, issue_limit: int = 20) -> Dict[str, Any]:
        """Get repository metadata plus labels for issue/PR planning."""
        try:
            info = await self.get_repository_info()
            if not self.repo:
                return info
            labels = [_serialize_label(label) for label in _iter_paginated_items(self.repo.get_labels(), 200)]
            open_pulls = self.repo.get_pulls(state="open").totalCount
            open_issues = self.repo.get_issues(state="open").totalCount
            return {
                **info,
                "labels": labels,
                "open_pull_requests": open_pulls,
                "open_issues_total": open_issues,
                "suggested_issue_fetch_limit": issue_limit,
            }
        except Exception as e:
            logger.error(f"Failed to get repository context: {e}")
            return {}

    def _serialize_issue(self, issue: Issue, include_comments: bool = True) -> Dict[str, Any]:
        comments: list[dict[str, Any]] = []
        if include_comments:
            comments = [
                _serialize_issue_comment(comment)
                for comment in _iter_paginated_items(issue.get_comments(), 50)
            ]
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "body": issue.body or "",
            "html_url": issue.html_url,
            "author": issue.user.login if issue.user else None,
            "labels": [_serialize_label(label) for label in issue.labels],
            "assignees": [assignee.login for assignee in issue.assignees],
            "comments_count": issue.comments,
            "comments": comments,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
        }

    async def get_issue(self, issue_number: int, include_comments: bool = True) -> Dict[str, Any]:
        """Get a single issue with labels and comments."""
        try:
            if not self.repo:
                return {}
            issue = self.repo.get_issue(number=issue_number)
            if issue.pull_request is not None:
                return {}
            return self._serialize_issue(issue, include_comments=include_comments)
        except GithubException as e:
            logger.error(f"Failed to get issue #{issue_number}: {e}")
            return {}

    async def list_issues(
        self,
        state: str = "open",
        limit: int = 20,
        include_comments: bool = False,
    ) -> List[Dict[str, Any]]:
        """List repository issues for planning/problem selection."""
        try:
            if not self.repo:
                return []
            issues = self.repo.get_issues(state=state)
            items: list[dict[str, Any]] = []
            for issue in issues:
                if issue.pull_request is not None:
                    continue
                items.append(self._serialize_issue(issue, include_comments=include_comments))
                if len(items) >= limit:
                    break
            return items
        except GithubException as e:
            logger.error(f"Failed to list issues: {e}")
            return []

    async def get_pull_request(
        self,
        pr_number: int,
        include_comments: bool = True,
    ) -> Dict[str, Any]:
        """Get a pull request with review and comment context."""
        try:
            if not self.repo:
                return {}
            pr = self.repo.get_pull(pr_number)
            issue_view = pr.as_issue()
            comments = []
            if include_comments:
                comments = [
                    _serialize_issue_comment(comment)
                    for comment in _iter_paginated_items(issue_view.get_comments(), 50)
                ]
            return {
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "body": pr.body or "",
                "html_url": pr.html_url,
                "author": pr.user.login if pr.user else None,
                "head_branch": pr.head.ref,
                "base_branch": pr.base.ref,
                "draft": bool(getattr(pr, "draft", False)),
                "mergeable": pr.mergeable,
                "mergeable_state": pr.mergeable_state,
                "labels": [_serialize_label(label) for label in issue_view.labels],
                "comments_count": issue_view.comments,
                "comments": comments,
                "review_status": self._get_review_status(pr),
                "created_at": pr.created_at.isoformat() if pr.created_at else None,
                "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
            }
        except GithubException as e:
            logger.error(f"Failed to get PR #{pr_number}: {e}")
            return {}

    async def list_pull_requests(self, state: str = "open", limit: int = 20) -> List[Dict[str, Any]]:
        """List pull requests with high-signal metadata."""
        try:
            if not self.repo:
                return []
            pulls = self.repo.get_pulls(state=state)
            items: list[dict[str, Any]] = []
            for pr in _iter_paginated_items(pulls, limit):
                items.append(
                    {
                        "number": pr.number,
                        "title": pr.title,
                        "state": pr.state,
                        "html_url": pr.html_url,
                        "author": pr.user.login if pr.user else None,
                        "head_branch": pr.head.ref,
                        "base_branch": pr.base.ref,
                        "draft": bool(getattr(pr, "draft", False)),
                        "created_at": pr.created_at.isoformat() if pr.created_at else None,
                        "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
                    }
                )
            return items
        except GithubException as e:
            logger.error(f"Failed to list pull requests: {e}")
            return []

# Example usage and helper functions
async def create_automated_pr_workflow(
    hosting: Any,
    task_description: str,
    changes_made: List[str],
    branch_name: Optional[str] = None,
    linked_issues: Optional[List[int]] = None,
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
    pr_lines = [
        "## Summary",
        f"- {task_description}",
        "",
        "## Changed Files",
        *[f"- `{change}`" for change in changes_made],
        "",
        "## Verification",
        "- Local verification completed before automated PR creation.",
        "",
        "## Notes",
        "- Generated automatically by Plodder AI Assistant.",
        "- Please review the changes before merging.",
    ]
    if linked_issues:
        pr_lines.extend(
            [
                "",
                "## Linked Issues",
                *[f"- Closes #{issue}" for issue in linked_issues],
            ]
        )
    pr_description = "\n".join(pr_lines)

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
