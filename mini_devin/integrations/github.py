"""
GitHub Integration for Mini-Devin
Handles PR creation, branch management, and commits
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from github import Github, GithubException
from github.Repository import Repository
from github.PullRequest import PullRequest
from git import Repo, InvalidGitRepositoryError
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
                
            self.github = Github(self.github_token)
            self.repo = self.github.get_repo(repo_name)
            
            # Initialize local git repository
            try:
                self.local_repo = Repo(repo_path)
            except InvalidGitRepositoryError:
                logger.error(f"Not a git repository: {repo_path}")
                return False
                
            logger.info(f"Connected to GitHub repo: {repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GitHub integration: {e}")
            return False
    
    async def create_feature_branch(self, branch_name: str, base_branch: str = "main") -> bool:
        """Create a new feature branch"""
        try:
            # Checkout base branch and pull latest
            self.local_repo.git.checkout(base_branch)
            self.local_repo.remotes.origin.pull(base_branch)
            
            # Create and checkout new branch
            self.local_repo.git.checkout('-b', branch_name)
            
            # Push branch to remote
            self.local_repo.git.push('-u', 'origin', branch_name)
            
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
        base_branch: str = "main",
        labels: Optional[List[str]] = None,
        reviewers: Optional[List[str]] = None
    ) -> Optional[PullRequest]:
        """Create a pull request"""
        try:
            # Check if PR already exists
            pulls = self.repo.get_pulls(state='open', head=head_branch)
            if pulls.totalCount > 0:
                logger.info(f"PR already exists for branch {head_branch}")
                return pulls[0]
            
            # Create new PR
            pr = self.repo.create_pull(
                title=title,
                body=description,
                head=head_branch,
                base=base_branch
            )
            
            # Add labels if provided
            if labels:
                pr.add_to_labels(*labels)
            
            # Request reviewers if provided
            if reviewers:
                pr.create_review_request(reviewers)
            
            logger.info(f"Created PR #{pr.number}: {title}")
            return pr
            
        except GithubException as e:
            logger.error(f"Failed to create PR: {e}")
            return None
    
    async def merge_pull_request(self, pr_number: int, merge_method: str = "squash") -> bool:
        """Merge a pull request"""
        try:
            pr = self.repo.get_pull(pr_number)
            
            if pr.mergeable:
                pr.merge(merge_method=merge_method)
                logger.info(f"Merged PR #{pr_number}")
                return True
            else:
                logger.warning(f"PR #{pr_number} is not mergeable")
                return False
                
        except GithubException as e:
            logger.error(f"Failed to merge PR #{pr_number}: {e}")
            return False
    
    async def get_pr_status(self, pr_number: int) -> Dict[str, Any]:
        """Get PR status and check results"""
        try:
            pr = self.repo.get_pull(pr_number)
            
            # Get commit status
            commits = pr.get_commits()
            latest_commit = commits[0] if commits.totalCount > 0 else None
            
            status = {
                "pr_number": pr_number,
                "state": pr.state,
                "mergeable": pr.mergeable,
                "mergeable_state": pr.mergeable_state,
                "title": pr.title,
                "description": pr.body,
                "head_branch": pr.head.ref,
                "base_branch": pr.base.ref,
                "review_status": self._get_review_status(pr),
                "ci_status": None
            }
            
            if latest_commit:
                combined_status = latest_commit.get_combined_status()
                status["ci_status"] = {
                    "state": combined_status.state,
                    "total_count": combined_status.total_count,
                    "statuses": [
                        {
                            "context": s.context,
                            "state": s.state,
                            "description": s.description
                        } for s in combined_status.statuses
                    ]
                }
            
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
    github_integration: GitHubIntegration,
    task_description: str,
    changes_made: List[str],
    branch_name: Optional[str] = None
) -> Optional[PullRequest]:
    """Complete automated PR creation workflow"""
    
    if not branch_name:
        # Generate branch name from task description
        branch_name = f"feature/{task_description.lower().replace(' ', '-').replace('.', '')[:50]}"
    
    # Create feature branch
    success = await github_integration.create_feature_branch(branch_name)
    if not success:
        return None
    
    # Commit changes
    commit_message = f"feat: {task_description}"
    success = await github_integration.commit_changes(commit_message, changes_made)
    if not success:
        return None
    
    # Create PR
    pr_title = f"Automated: {task_description}"
    pr_description = f"""
## Automated Pull Request

**Task:** {task_description}

**Changes Made:**
{chr(10).join(f"- {change}" for change in changes_made)}

**Generated by:** Mini-Devin AI Assistant

---
*This PR was created automatically by Mini-Devin. Please review the changes before merging.*
"""
    
    pr = await github_integration.create_pull_request(
        title=pr_title,
        description=pr_description,
        head_branch=branch_name,
        labels=["automated", "mini-devin"]
    )
    
    return pr
