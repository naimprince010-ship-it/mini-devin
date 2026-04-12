"""
GitLab merge request workflow alongside local GitPython operations.

Uses GitLab REST v4 (``PRIVATE-TOKEN``). Set ``GITLAB_TOKEN`` (or pass token explicitly).
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

try:
    from git import InvalidGitRepositoryError, Repo

    _GIT_OK = True
except Exception:
    Repo = None
    InvalidGitRepositoryError = Exception
    _GIT_OK = False

logger = logging.getLogger(__name__)


class GitLabProjectIntegration:
    """Branch / commit / MR for a GitLab project (local git + REST)."""

    def __init__(
        self,
        token: Optional[str],
        path_with_namespace: str,
        api_v4_base: str,
    ):
        self.token = (token or os.getenv("GITLAB_TOKEN") or "").strip()
        self.path_with_namespace = path_with_namespace.strip().strip("/")
        self.api_v4_base = api_v4_base.rstrip("/")
        self.local_repo: Any = None
        self._default_branch: str = "main"
        self._project_id_enc: str = quote(self.path_with_namespace, safe="")

    def _headers(self) -> dict[str, str]:
        return {"PRIVATE-TOKEN": self.token}

    def _get(self, path: str) -> Any:
        url = f"{self.api_v4_base}{path}"
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, headers=self._headers())
            r.raise_for_status()
            return r.json()

    def _post(self, path: str, data: dict) -> Any:
        url = f"{self.api_v4_base}{path}"
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=self._headers(), json=data)
            if r.status_code >= 400:
                logger.error("GitLab POST %s: %s", path, r.text[:500])
            r.raise_for_status()
            return r.json()

    def _put(self, path: str, data: Optional[dict] = None) -> Any:
        url = f"{self.api_v4_base}{path}"
        with httpx.Client(timeout=30.0) as client:
            r = client.put(url, headers=self._headers(), json=data or {})
            if r.status_code >= 400:
                logger.error("GitLab PUT %s: %s", path, r.text[:500])
            r.raise_for_status()
            if r.content:
                return r.json()
            return {}

    async def initialize(self, repo_path: str) -> bool:
        try:
            if not self.token:
                raise ValueError("GITLAB_TOKEN (or token) required for GitLab remote")
            if not _GIT_OK:
                logger.warning("GitPython not available")
                return False
            self.local_repo = Repo(repo_path)
            proj = self._get(f"/projects/{self._project_id_enc}")
            self._default_branch = (proj.get("default_branch") or "main").strip() or "main"
            logger.info("GitLab project %s default_branch=%s", self.path_with_namespace, self._default_branch)
            return True
        except Exception as e:
            logger.error("GitLab initialize failed: %s", e)
            return False

    async def create_feature_branch(self, branch_name: str, base_branch: str | None = None) -> bool:
        stashed = False
        try:
            base = (base_branch or "").strip() if base_branch else ""
            if not base or base.lower() == "default":
                base = self._default_branch
            if self.local_repo.is_dirty(untracked_files=True):
                self.local_repo.git.stash("push", "-u", "-m", "plodder: auto-stash before create_feature_branch")
                stashed = True
            self.local_repo.git.checkout(base)
            self.local_repo.remotes.origin.pull(base)
            self.local_repo.git.checkout("-b", branch_name)
            self.local_repo.git.push("-u", "origin", branch_name)
            if stashed:
                try:
                    self.local_repo.git.stash("pop")
                except Exception as pop_e:
                    logger.warning("Stash pop failed after branch create: %s", pop_e)
            return True
        except Exception as e:
            logger.error("GitLab create_feature_branch: %s", e)
            return False

    async def commit_changes(self, message: str, files: Optional[List[str]] = None) -> bool:
        try:
            if files:
                for fp in files:
                    self.local_repo.index.add([fp])
            else:
                self.local_repo.git.add("-A")
            if not self.local_repo.is_dirty(untracked_files=True):
                return True
            self.local_repo.index.commit(message)
            branch = self.local_repo.active_branch.name
            self.local_repo.git.push("origin", branch)
            return True
        except Exception as e:
            logger.error("GitLab commit_changes: %s", e)
            return False

    def _list_open_mrs_source(self, source_branch: str) -> list[dict[str, Any]]:
        enc = quote(source_branch, safe="")
        path = f"/projects/{self._project_id_enc}/merge_requests?state=opened&source_branch={enc}"
        data = self._get(path)
        return data if isinstance(data, list) else []

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
    ) -> Any:
        del labels, reviewers, assignees  # v1: not mapped to GitLab IDs
        try:
            base = (base_branch or "").strip() if base_branch else ""
            if not base or base.lower() == "default":
                base = self._default_branch
            existing = self._list_open_mrs_source(head_branch)
            if existing:
                mr = existing[0]
                return SimpleNamespace(
                    number=int(mr.get("iid", 0)),
                    html_url=mr.get("web_url", ""),
                )
            body: dict[str, Any] = {
                "source_branch": head_branch,
                "target_branch": base,
                "title": title,
                "description": description or "",
            }
            if draft:
                body["draft"] = True
            mr = self._post(f"/projects/{self._project_id_enc}/merge_requests", body)
            return SimpleNamespace(
                number=int(mr.get("iid", 0)),
                html_url=mr.get("web_url", ""),
            )
        except Exception as e:
            logger.error("GitLab create MR: %s", e)
            return None

    async def merge_pull_request(self, pr_number: int, merge_method: str = "squash") -> bool:
        try:
            mm = (merge_method or "squash").lower()
            body: dict[str, Any] = {}
            if mm == "squash":
                body["squash"] = True
            elif mm == "rebase":
                body["squash"] = False
                body["should_remove_source_branch"] = False
            self._put(
                f"/projects/{self._project_id_enc}/merge_requests/{pr_number}/merge",
                body,
            )
            return True
        except Exception as e:
            logger.error("GitLab merge MR %s: %s", pr_number, e)
            return False

    async def get_pr_status(self, pr_number: int) -> Dict[str, Any]:
        try:
            mr = self._get(f"/projects/{self._project_id_enc}/merge_requests/{pr_number}")
            return {
                "pr_number": pr_number,
                "html_url": mr.get("web_url"),
                "state": mr.get("state"),
                "mergeable": mr.get("merge_status") == "can_be_merged",
                "mergeable_state": mr.get("merge_status"),
                "title": mr.get("title"),
                "description": mr.get("description"),
                "head_branch": mr.get("source_branch"),
                "base_branch": mr.get("target_branch"),
                "draft": mr.get("draft"),
                "review_status": {},
                "ci_status": None,
                "check_runs": [],
            }
        except Exception as e:
            logger.error("GitLab get MR status: %s", e)
            return {}
