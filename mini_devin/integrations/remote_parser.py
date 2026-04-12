"""
Parse ``git remote`` URLs into host + repo path for GitHub / GitLab / Bitbucket.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional
from urllib.parse import urlparse

Platform = Literal["github", "gitlab", "bitbucket", "unknown"]


@dataclass(frozen=True)
class ParsedGitRemote:
    platform: Platform
    """``owner/repo`` (GitHub) or ``group/sub/repo`` (GitLab) without ``.git``."""

    path_with_namespace: str
    web_host: str
    """REST root for GitLab v4, e.g. ``https://gitlab.com/api/v4``. Empty if not GitLab."""

    api_v4_base: str


def _strip_git_suffix(path: str) -> str:
    p = path.strip().strip("/")
    if p.endswith(".git"):
        p = p[:-4]
    return p


def parse_git_remote_url(url: str) -> Optional[ParsedGitRemote]:
    u = (url or "").strip()
    if not u:
        return None

    host = ""
    path_part = ""

    if u.startswith("git@"):
        m = re.match(r"git@([^:]+):(.+)$", u)
        if not m:
            return None
        host, path_part = m.group(1).lower(), m.group(2)
    elif u.startswith(("http://", "https://")):
        parsed = urlparse(u)
        host = (parsed.hostname or "").lower()
        path_part = _strip_git_suffix(parsed.path or "")
    else:
        return None

    if not host or not path_part:
        return None

    path_part = _strip_git_suffix(path_part)

    if "bitbucket.org" in host or host.endswith("bitbucket.org"):
        return ParsedGitRemote(
            platform="bitbucket",
            path_with_namespace=path_part,
            web_host=host,
            api_v4_base="",
        )

    if host in ("github.com", "www.github.com") or host.endswith(".github.com"):
        path_part = path_part.lstrip("/")
        if path_part.count("/") < 1:
            return None
        return ParsedGitRemote(
            platform="github",
            path_with_namespace=path_part,
            web_host=host,
            api_v4_base="",
        )

    env_api = (os.getenv("GITLAB_API_URL") or "").strip().rstrip("/")
    env_gl_host = (os.getenv("GITLAB_HOST") or "").strip().lower()

    def _default_gitlab_api(h: str) -> str:
        return f"https://{h}/api/v4"

    is_gitlab_com = host == "gitlab.com" or host.endswith(".gitlab.com") or host == "gitlab.cn"
    if env_api:
        api_h = (urlparse(env_api).hostname or "").lower()
        if api_h and host == api_h:
            return ParsedGitRemote(
                platform="gitlab",
                path_with_namespace=path_part.lstrip("/"),
                web_host=host,
                api_v4_base=env_api,
            )
    if env_gl_host and host == env_gl_host:
        api_base = env_api if env_api else _default_gitlab_api(host)
        return ParsedGitRemote(
            platform="gitlab",
            path_with_namespace=path_part.lstrip("/"),
            web_host=host,
            api_v4_base=api_base,
        )
    if is_gitlab_com:
        api_base = env_api if env_api else _default_gitlab_api(host)
        return ParsedGitRemote(
            platform="gitlab",
            path_with_namespace=path_part.lstrip("/"),
            web_host=host,
            api_v4_base=api_base,
        )

    return ParsedGitRemote(
        platform="unknown",
        path_with_namespace=path_part.lstrip("/"),
        web_host=host,
        api_v4_base="",
    )
