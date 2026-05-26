#!/usr/bin/env python3
"""
Seed Plodder project memory from many popular GitHub repositories.

This is retrieval/context training, not model weight training. The script discovers
popular repositories, creates one Plodder project per repository, and calls the
existing enhanced repo ingest endpoint. Progress is written to a state file so the
run can be resumed safely after rate limits, network errors, or server restarts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_LANGUAGES = (
    "TypeScript",
    "Python",
    "JavaScript",
    "Go",
    "Rust",
    "Java",
    "C++",
    "C#",
    "PHP",
    "Ruby",
)


@dataclass
class RepoSpec:
    full_name: str
    clone_url: str
    html_url: str = ""
    description: str = ""
    language: str = ""
    stars: int = 0
    license_spdx: str = ""


@dataclass
class TrainingState:
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "TrainingState":
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        return cls(
            completed=dict(raw.get("completed") or {}),
            failed=dict(raw.get("failed") or {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or "repo"


def project_id_for_repo(full_name: str) -> str:
    return f"gh-{_slug(full_name)}"


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> Any:
    data = None
    req_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        raw = res.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw else {}


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "plodder-top-repo-training",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _repo_from_github_item(item: dict[str, Any]) -> RepoSpec:
    license_info = item.get("license") or {}
    return RepoSpec(
        full_name=str(item.get("full_name") or ""),
        clone_url=str(item.get("clone_url") or ""),
        html_url=str(item.get("html_url") or ""),
        description=str(item.get("description") or ""),
        language=str(item.get("language") or ""),
        stars=int(item.get("stargazers_count") or 0),
        license_spdx=str(license_info.get("spdx_id") or ""),
    )


def fetch_top_github_repos(
    *,
    max_repos: int,
    languages: list[str],
    per_language: int,
    min_stars: int,
    pause_seconds: float,
) -> list[RepoSpec]:
    """Fetch popular repositories by language buckets to avoid GitHub's 1000-result cap."""
    repos: dict[str, RepoSpec] = {}
    headers = _github_headers()

    for language in languages:
        if len(repos) >= max_repos:
            break
        pages = max(1, min(10, (per_language + 99) // 100))
        for page in range(1, pages + 1):
            if len(repos) >= max_repos:
                break
            query = f"stars:>={min_stars} language:{language}"
            qs = urllib.parse.urlencode(
                {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                }
            )
            url = f"https://api.github.com/search/repositories?{qs}"
            data = _request_json("GET", url, headers=headers, timeout=60)
            items = data.get("items") or []
            if not items:
                break
            for item in items:
                spec = _repo_from_github_item(item)
                if spec.full_name and spec.clone_url:
                    repos.setdefault(spec.full_name, spec)
            time.sleep(pause_seconds)

    return sorted(repos.values(), key=lambda r: (-r.stars, r.full_name))[:max_repos]


def load_repos_file(path: Path) -> list[RepoSpec]:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []
    repos: list[RepoSpec] = []
    if raw.startswith("["):
        data = json.loads(raw)
        for item in data:
            if not isinstance(item, dict):
                continue
            full_name = str(item.get("full_name") or item.get("name") or "")
            url = str(item.get("clone_url") or item.get("url") or "")
            if full_name and url:
                repos.append(RepoSpec(full_name=full_name, clone_url=url, html_url=url))
        return repos

    for line in raw.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        url = line
        full = url.rstrip("/").replace(".git", "").split("github.com/")[-1]
        clone_url = url if url.endswith(".git") else f"{url.rstrip('/')}.git"
        repos.append(RepoSpec(full_name=full, clone_url=clone_url, html_url=url))
    return repos


def create_project(api_base: str, repo: RepoSpec) -> str:
    project_id = project_id_for_repo(repo.full_name)
    payload = {
        "project_id": project_id,
        "name": repo.full_name,
        "description": repo.description or f"GitHub repository knowledge base for {repo.full_name}.",
        "repo_url": repo.html_url or repo.clone_url,
        "tech_stack": [x for x in [repo.language] if x],
    }
    _request_json("POST", f"{api_base}/api/projects", payload=payload, timeout=60)
    return project_id


def ingest_repo(api_base: str, project_id: str, repo: RepoSpec, *, dry_run: bool) -> dict[str, Any]:
    payload = {
        "repo_url": repo.clone_url,
        "dry_run": dry_run,
        "keep_clone": False,
        "skip_if_duplicate": True,
    }
    return dict(
        _request_json(
            "POST",
            f"{api_base}/api/projects/{urllib.parse.quote(project_id)}/ingest-repo",
            payload=payload,
            timeout=900,
        )
    )


def run_training(args: argparse.Namespace) -> int:
    api_base = args.api_base.rstrip("/")
    state_path = Path(args.state)
    state = TrainingState.load(state_path)

    if args.repos_file:
        repos = load_repos_file(Path(args.repos_file))
    else:
        languages = [x.strip() for x in args.languages.split(",") if x.strip()]
        repos = fetch_top_github_repos(
            max_repos=args.max_repos,
            languages=languages,
            per_language=args.per_language,
            min_stars=args.min_stars,
            pause_seconds=args.github_pause,
        )

    total = min(args.max_repos, len(repos))
    print(f"[train] candidate repos: {len(repos)}; target this run: {total}")
    if args.dry_run:
        print("[train] dry-run mode: Plodder will scan previews but not save ingest entries.")

    processed = 0
    for repo in repos[:total]:
        if repo.full_name in state.completed and not args.force:
            continue
        processed += 1
        print(f"[train] {processed}/{total} {repo.full_name} ({repo.stars} stars)")
        try:
            project_id = create_project(api_base, repo)
            result = ingest_repo(api_base, project_id, repo, dry_run=args.dry_run)
            state.completed[repo.full_name] = {
                "project_id": project_id,
                "clone_url": repo.clone_url,
                "stars": repo.stars,
                "paths_indexed": result.get("paths_indexed"),
                "warnings": result.get("warnings"),
                "skipped": result.get("skipped", False),
                "content_sha256": result.get("content_sha256"),
                "dry_run": args.dry_run,
                "updated_at": int(time.time()),
            }
            state.failed.pop(repo.full_name, None)
            state.save(state_path)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1200]
            state.failed[repo.full_name] = {"error": f"HTTP {exc.code}: {detail}", "updated_at": int(time.time())}
            state.save(state_path)
            print(f"[train] failed {repo.full_name}: HTTP {exc.code}", file=sys.stderr)
            if exc.code in {403, 429}:
                print("[train] hit rate limit; stop and resume later with the same command.", file=sys.stderr)
                return 2
        except Exception as exc:
            state.failed[repo.full_name] = {"error": str(exc)[:1200], "updated_at": int(time.time())}
            state.save(state_path)
            print(f"[train] failed {repo.full_name}: {exc}", file=sys.stderr)
        time.sleep(args.ingest_pause)

    print(f"[train] completed: {len(state.completed)}; failed: {len(state.failed)}")
    print(f"[train] state: {state_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Plodder project memory from popular GitHub repositories")
    parser.add_argument("--api-base", default=os.environ.get("PLODDER_API_BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--max-repos", type=int, default=20)
    parser.add_argument("--repos-file", help="Optional JSON/text manifest of repos instead of GitHub search")
    parser.add_argument("--state", default="data/top_repo_training_state.json")
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--per-language", type=int, default=250)
    parser.add_argument("--min-stars", type=int, default=5000)
    parser.add_argument("--github-pause", type=float, default=1.5)
    parser.add_argument("--ingest-pause", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Reprocess repos already marked completed")
    return parser


def main() -> None:
    raise SystemExit(run_training(build_parser().parse_args()))


if __name__ == "__main__":
    main()
