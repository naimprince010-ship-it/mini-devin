"""
Lightweight Mini-Devin API for production deployment.

This is a minimal version that doesn't load heavy dependencies
to work within free tier memory constraints.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, List
import uuid
import json
from datetime import datetime, timezone

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import time
import os

# Load environment variables
load_dotenv()

from .websocket import ConnectionManager, WebSocketMessage, MessageType
from ..database.config import init_db
from ..sessions.db_manager import DatabaseSessionManager

# Simple in-memory rate limiter (no external dependency)
_rate_buckets: dict = {}

def _check_rate_limit(key: str, max_calls: int, window_seconds: int = 60) -> bool:
    """Returns True if allowed, False if rate limited."""
    now = time.time()
    bucket = _rate_buckets.get(key, {"count": 0, "reset_at": now + window_seconds})
    if now > bucket["reset_at"]:
        bucket = {"count": 0, "reset_at": now + window_seconds}
    if bucket["count"] >= max_calls:
        _rate_buckets[key] = bucket
        return False
    bucket["count"] += 1
    _rate_buckets[key] = bucket
    return True

# Database-backed session management
session_manager = DatabaseSessionManager()
connection_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    print(f"[API] Starting Mini-Devin API at {datetime.now(timezone.utc).isoformat()}")
    print(f"[API] Environment: PORT={os.getenv('PORT')}, DATABASE_URL={'Set' if os.getenv('DATABASE_URL') else 'Default SLite'}")
    
    try:
        print("[API] Initializing Database...")
        await init_db()
        print("[API] Database initialized successfully.")
        
        # Reset stale sessions (IDLE/RUNNING) from previous server runs
        try:
            from sqlalchemy import update
            from ..database.models import SessionModel, SessionStatus as DBSessionStatus
            from ..database.config import get_session_maker
            async with get_session_maker()() as db:
                result = await db.execute(
                    update(SessionModel)
                    .where(SessionModel.status.in_([DBSessionStatus.IDLE, DBSessionStatus.RUNNING]))
                    .values(status=DBSessionStatus.TERMINATED)
                )
                await db.commit()
                if result.rowcount > 0:
                    print(f"[API] Reset {result.rowcount} stale session(s) from previous run.")
        except Exception as cleanup_err:
            print(f"[API] Warning: Could not reset stale sessions: {cleanup_err}")
    except Exception as e:
        print(f"[API] Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        # We don't exit here to allow health checks to potentially still pass if DB isn't critical
    
    # Register self-heal callback so the monitor can trigger an agent fix
    try:
        from ..integrations.monitor import register_heal_callback

        async def _auto_heal(app_name: str, logs_excerpt: str, heal_session_id: str | None):
            """When a crash is detected, create an agent task to diagnose and fix it."""
            print(f"[monitor] Auto-heal triggered for {app_name}")
            task_description = (
                f"The production app '{app_name}' appears to have crashed. "
                f"Here are the relevant log lines:\n\n{logs_excerpt}\n\n"
                "Please diagnose the root cause, fix the code, and redeploy."
            )
            try:
                target_session = await session_manager.get_session(heal_session_id) if heal_session_id else None
                if target_session is None:
                    target_session = await session_manager.create_session()
                task = await session_manager.create_task(
                    session_id=target_session.session_id,
                    description=task_description,
                    connection_manager=connection_manager,
                )
                asyncio.create_task(session_manager.run_task(
                    session_id=target_session.session_id,
                    task_id=task.task_id,
                    connection_manager=connection_manager,
                ))
                print(f"[monitor] Heal task created: session={target_session.session_id} task={task.task_id}")
            except Exception as heal_err:
                print(f"[monitor] Failed to create heal task: {heal_err}")

        register_heal_callback(_auto_heal)
        print("[API] Self-heal callback registered.")
    except Exception as e:
        print(f"[API] Warning: Could not register self-heal callback: {e}")

    yield
    print(f"[API] Shutting down Mini-Devin API at {datetime.now(timezone.utc).isoformat()}")
    # Stop monitor on shutdown
    try:
        from ..integrations.monitor import stop_monitor
        stop_monitor()
    except Exception:
        pass


app = FastAPI(
    title="Mini-Devin API",
    version="1.0.0",
    description="Autonomous AI Software Engineer Agent API (Lightweight Mode)",
    lifespan=lifespan,
)


def create_app() -> FastAPI:
    """Factory function to create the app instance."""
    return app

# Configure CORS - allow all origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "Mini-Devin API",
        "version": "1.0.0",
        "status": "running",
        "mode": "lightweight",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    return {"status": "healthy", "mode": "lightweight"}


# ── Repo management endpoints ─────────────────────────────────────────────────
# Supports cloning public (and token-authenticated private) GitHub repos.
# No OAuth required — just paste the repo URL.

# In-memory repo registry (persists until server restart)
_repos: dict[str, dict] = {}

def _get_repos_root() -> str:
    _mini_devin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    repos_root = os.path.join(os.path.dirname(_mini_devin_root), "agent-workspace", "repos")
    os.makedirs(repos_root, exist_ok=True)
    return repos_root

def _parse_github_url(url: str) -> tuple[str, str]:
    """Extract owner and repo_name from a GitHub URL."""
    import re
    url = url.strip().rstrip("/").replace(".git", "")
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+)", url)
    if not match:
        raise ValueError(f"Invalid GitHub URL: {url}")
    return match.group(1), match.group(2)

@app.get("/api/repos")
async def list_repos():
    return {"repos": list(_repos.values()), "total": len(_repos)}

@app.post("/api/repos")
async def add_repo(request: Request):
    body = await request.json()
    repo_url = (body.get("repo_url") or "").strip()
    github_token = body.get("github_token") or ""
    branch = body.get("branch") or "main"

    if not repo_url:
        raise HTTPException(status_code=400, detail="repo_url is required")

    try:
        owner, repo_name = _parse_github_url(repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    repo_id = str(uuid.uuid4())[:8]
    clone_url = f"https://github.com/{owner}/{repo_name}.git"
    if github_token:
        clone_url = f"https://{github_token}@github.com/{owner}/{repo_name}.git"

    local_path = os.path.join(_get_repos_root(), repo_name)

    repo_info = {
        "repo_id": repo_id,
        "repo_url": f"https://github.com/{owner}/{repo_name}",
        "repo_name": repo_name,
        "owner": owner,
        "default_branch": branch,
        "local_path": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_synced": None,
        "status": "pending",
        "has_token": bool(github_token),
        "_clone_url": clone_url,
        "_local_target": local_path,
        "_token": github_token or os.getenv("GITHUB_TOKEN", ""),
    }
    _repos[repo_id] = repo_info

    # Auto-clone in background
    asyncio.create_task(_clone_repo_bg(repo_id, clone_url, local_path, branch))
    return {k: v for k, v in repo_info.items() if not k.startswith("_")}

async def _run_git(*args, cwd=None, timeout=60) -> tuple[int, str, str]:
    """Run a git command asynchronously without blocking the event loop."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return -1, "", "timeout"
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _clone_repo_bg(repo_id: str, clone_url: str, local_path: str, branch: str):
    """Clone a repo in the background and update status."""
    try:
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, ".git")):
            await _run_git("-C", local_path, "pull", timeout=60)
            if repo_id in _repos:
                _repos[repo_id]["status"] = "cloned"
                _repos[repo_id]["local_path"] = local_path
                _repos[repo_id]["last_synced"] = datetime.now(timezone.utc).isoformat()
            return

        # Try clone with specified branch first
        rc, _, stderr1 = await _run_git("clone", "--depth", "1", "-b", branch, clone_url, local_path, timeout=60)
        if rc == 0:
            if repo_id in _repos:
                _repos[repo_id]["status"] = "cloned"
                _repos[repo_id]["local_path"] = local_path
                _repos[repo_id]["last_synced"] = datetime.now(timezone.utc).isoformat()
            return

        # Remove partial clone and try without branch
        import shutil
        if os.path.exists(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
        rc2, _, stderr2 = await _run_git("clone", "--depth", "1", clone_url, local_path, timeout=60)
        if rc2 == 0 and repo_id in _repos:
            _repos[repo_id]["status"] = "cloned"
            _repos[repo_id]["local_path"] = local_path
            _repos[repo_id]["last_synced"] = datetime.now(timezone.utc).isoformat()
            return

        # Both failed — check if empty repo
        combined_err = (stderr1 + stderr2).lower()
        is_empty = any(k in combined_err for k in ("empty", "did not", "no branch", "remote head", "warning: you appear"))
        if is_empty or rc2 != 0:
            # Empty repo — init a fresh local workspace and point remote to GitHub
            os.makedirs(local_path, exist_ok=True)
            await _run_git("init", cwd=local_path)
            await _run_git("remote", "add", "origin", clone_url, cwd=local_path)
            await _run_git("config", "user.email", "agent@mini-devin.local", cwd=local_path)
            await _run_git("config", "user.name", "Mini-Devin Agent", cwd=local_path)
            if repo_id in _repos:
                _repos[repo_id]["status"] = "cloned"
                _repos[repo_id]["local_path"] = local_path
                _repos[repo_id]["last_synced"] = datetime.now(timezone.utc).isoformat()
                _repos[repo_id]["note"] = "Empty repo — local workspace initialized, remote linked"
        else:
            if repo_id in _repos:
                _repos[repo_id]["status"] = "clone_failed"
                _repos[repo_id]["error"] = stderr2[:200] or "Unknown error"
    except Exception as e:
        if repo_id in _repos:
            _repos[repo_id]["status"] = "clone_failed"
        print(f"[Repos] Clone failed for {repo_id}: {e}")

@app.post("/api/repos/{repo_id}/clone")
async def clone_repo(repo_id: str):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    asyncio.create_task(_clone_repo_bg(
        repo_id, repo.get("_clone_url", repo["repo_url"]),
        repo.get("_local_target", repo["local_path"] or ""),
        repo["default_branch"]
    ))
    return {"status": "cloning", "repo_id": repo_id}

@app.post("/api/repos/{repo_id}/pull")
async def pull_repo(repo_id: str):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    asyncio.create_task(_clone_repo_bg(
        repo_id, repo.get("_clone_url", repo["repo_url"]),
        repo.get("_local_target", repo["local_path"] or ""),
        repo["default_branch"]
    ))
    return {"status": "pulling", "repo_id": repo_id}

@app.delete("/api/repos/{repo_id}")
async def delete_repo(repo_id: str):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos.pop(repo_id)
    # Optionally remove local clone
    local_path = repo.get("local_path")
    if local_path and os.path.exists(local_path):
        import shutil
        shutil.rmtree(local_path, ignore_errors=True)
    return {"status": "deleted", "repo_id": repo_id}

@app.get("/api/github/oauth/status")
async def github_oauth_status():
    return {
        "connected": False,
        "github_configured": bool(os.getenv("GITHUB_CLIENT_ID")),
    }

@app.get("/api/github/oauth/start")
async def github_oauth_start():
    raise HTTPException(status_code=501, detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET.")

# ── GitHub API (token-based) ──────────────────────────────────────────────────
# All GitHub operations using a Personal Access Token stored per-repo.

def _gh_headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

def _get_repo_token(repo_id: str) -> str:
    repo = _repos.get(repo_id, {})
    token = repo.get("_token") or os.getenv("GITHUB_TOKEN", "")
    if not token:
        raise HTTPException(status_code=400, detail="GitHub token required. Add the repo with a token, or set GITHUB_TOKEN env var.")
    return token

async def _gh_get(url: str, token: str) -> dict:
    import urllib.request
    req = urllib.request.Request(url, headers=_gh_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")

async def _gh_post(url: str, token: str, data: dict) -> dict:
    import urllib.request
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={**_gh_headers(token), "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise HTTPException(status_code=e.code, detail=f"GitHub API: {body[:300]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")

@app.post("/api/repos/{repo_id}/token")
async def set_repo_token(repo_id: str, request: Request):
    """Set/update the GitHub token for a repo."""
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    body = await request.json()
    token = body.get("token", "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    _repos[repo_id]["_token"] = token
    _repos[repo_id]["has_token"] = True
    return {"status": "ok"}

# ── Branches ──────────────────────────────────────────────────────────────────
@app.get("/api/repos/{repo_id}/branches")
async def list_branches(repo_id: str):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    token = _get_repo_token(repo_id)
    owner, name = repo["owner"], repo["repo_name"]
    data = await _gh_get(f"https://api.github.com/repos/{owner}/{name}/branches", token)
    return {"branches": [b["name"] for b in data]}

@app.post("/api/repos/{repo_id}/branches")
async def create_branch(repo_id: str, request: Request):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    token = _get_repo_token(repo_id)
    body = await request.json()
    branch_name = body.get("branch_name", "").strip()
    from_branch = body.get("from_branch", repo["default_branch"])
    if not branch_name:
        raise HTTPException(status_code=400, detail="branch_name required")
    owner, name = repo["owner"], repo["repo_name"]
    # Get SHA of from_branch
    ref_data = await _gh_get(f"https://api.github.com/repos/{owner}/{name}/git/ref/heads/{from_branch}", token)
    sha = ref_data["object"]["sha"]
    result = await _gh_post(f"https://api.github.com/repos/{owner}/{name}/git/refs", token, {
        "ref": f"refs/heads/{branch_name}", "sha": sha
    })
    # Also create local branch
    local_path = repo.get("local_path")
    if local_path and os.path.exists(local_path):
        await _run_git("fetch", "origin", cwd=local_path)
        await _run_git("checkout", "-b", branch_name, cwd=local_path)
    return {"branch": branch_name, "sha": sha, "status": "created"}

# ── Pull Requests ──────────────────────────────────────────────────────────────
@app.get("/api/repos/{repo_id}/pulls")
async def list_pulls(repo_id: str, state: str = "open"):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    token = _get_repo_token(repo_id)
    owner, name = repo["owner"], repo["repo_name"]
    data = await _gh_get(f"https://api.github.com/repos/{owner}/{name}/pulls?state={state}&per_page=20", token)
    return {"pulls": [{"number": p["number"], "title": p["title"], "state": p["state"],
                        "url": p["html_url"], "author": p["user"]["login"],
                        "head": p["head"]["ref"], "base": p["base"]["ref"],
                        "created_at": p["created_at"]} for p in data]}

@app.post("/api/repos/{repo_id}/pulls")
async def create_pull_request(repo_id: str, request: Request):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    token = _get_repo_token(repo_id)
    body = await request.json()
    owner, name = repo["owner"], repo["repo_name"]
    result = await _gh_post(f"https://api.github.com/repos/{owner}/{name}/pulls", token, {
        "title": body.get("title", "Automated PR by Mini-Devin"),
        "body": body.get("body", "Created by Mini-Devin AI agent."),
        "head": body.get("head"),
        "base": body.get("base", repo["default_branch"]),
    })
    return {"number": result["number"], "url": result["html_url"], "title": result["title"], "state": result["state"]}

# ── Issues ────────────────────────────────────────────────────────────────────
@app.get("/api/repos/{repo_id}/issues")
async def list_issues(repo_id: str, state: str = "open"):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    token = _get_repo_token(repo_id)
    owner, name = repo["owner"], repo["repo_name"]
    data = await _gh_get(f"https://api.github.com/repos/{owner}/{name}/issues?state={state}&per_page=20", token)
    return {"issues": [{"number": i["number"], "title": i["title"], "state": i["state"],
                         "url": i["html_url"], "author": i["user"]["login"],
                         "created_at": i["created_at"],
                         "labels": [l["name"] for l in i.get("labels", [])]} for i in data if "pull_request" not in i]}

@app.post("/api/repos/{repo_id}/issues")
async def create_issue(repo_id: str, request: Request):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    token = _get_repo_token(repo_id)
    body = await request.json()
    owner, name = repo["owner"], repo["repo_name"]
    result = await _gh_post(f"https://api.github.com/repos/{owner}/{name}/issues", token, {
        "title": body.get("title", ""),
        "body": body.get("body", ""),
        "labels": body.get("labels", []),
    })
    return {"number": result["number"], "url": result["html_url"], "title": result["title"]}

# ── Create new GitHub Repo ────────────────────────────────────────────────────
@app.post("/api/github/create-repo")
async def create_github_repo(request: Request):
    body = await request.json()
    token = body.get("token") or os.getenv("GITHUB_TOKEN", "")
    if not token:
        raise HTTPException(status_code=400, detail="GitHub token required")
    result = await _gh_post("https://api.github.com/user/repos", token, {
        "name": body.get("name", ""),
        "description": body.get("description", "Created by Mini-Devin"),
        "private": body.get("private", False),
        "auto_init": body.get("auto_init", True),
    })
    return {"repo_url": result["html_url"], "clone_url": result["clone_url"],
            "name": result["name"], "full_name": result["full_name"]}

# ── Commits ───────────────────────────────────────────────────────────────────
@app.get("/api/repos/{repo_id}/commits")
async def list_commits(repo_id: str, branch: str = ""):
    if repo_id not in _repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    repo = _repos[repo_id]
    token = _get_repo_token(repo_id)
    owner, name = repo["owner"], repo["repo_name"]
    b = branch or repo["default_branch"]
    data = await _gh_get(f"https://api.github.com/repos/{owner}/{name}/commits?sha={b}&per_page=15", token)
    return {"commits": [{"sha": c["sha"][:7], "message": c["commit"]["message"].split("\n")[0],
                          "author": c["commit"]["author"]["name"],
                          "date": c["commit"]["author"]["date"]} for c in data]}
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/browse")
async def browse_directory(path: str = "."):
    """Browse server filesystem directories for the folder picker UI."""
    import pathlib
    try:
        target = pathlib.Path(path).resolve()
        if not target.exists() or not target.is_dir():
            raise HTTPException(status_code=404, detail="Directory not found")

        entries = []
        # Add parent navigation (go up one level)
        parent = str(target.parent) if target != target.parent else None

        for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith('.') and entry.name not in ('.env',):
                continue
            if entry.name in ('__pycache__', 'node_modules', '.git'):
                continue
            entries.append({
                "name": entry.name,
                "path": str(entry),
                "is_directory": entry.is_dir(),
            })

        return {
            "current": str(target),
            "parent": parent,
            "entries": entries,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
@app.get("/sessions")
async def list_sessions():
    sessions = await session_manager.list_sessions()
    return [
        {
            "session_id": s.session_id,
            "created_at": s.created_at.isoformat(),
            "status": s.status.value,
            "working_directory": s.working_directory,
            "current_task": s.current_task_id,
            "iteration": s.iteration,
            "total_tasks": s.total_tasks,
            "title": getattr(s, 'title', ''),
            "model": getattr(s, 'model', 'gpt-4o'),
        }
        for s in sessions
    ]

class CreateSessionRequest(BaseModel):
    working_directory: str = "."
    model: str = "gpt-4o"
    max_iterations: int = 50
    auto_git_commit: bool = False
    git_push: bool = False

@app.post("/api/sessions")
@app.post("/sessions")
async def create_session(raw_request: Request):
    # Simple rate limiting: 10 sessions/minute per IP
    client_ip = raw_request.client.host if raw_request.client else "unknown"
    if not _check_rate_limit(f"create_session:{client_ip}", max_calls=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait before creating another session.")

    # Workspace root: outside mini-devin source code
    _mini_devin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _workspaces_root = os.path.join(os.path.dirname(_mini_devin_root), "agent-workspace")

    # Generate session ID early so we can create a per-session directory
    new_session_id = str(uuid.uuid4())[:8]

    model = "gpt-4o"
    max_iterations = 50
    auto_git_commit = False
    git_push = False
    requested_dir = ""
    project_id = ""
    try:
        body = await raw_request.json()
        requested_dir = body.get("working_directory", "") or ""
        model = body.get("model", "gpt-4o") or "gpt-4o"
        max_iterations = int(body.get("max_iterations", 50) or 50)
        auto_git_commit = bool(body.get("auto_git_commit", False))
        git_push = bool(body.get("git_push", False))
        project_id = body.get("project_id", "") or ""
    except Exception:
        pass

    # Determine working directory
    if requested_dir in ("", ".", "./"):
        # Per-session isolated workspace — each user/session gets their own folder
        working_dir = os.path.join(_workspaces_root, new_session_id)
        os.makedirs(working_dir, exist_ok=True)
        # Initialize git in the fresh workspace so agent can commit
        import subprocess
        git_dir = os.path.join(working_dir, ".git")
        if not os.path.exists(git_dir):
            subprocess.run(["git", "init"], cwd=working_dir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "agent@mini-devin.local"], cwd=working_dir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Mini-Devin Agent"], cwd=working_dir, capture_output=True)
    else:
        # User provided a specific path — use it directly
        working_dir = requested_dir

    try:
        session = await session_manager.create_session(
            working_directory=working_dir,
            model=model,
            max_iterations=max_iterations,
            auto_git_commit=auto_git_commit,
            git_push=git_push,
            session_id=new_session_id,
        )
    except Exception as e:
        import traceback
        print(f"[API] create_session ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    # Store project_id on agent for memory auto-injection
    if project_id:
        agent = session_manager._agents.get(session.session_id)
        if agent:
            agent.session_id = session.session_id
            agent._project_id = project_id
            # Inject project context into agent's initial system context
            try:
                from ..integrations.project_memory import get_project_memory
                pm = get_project_memory()
                ctx = pm.get_context_for_task(project_id, "", max_tokens=600)
                if ctx:
                    agent._project_context_injection = ctx
                    print(f"[API] Injected project memory for project '{project_id}' into session {session.session_id}")
            except Exception as e:
                print(f"[API] Could not inject project memory: {e}")

    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "status": session.status.value,
        "working_directory": session.working_directory,
        "workspace_path": session.working_directory,
        "model": getattr(session, 'model', model),
        "auto_git_commit": auto_git_commit,
        "git_push": git_push,
        "iteration": session.iteration,
        "total_tasks": session.total_tasks,
        "project_id": project_id or None,
    }

@app.get("/api/sessions/{session_id}")
@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    s = await session_manager.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": s.session_id,
        "created_at": s.created_at.isoformat(),
        "status": s.status.value,
        "working_directory": s.working_directory,
        "current_task": s.current_task_id,
        "iteration": s.iteration,
        "total_tasks": s.total_tasks,
        "title": getattr(s, 'title', ''),
    }

@app.delete("/api/sessions/{session_id}")
@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}

@app.post("/api/sessions/{session_id}/answer")
@app.post("/sessions/{session_id}/answer")
async def answer_clarification(session_id: str, request: Request):
    """Provide a user answer to a clarification question from the agent."""
    body = await request.json()
    answer = body.get("answer", "")
    if not answer:
        raise HTTPException(status_code=400, detail="Missing 'answer' field")
    ok = await session_manager.answer_clarification(session_id, answer)
    if not ok:
        raise HTTPException(status_code=404, detail="No pending clarification for this session")
    return {"status": "answered", "session_id": session_id}


@app.post("/api/sessions/{session_id}/sandbox/start")
@app.post("/sessions/{session_id}/sandbox/start")
async def start_sandbox(session_id: str):
    """Start a Docker sandbox for the given session."""
    result = await session_manager.start_sandbox(session_id)
    if not result.get("started"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to start sandbox"))
    # Broadcast sandbox_started event over WebSocket
    await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
        type=MessageType.STATUS,
        data={"event": "sandbox_started", "container_id": result.get("container_id"), "status": result.get("status")},
    ))
    return result


@app.post("/api/sessions/{session_id}/sandbox/stop")
@app.post("/sessions/{session_id}/sandbox/stop")
async def stop_sandbox(session_id: str):
    """Stop the Docker sandbox for the given session."""
    result = await session_manager.stop_sandbox(session_id)
    if not result.get("stopped"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to stop sandbox"))
    # Broadcast sandbox_stopped event over WebSocket
    await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
        type=MessageType.STATUS,
        data={"event": "sandbox_stopped"},
    ))
    return result



@app.post("/api/sandbox/build")
async def build_sandbox_image_endpoint():
    """Trigger a build of the mini-devin-sandbox Docker image (runs in background)."""
    async def _do_build():
        try:
            from ..sandbox.docker_sandbox import build_sandbox_from_repo
            ok = await build_sandbox_from_repo()
            print(f"[API] Sandbox image build {'succeeded' if ok else 'failed'}")
        except Exception as e:
            print(f"[API] Sandbox image build error: {e}")

    asyncio.create_task(_do_build())
    return {
        "status": "building",
        "image": "mini-devin-sandbox:latest",
        "message": "Build started in background. Check server logs for progress.",
    }


@app.get("/api/sessions/{session_id}/ls")
@app.get("/sessions/{session_id}/ls")
async def list_workspace_files(session_id: str, directory: str = "."):
    """List directory contents for a session."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    base_dir = session.working_directory or "."
    target_dir = os.path.join(base_dir, directory)
    
    try:
        abs_target = os.path.abspath(target_dir)
        if not os.path.exists(abs_target):
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
        
        entries = []
        for entry in os.scandir(abs_target):
            if entry.name.startswith("."):
                continue
                
            is_dir = entry.is_dir()
            stat = entry.stat()
            entries.append({
                "name": entry.name,
                "path": os.path.join(directory, entry.name),
                "is_directory": is_dir,
                "size": stat.st_size if not is_dir else 0,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
            
        entries.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
        return entries
        
    except Exception as e:
        print(f"Error listing directory {target_dir}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/file")
@app.get("/sessions/{session_id}/file")
async def read_file_content(session_id: str, path: str):
    """Read a file from the session's workspace."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    base_dir = os.path.abspath(session.working_directory or ".")
    target = os.path.abspath(os.path.join(base_dir, path))
    if not target.startswith(base_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(target) or os.path.isdir(target):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(target, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return {"path": path, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/sessions/{session_id}/file")
@app.put("/sessions/{session_id}/file")
async def write_file_content(session_id: str, req: Request):
    """Write content to a file in the session's workspace."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    body = await req.json()
    path = body.get("path", "")
    content = body.get("content", "")
    
    base_dir = os.path.abspath(session.working_directory or ".")
    target = os.path.abspath(os.path.join(base_dir, path))
    if not target.startswith(base_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"path": path, "saved": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LspDiagnosticsRequest(BaseModel):
    path: str
    content: Optional[str] = None


class LspHoverRequest(BaseModel):
    path: str
    line: int  # 1-based (Monaco)
    column: int = 1
    content: Optional[str] = None


@app.post("/api/sessions/{session_id}/lsp/diagnostics")
@app.post("/sessions/{session_id}/lsp/diagnostics")
async def session_lsp_diagnostics(session_id: str, body: LspDiagnosticsRequest):
    """
    Pull diagnostics for a workspace file (Python: Pyright/syntax; TS/JS: tsc when npx available).
    Optional ``content`` uses an in-memory snapshot without writing the workspace file.
    """
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    base_dir = os.path.abspath(session.working_directory or ".")
    if not _check_rate_limit(f"lsp:diag:{session_id}", max_calls=60, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many diagnostics requests")
    try:
        from ..lsp.diagnostics import collect_diagnostics

        items, source = collect_diagnostics(base_dir, body.path, body.content)
        return {"path": body.path, "diagnostics": items, "source": source}
    except Exception as e:
        print(f"[LSP] diagnostics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/lsp/hover")
@app.post("/sessions/{session_id}/lsp/hover")
async def session_lsp_hover(session_id: str, body: LspHoverRequest):
    """Symbol-aware hover (Tree-sitter) for Python / TS / JS."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    base_dir = os.path.abspath(session.working_directory or ".")
    if not _check_rate_limit(f"lsp:hover:{session_id}", max_calls=120, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many hover requests")
    try:
        from ..lsp.hover import collect_hover

        text = collect_hover(base_dir, body.path, body.line, body.column, body.content)
        return {"path": body.path, "contents": [{"value": text}] if text else []}
    except Exception as e:
        print(f"[LSP] hover error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lsp/capabilities")
@app.get("/lsp/capabilities")
async def lsp_capabilities():
    """Advertise LSP-style features for the IDE."""
    return {
        "diagnostics": True,
        "hover": True,
        "completion": False,
        "note": "Diagnostics use Pyright/basedpyright when installed; TS/JS uses npx tsc.",
    }


@app.post("/api/sessions/{session_id}/stop")
@app.post("/sessions/{session_id}/stop")
async def stop_session_app(session_id: str):
    """Stop the running task in a session."""
    success = await session_manager.stop_session(session_id)
    return {"stopped": True, "session_id": session_id}


@app.get("/api/sessions/{session_id}/history")
@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get chat conversation history for a session (agent messages)."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    agent = session_manager._agents.get(session_id)
    if not agent:
        return {"messages": [], "session_id": session_id}
    # Return LLM conversation history, excluding system prompt
    messages = []
    for msg in agent.llm.conversation:
        if msg.role == "system":
            continue
        entry = {
            "role": msg.role,
            "content": msg.content or "",
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls_out = []
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    name = tc.get("function", {}).get("name", "") or tc.get("name", "")
                    args = tc.get("function", {}).get("arguments", {}) or tc.get("args", {})
                else:
                    # ToolCall object
                    fn = getattr(tc, "function", None) or tc
                    name = getattr(fn, "name", "") or getattr(tc, "name", "")
                    args = getattr(fn, "arguments", {}) or getattr(tc, "args", {})
                tool_calls_out.append({"name": name, "args": args})
            entry["tool_calls"] = tool_calls_out
        messages.append(entry)
    return {"messages": messages, "session_id": session_id, "total": len(messages)}

@app.get("/api/sessions/{session_id}/tasks")
@app.get("/sessions/{session_id}/tasks")
async def list_tasks(session_id: str):
    tasks = await session_manager.list_tasks(session_id)
    return [
        {
            "task_id": t.task_id,
            "session_id": session_id,
            "description": t.description,
            "status": t.status.value,
            "iteration": t.iteration,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "started_at": t.started_at.isoformat() if t.started_at else None,
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            "error_message": t.error_message,
            "summary": (t.result.summary if getattr(t, "result", None) else "") or "",
        }
        for t in tasks
    ]

@app.get("/api/sessions/{session_id}/tasks/{task_id}")
@app.get("/sessions/{session_id}/tasks/{task_id}")
async def get_task(session_id: str, task_id: str):
    t = await session_manager.get_task(session_id, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": t.task_id,
        "session_id": session_id,
        "description": t.goal.description,
        "status": t.status.value,
        "iteration": t.iteration,
        "last_error": t.last_error,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
    }

@app.get("/api/sessions/{session_id}/tasks/{task_id}/result")
@app.get("/sessions/{session_id}/tasks/{task_id}/result")
async def get_task_result(session_id: str, task_id: str):
    result = await session_manager.get_task_result(session_id, task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return {
        "success": result.success,
        "summary": result.summary,
        "files_modified": result.files_modified,
        "commands_executed": result.commands_executed,
        "total_tokens": result.total_tokens,
        "duration_ms": result.duration_ms if hasattr(result, "duration_ms") else 0,
    }

@app.get("/api/sessions/{session_id}/tasks/{task_id}/artifacts")
@app.get("/sessions/{session_id}/tasks/{task_id}/artifacts")
async def list_artifacts(session_id: str, task_id: str):
    artifacts = await session_manager.list_artifacts(session_id, task_id)
    return [
        {
            "name": a.name,
            "type": a.type,
            "file_path": a.file_path,
            "size": a.size,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in artifacts
    ]

@app.get("/api/providers")
@app.get("/providers")
async def list_providers():
    """List available LLM providers."""
    providers = []
    if os.getenv("OPENAI_API_KEY") or os.getenv("MiniDevin"):
        providers.append({
            "id": "openai", "name": "OpenAI", "configured": True, "enabled": True,
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        })
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append({
            "id": "anthropic", "name": "Anthropic", "configured": True, "enabled": True,
            "models": ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"]
        })
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers.append({
            "id": "google", "name": "Google", "configured": True, "enabled": True,
            "models": ["gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash"]
        })
    return {"providers": providers}

@app.get("/api/models")
@app.get("/models")
async def list_models():
    """List available LLM models based on configured API keys."""
    models = []
    if os.getenv("OPENAI_API_KEY") or os.getenv("MiniDevin"):
        models.extend([
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai",
             "context_window": 128000, "supports_tools": True, "supports_vision": True,
             "max_output_tokens": 4096, "description": "Most capable GPT-4 model"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai",
             "context_window": 128000, "supports_tools": True, "supports_vision": True,
             "max_output_tokens": 16384, "description": "Faster, cheaper GPT-4o"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "provider": "openai",
             "context_window": 128000, "supports_tools": True, "supports_vision": True,
             "max_output_tokens": 4096, "description": "Powerful GPT-4 with vision"},
        ])
    if os.getenv("ANTHROPIC_API_KEY"):
        models.extend([
            {"id": "claude-3-5-sonnet-latest", "name": "Claude 3.5 Sonnet", "provider": "anthropic",
             "context_window": 200000, "supports_tools": True, "supports_vision": True,
             "max_output_tokens": 8192, "description": "Best Claude for coding tasks"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "provider": "anthropic",
             "context_window": 200000, "supports_tools": True, "supports_vision": False,
             "max_output_tokens": 4096, "description": "Fastest Claude model"},
        ])
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        models.extend([
            {"id": "gemini/gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "google",
             "context_window": 1000000, "supports_tools": True, "supports_vision": True,
             "max_output_tokens": 8192, "description": "Google's most capable model"},
            {"id": "gemini/gemini-1.5-flash", "name": "Gemini 1.5 Flash", "provider": "google",
             "context_window": 1000000, "supports_tools": True, "supports_vision": True,
             "max_output_tokens": 8192, "description": "Fast & efficient Gemini"},
        ])
    if not models:
        # Fallback defaults so UI always has something to show
        models = [
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai",
             "context_window": 128000, "supports_tools": True, "supports_vision": True,
             "max_output_tokens": 4096, "description": "Requires OPENAI_API_KEY"},
        ]
    return {"models": models}

@app.get("/api/status")
@app.get("/status")
async def get_system_status():
    """Get system status with uptime and metrics."""
    # Determine browser mode
    if os.getenv("BROWSERLESS_API_KEY") or os.getenv("BROWSERLESS_WS_URL"):
        browser_mode = "browserless"
    else:
        import shutil
        browser_mode = "local" if shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome") else "unavailable"

    return {
        "status": "running",
        "mode": "lightweight",
        "version": "1.0.0",
        "active_sessions": await session_manager.get_active_session_count(),
        "total_tasks_completed": await session_manager.get_total_tasks_completed(),
        "uptime_seconds": session_manager.get_uptime_seconds(),
        "llm_configured": bool(os.getenv("OPENAI_API_KEY") or os.getenv("MiniDevin") or os.getenv("ANTHROPIC_API_KEY")),
        "browser_mode": browser_mode,
    }


@app.get("/api/skills")
async def list_skills():
    return {
        "skills": [
            {
                "id": "add-endpoint",
                "name": "Add API Endpoint",
                "description": "Add a new REST API endpoint to a FastAPI application",
                "tags": ["api", "fastapi", "backend"],
            },
            {
                "id": "add-test",
                "name": "Add Unit Test",
                "description": "Add a unit test for a function or class",
                "tags": ["testing", "pytest"],
            },
            {
                "id": "refactor-function",
                "name": "Refactor Function",
                "description": "Refactor a function to improve readability and maintainability",
                "tags": ["refactoring", "code-quality"],
            },
        ],
        "message": "Demo skills - full functionality requires local deployment with OPENAI_API_KEY",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Browser-Based UI Testing + Visual Regression endpoints
# ──────────────────────────────────────────────────────────────────────────────

class UITestRequest(BaseModel):
    suite_name: str = "UI Test"
    url: str
    steps: list
    threshold_percent: float = 0.5
    working_dir: str = "."


class VisualRegressionSetBaselineRequest(BaseModel):
    name: str
    screenshot_b64: str
    url: str = ""
    working_dir: str = "."


class VisualRegressionCompareRequest(BaseModel):
    name: str
    screenshot_b64: str
    threshold_percent: Optional[float] = None
    working_dir: str = "."


@app.post("/api/ui-test/run")
async def run_ui_test(req: UITestRequest):
    """Run a structured browser UI test suite via Playwright."""
    from ..tools.browser.ui_tester import UITestRunner, steps_from_spec
    steps = steps_from_spec(req.steps)
    runner = UITestRunner(working_dir=req.working_dir, headless=True)
    result = await runner.run(
        suite_name=req.suite_name,
        start_url=req.url,
        steps=steps,
        threshold_percent=req.threshold_percent,
    )
    return result.to_dict()


@app.get("/api/visual-regression/baselines")
async def vr_list_baselines(working_dir: str = "."):
    """List all stored visual regression baselines."""
    from ..tools.browser.visual_regression import get_engine
    engine = get_engine(working_dir)
    return {"baselines": engine.list_baselines()}


@app.post("/api/visual-regression/set-baseline")
async def vr_set_baseline(req: VisualRegressionSetBaselineRequest):
    """Store a screenshot as the new visual baseline."""
    from ..tools.browser.visual_regression import get_engine
    engine = get_engine(req.working_dir)
    rec = engine.save_screenshot_b64(req.name, req.screenshot_b64, url=req.url, set_as_baseline=True)
    return {"message": f"Baseline set for '{req.name}'", "width": rec.width, "height": rec.height}


@app.post("/api/visual-regression/compare")
async def vr_compare(req: VisualRegressionCompareRequest):
    """Compare a screenshot against the stored baseline."""
    from ..tools.browser.visual_regression import get_engine
    engine = get_engine(req.working_dir)
    try:
        diff = engine.compare_b64(req.name, req.screenshot_b64, req.threshold_percent)
        return diff.to_dict()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/visual-regression/history/{name}")
async def vr_history(name: str, working_dir: str = "."):
    """Get the test history for a named page."""
    from ..tools.browser.visual_regression import get_engine
    engine = get_engine(working_dir)
    return engine.get_history(name)


@app.delete("/api/visual-regression/baseline/{name}")
async def vr_delete_baseline(name: str, working_dir: str = "."):
    """Delete a stored baseline."""
    from ..tools.browser.visual_regression import get_engine
    engine = get_engine(working_dir)
    removed = engine.delete_baseline(name)
    return {"removed": removed}


@app.get("/api/visual-regression/screenshot")
async def vr_screenshot(rel_path: str, working_dir: str = "."):
    """Return a stored screenshot as base64."""
    from ..tools.browser.visual_regression import get_engine
    engine = get_engine(working_dir)
    b64 = engine.get_screenshot_b64(rel_path)
    if b64:
        return {"b64": b64}
    raise HTTPException(status_code=404, detail="Screenshot not found")


# ──────────────────────────────────────────────────────────────────────────────
# Self-Healing Monitor endpoints
# ──────────────────────────────────────────────────────────────────────────────

class MonitorRegisterRequest(BaseModel):
    name: str
    health_url: str
    platform: str = "generic"
    platform_config: dict = {}
    check_interval_seconds: int = 60
    failure_threshold: int = 3
    session_id: Optional[str] = None


@app.get("/api/monitor/status")
async def monitor_status():
    """Return current state of all monitored apps + recent history."""
    from ..integrations.monitor import get_status
    return get_status()


@app.post("/api/monitor/register")
async def monitor_register(req: MonitorRegisterRequest):
    """Register an app for continuous health monitoring."""
    from ..integrations.monitor import MonitoredApp, register_app, start_monitor
    app_cfg = MonitoredApp(
        name=req.name,
        health_url=req.health_url,
        platform=req.platform,
        platform_config=req.platform_config,
        check_interval_seconds=req.check_interval_seconds,
        failure_threshold=req.failure_threshold,
        session_id=req.session_id,
    )
    register_app(app_cfg)
    start_monitor()
    return {"message": f"Registered {req.name} and started monitor"}


@app.delete("/api/monitor/{app_name}")
async def monitor_unregister(app_name: str):
    """Remove an app from continuous monitoring."""
    from ..integrations.monitor import unregister_app
    removed = unregister_app(app_name)
    return {"removed": removed}


@app.post("/api/monitor/health-check")
async def monitor_one_shot(url: str):
    """One-shot health check for any URL."""
    from ..integrations.monitor import check_app_health
    return await check_app_health(url)


@app.post("/api/monitor/start")
async def monitor_start():
    """Start the background monitor loop."""
    from ..integrations.monitor import start_monitor
    start_monitor()
    return {"message": "Monitor started"}


@app.post("/api/monitor/stop")
async def monitor_stop():
    """Stop the background monitor loop."""
    from ..integrations.monitor import stop_monitor
    stop_monitor()
    return {"message": "Monitor stopped"}


@app.post("/api/monitor/fetch-logs")
async def monitor_fetch_logs(platform: str = "docker", lines: int = 50, config: dict = {}):
    """Fetch recent logs from a cloud platform or Docker container."""
    from ..integrations.monitor import fetch_app_logs
    logs = await fetch_app_logs(platform, config, lines)
    return {"logs": logs}


# ──────────────────────────────────────────────────────────────────────────────
# Project Memory endpoints
# ──────────────────────────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""
    repo_url: Optional[str] = None
    tech_stack: list = []
    project_id: Optional[str] = None


class MemoryEntryRequest(BaseModel):
    project_id: str
    category: str = "context"
    title: str
    content: str
    tags: list = []
    importance: int = 5
    session_id: Optional[str] = None


class MemorySearchRequest(BaseModel):
    project_id: str
    query: str
    top_k: int = 5
    category: Optional[str] = None
    min_importance: int = 1


@app.post("/api/projects")
async def create_project(req: CreateProjectRequest):
    from ..integrations.project_memory import get_project_memory
    pm = get_project_memory()
    proj = pm.create_project(
        name=req.name,
        description=req.description,
        repo_url=req.repo_url,
        tech_stack=req.tech_stack,
        project_id=req.project_id,
    )
    return proj.to_dict()


@app.get("/api/projects")
async def list_projects():
    from ..integrations.project_memory import get_project_memory
    pm = get_project_memory()
    return {"projects": [p.to_dict() for p in pm.list_projects()]}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    from ..integrations.project_memory import get_project_memory
    pm = get_project_memory()
    proj = pm.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj.to_dict()


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    from ..integrations.project_memory import get_project_memory
    pm = get_project_memory()
    ok = pm.delete_project(project_id)
    return {"deleted": ok}


@app.post("/api/projects/{project_id}/memory")
async def add_memory(project_id: str, req: MemoryEntryRequest):
    from ..integrations.project_memory import get_project_memory, MemoryCategory
    pm = get_project_memory()
    try:
        cat = MemoryCategory(req.category)
    except ValueError:
        cat = MemoryCategory.CONTEXT
    entry = pm.add_entry(
        project_id=project_id,
        category=cat,
        title=req.title,
        content=req.content,
        tags=req.tags,
        importance=req.importance,
        session_id=req.session_id,
    )
    return entry.to_dict()


@app.get("/api/projects/{project_id}/memory")
async def list_memory(project_id: str, category: Optional[str] = None, min_importance: int = 1):
    from ..integrations.project_memory import get_project_memory, MemoryCategory
    pm = get_project_memory()
    cat = MemoryCategory(category) if category else None
    entries = pm.list_entries(project_id, category=cat, min_importance=min_importance)
    return {"entries": [e.to_dict() for e in entries]}


@app.post("/api/projects/{project_id}/memory/search")
async def search_memory(project_id: str, req: MemorySearchRequest):
    from ..integrations.project_memory import get_project_memory, MemoryCategory
    pm = get_project_memory()
    cat = MemoryCategory(req.category) if req.category else None
    results = pm.search(
        project_id,
        req.query,
        top_k=req.top_k,
        category=cat,
        min_importance=req.min_importance,
    )
    return {"results": results}


@app.get("/api/projects/{project_id}/memory/context")
async def get_memory_context(project_id: str, task: str = ""):
    from ..integrations.project_memory import get_project_memory
    pm = get_project_memory()
    ctx = pm.get_context_for_task(project_id, task)
    return {"context": ctx}


@app.delete("/api/memory/{entry_id}")
async def delete_memory_entry(entry_id: str):
    from ..integrations.project_memory import get_project_memory
    pm = get_project_memory()
    ok = pm.delete_entry(entry_id)
    return {"deleted": ok}


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical Project Plan endpoints
# ──────────────────────────────────────────────────────────────────────────────

class CreatePlanRequest(BaseModel):
    project_id: str
    goal: str
    milestones: Optional[list] = None
    working_dir: str = "."


class RetryMilestoneRequest(BaseModel):
    milestone_id: str


@app.post("/api/project-plans")
async def create_plan(req: CreatePlanRequest):
    """Decompose a project goal into milestones using LLM."""
    from ..integrations.hierarchical_planner import get_planner
    from ..integrations.project_memory import get_project_memory
    planner = get_planner()
    # Fetch project context for better decomposition
    try:
        pm = get_project_memory()
        ctx = pm.get_context_for_task(req.project_id, req.goal, max_tokens=400) if pm.get_project(req.project_id) else ""
    except Exception:
        ctx = ""
    plan = await planner.create_plan(
        project_id=req.project_id,
        goal=req.goal,
        milestones=req.milestones,
        working_dir=req.working_dir,
        project_context=ctx,
    )
    return plan.to_dict()


@app.get("/api/project-plans")
async def list_plans(project_id: Optional[str] = None):
    from ..integrations.hierarchical_planner import get_planner
    planner = get_planner()
    plans = planner.list_plans(project_id)
    return {"plans": [p.to_dict() for p in plans]}


@app.get("/api/project-plans/{plan_id}")
async def get_plan(plan_id: str):
    from ..integrations.hierarchical_planner import get_planner
    planner = get_planner()
    plan = planner.get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return plan.to_dict()


@app.delete("/api/project-plans/{plan_id}")
async def delete_plan(plan_id: str):
    from ..integrations.hierarchical_planner import get_planner
    planner = get_planner()
    ok = planner.delete_plan(plan_id)
    return {"deleted": ok}


@app.post("/api/project-plans/{plan_id}/execute")
async def execute_plan(plan_id: str):
    """Start executing remaining milestones as sub-agent tasks."""
    from ..integrations.hierarchical_planner import get_planner
    planner = get_planner()
    plan = planner.get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    asyncio.create_task(
        planner.execute_plan(
            plan_id,
            session_manager,
            connection_manager,
        )
    )
    return {"message": f"Execution started for plan {plan_id}", "plan_id": plan_id}


@app.post("/api/project-plans/{plan_id}/retry")
async def retry_milestone_endpoint(plan_id: str, req: RetryMilestoneRequest):
    from ..integrations.hierarchical_planner import get_planner
    planner = get_planner()
    ok = planner.retry_milestone(plan_id, req.milestone_id)
    return {"reset": ok}


# ──────────────────────────────────────────────────────────────────────────────
# Environment Parity endpoints
# ──────────────────────────────────────────────────────────────────────────────

class EnvDiffRequest(BaseModel):
    project_root: str = "."
    env_file: str = ".env"
    production_env: Optional[dict] = None


class GenerateDockerfileRequest(BaseModel):
    project_root: str = "."
    project_type: str = "auto"
    frontend_dir: str = "frontend"
    requirements_file: str = "requirements.txt"
    port: Optional[int] = None
    health_path: str = "/health"
    output_path: Optional[str] = None


class GenerateEnvExampleRequest(BaseModel):
    project_root: str = "."
    source_env_file: str = ".env"
    output_path: Optional[str] = None
    include_current_values: bool = False


class GenerateComposeRequest(BaseModel):
    project_root: str = "."
    port: Optional[int] = None
    include_redis: bool = False
    include_postgres: bool = False
    output_path: Optional[str] = None


@app.post("/api/env-parity/diff")
async def env_parity_diff(req: EnvDiffRequest):
    """Compare local .env with production environment."""
    from ..integrations.env_parity import diff_environments
    import os as _os
    full_env = str(_os.path.join(req.project_root, req.env_file))
    return diff_environments(local_env_file=full_env, production_env=req.production_env)


@app.post("/api/env-parity/generate-dockerfile")
async def env_parity_dockerfile(req: GenerateDockerfileRequest):
    """Generate a production-ready Dockerfile for the project."""
    from ..integrations.env_parity import generate_dockerfile
    content, path = generate_dockerfile(
        req.project_root,
        project_type=req.project_type,
        frontend_dir=req.frontend_dir,
        requirements_file=req.requirements_file,
        port=req.port,
        health_path=req.health_path,
        output_path=req.output_path,
    )
    return {"content": content, "path": path}


@app.post("/api/env-parity/generate-env-example")
async def env_parity_env_example(req: GenerateEnvExampleRequest):
    """Generate .env.example from an existing .env file."""
    from ..integrations.env_parity import generate_env_example
    content, path = generate_env_example(
        req.project_root,
        source_env_file=req.source_env_file,
        output_path=req.output_path,
        include_current_values=req.include_current_values,
    )
    return {"content": content, "path": path}


@app.post("/api/env-parity/generate-docker-compose")
async def env_parity_compose(req: GenerateComposeRequest):
    """Generate a docker-compose.yml for local dev parity."""
    from ..integrations.env_parity import generate_docker_compose
    content, path = generate_docker_compose(
        req.project_root,
        port=req.port,
        include_redis=req.include_redis,
        include_postgres=req.include_postgres,
        output_path=req.output_path,
    )
    return {"content": content, "path": path}


@app.websocket("/api/ws/{session_id}")
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await connection_manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received from {session_id}: {data}")
            
            # Check if session exists, create if not
            session = await session_manager.get_session(session_id)
            if not session:
                session = await session_manager.create_session(session_id=session_id)
                session_id = session.session_id
            
            # If agent is currently running -> inject as follow-up
            if session.status.value == 'running':
                await session_manager.inject_followup(session_id, data)
                # Acknowledge to UI
                await connection_manager.broadcast_to_session(session_id, WebSocketMessage(
                    type=MessageType.TOKEN,
                    data={"content": f"\n\n**[Follow-up received]** {data}\n\n"},
                    task_id=session.current_task_id,
                ))
                continue
            
            # Create a new task
            task = await session_manager.create_task(
                session_id=session_id,
                description=data,
                connection_manager=connection_manager,
            )
            
            asyncio.create_task(session_manager.run_task(
                session_id=session_id,
                task_id=task.task_id,
                connection_manager=connection_manager
            ))

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────────
# Serve React frontend (SPA) — must be LAST, after all API routes
# ─────────────────────────────────────────────────────────────
import pathlib

_FRONTEND_DIST = pathlib.Path(__file__).parent.parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    # Serve static assets (JS, CSS, images…)
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for non-API routes so React Router works.

        Unknown /api/* paths must not return the SPA shell (would confuse clients and tests).
        """
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        index = _FRONTEND_DIST / "index.html"
        return FileResponse(str(index))


# ──────────────────────────────────────────────────────────────────────────────
# SWE-bench Benchmarking endpoints
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkRunRequest(BaseModel):
    split: str = "lite"
    limit: int = 5
    repo_filter: str = ""
    name: str = ""
    use_agent: bool = True


@app.get("/api/benchmark/tasks")
async def list_benchmark_tasks(
    split: str = "lite",
    limit: int = 10,
    repo_filter: str = "",
):
    """List available SWE-bench tasks (preview, no run)."""
    from ..integrations.swe_bench import load_tasks
    tasks = load_tasks(split=split, limit=limit, repo_filter=repo_filter)
    return {"tasks": [t.to_dict() for t in tasks], "total": len(tasks)}


@app.get("/api/benchmark/runs")
async def list_benchmark_runs():
    from ..integrations.swe_bench import get_runner
    return {"runs": get_runner().list_runs()}


@app.get("/api/benchmark/runs/{run_id}")
async def get_benchmark_run(run_id: str):
    from ..integrations.swe_bench import get_runner
    run = get_runner().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/benchmark/runs/{run_id}/results")
async def get_benchmark_run_results(run_id: str):
    from ..integrations.swe_bench import get_runner
    results = get_runner().get_run_results(run_id)
    return {"results": results}


@app.post("/api/benchmark/runs")
async def start_benchmark_run(req: BenchmarkRunRequest, background_tasks: BackgroundTasks):
    from ..integrations.swe_bench import get_runner, make_agent_runner
    runner = get_runner()
    agent_fn = make_agent_runner(db_manager) if req.use_agent else None

    async def _run():
        await runner.start_run(
            split=req.split,
            limit=req.limit,
            repo_filter=req.repo_filter,
            name=req.name,
            agent_runner=agent_fn,
        )

    background_tasks.add_task(_run)

    # Return a preview run object immediately
    from ..integrations.swe_bench import BenchmarkRun
    from ..integrations.swe_bench import load_tasks
    tasks = load_tasks(split=req.split, limit=req.limit, repo_filter=req.repo_filter)
    preview = BenchmarkRun(
        name=req.name or f"SWE-bench {req.split} × {len(tasks)}",
        split=req.split,
        limit=req.limit,
        repo_filter=req.repo_filter,
        status="running",
        task_ids=[t.task_id for t in tasks],
        total=len(tasks),
    )
    return preview.to_dict()


@app.delete("/api/benchmark/runs/{run_id}")
async def delete_benchmark_run(run_id: str):
    from ..integrations.swe_bench import get_runner
    ok = get_runner().delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"deleted": True}


@app.post("/api/benchmark/runs/cancel")
async def cancel_benchmark_run():
    from ..integrations.swe_bench import get_runner
    get_runner().cancel_run()
    return {"cancelled": True}


@app.get("/api/benchmark/stats")
async def benchmark_stats():
    """Aggregate stats across all runs."""
    from ..integrations.swe_bench import get_runner
    runner = get_runner()
    runs = runner.list_runs()
    completed = [r for r in runs if r["status"] == "completed"]
    total_resolved = sum(r["resolved"] for r in completed)
    total_tasks = sum(r["total"] for r in completed)
    return {
        "total_runs": len(runs),
        "completed_runs": len(completed),
        "total_tasks_evaluated": total_tasks,
        "total_resolved": total_resolved,
        "overall_resolve_rate": round(total_resolved / total_tasks * 100, 1) if total_tasks > 0 else 0.0,
        "best_run": max(completed, key=lambda r: r["resolve_rate"])["run_id"] if completed else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
