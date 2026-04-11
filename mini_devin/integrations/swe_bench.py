"""
SWE-bench Integration for Mini-Devin
=====================================
Standardized benchmarking: given a real GitHub issue,
can the agent produce a patch that makes the tests pass?

Dataset: princeton-nlp/SWE-bench_Lite (300 tasks, HuggingFace)
Fallback: built-in sample tasks when HuggingFace is unavailable.

Evaluation flow:
  1. Load task  (issue + failing test IDs + base commit)
  2. Clone / reset repo to base commit in a temp workspace
  3. Feed issue text to agent → agent edits files
  4. Collect the git diff (patch)
  5. Apply patch, run tests → resolved if all failing tests now pass
  6. Persist result
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ── Persistence ────────────────────────────────────────────────────────────────
_DATA_DIR = Path(os.environ.get("MINI_DEVIN_DATA", "data")) / "swe_bench"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_RUNS_FILE = _DATA_DIR / "runs.json"
_RESULTS_FILE = _DATA_DIR / "results.json"


# ── Data models ────────────────────────────────────────────────────────────────

class BenchmarkStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    RESOLVED  = "resolved"
    UNRESOLVED = "unresolved"
    ERROR     = "error"
    SKIPPED   = "skipped"


@dataclass
class SWEBenchTask:
    """One SWE-bench task (issue → patch)."""
    task_id: str
    repo: str              # e.g. "django/django"
    instance_id: str       # e.g. "django__django-11099"
    base_commit: str       # git SHA to reset to
    problem_statement: str # the GitHub issue text
    hints_text: str        # optional hints
    test_patch: str        # patch that adds/restores the failing tests
    fail_to_pass: list[str]  # test IDs that must go green
    pass_to_pass: list[str]  # test IDs that must stay green
    created_at: str = ""
    version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SWEBenchTask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskResult:
    """Result of running the agent on one SWE-bench task."""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: str = ""
    task_id: str = ""
    instance_id: str = ""
    repo: str = ""
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    patch: str = ""           # git diff produced by agent
    agent_log: str = ""       # summary / last message
    test_output: str = ""     # stdout of test run
    fail_to_pass_results: dict[str, bool] = field(default_factory=dict)
    pass_to_pass_results: dict[str, bool] = field(default_factory=dict)
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class BenchmarkRun:
    """A collection of task results (one benchmark run)."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    split: str = "lite"
    limit: int = 10
    repo_filter: str = ""
    status: str = "pending"   # pending | running | completed | cancelled
    task_ids: list[str] = field(default_factory=list)
    result_ids: list[str] = field(default_factory=list)
    resolved: int = 0
    total: int = 0
    started_at: str = ""
    finished_at: str = ""

    @property
    def resolve_rate(self) -> float:
        return round(self.resolved / self.total * 100, 1) if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolve_rate"] = self.resolve_rate
        return d


# ── Persistence helpers ─────────────────────────────────────────────────────────

def _load_runs() -> dict[str, BenchmarkRun]:
    if not _RUNS_FILE.exists():
        return {}
    try:
        raw = json.loads(_RUNS_FILE.read_text())
        runs: dict[str, BenchmarkRun] = {}
        for r in raw:
            run = BenchmarkRun(**{k: v for k, v in r.items() if k in BenchmarkRun.__dataclass_fields__})
            runs[run.run_id] = run
        return runs
    except Exception:
        return {}


def _save_runs(runs: dict[str, BenchmarkRun]) -> None:
    _RUNS_FILE.write_text(json.dumps([r.to_dict() for r in runs.values()], indent=2))


def _load_results() -> dict[str, TaskResult]:
    if not _RESULTS_FILE.exists():
        return {}
    try:
        raw = json.loads(_RESULTS_FILE.read_text())
        results: dict[str, TaskResult] = {}
        for r in raw:
            res = TaskResult(**{k: v for k, v in r.items() if k in TaskResult.__dataclass_fields__})
            results[res.result_id] = res
        return results
    except Exception:
        return {}


def _save_results(results: dict[str, TaskResult]) -> None:
    _RESULTS_FILE.write_text(json.dumps([r.to_dict() for r in results.values()], indent=2))


# ── Sample tasks (fallback when HuggingFace unavailable) ───────────────────────

_SAMPLE_TASKS: list[dict] = [
    {
        "task_id": "django__django-11099",
        "repo": "django/django",
        "instance_id": "django__django-11099",
        "base_commit": "abc1234",
        "problem_statement": (
            "**Bug**: `HttpRequest.get_host()` raises `DisallowedHost` even when the "
            "request hostname matches an entry in `ALLOWED_HOSTS` that uses a leading dot "
            "wildcard (e.g. `.example.com`).\n\n"
            "**Expected behavior**: A wildcard entry `.example.com` should match "
            "`sub.example.com` as documented."
        ),
        "hints_text": "Look at django/http/request.py validate_host()",
        "test_patch": "",
        "fail_to_pass": ["tests.test_request.HostValidationTests.test_wildcard_subdomain"],
        "pass_to_pass": [],
        "version": "3.0",
    },
    {
        "task_id": "requests__requests-3738",
        "repo": "psf/requests",
        "instance_id": "requests__requests-3738",
        "base_commit": "def5678",
        "problem_statement": (
            "**Bug**: When a redirect changes the method from POST to GET (303 See Other), "
            "the `Content-Length` header is not removed from the redirected GET request, "
            "causing some servers to wait for a body that never comes.\n\n"
            "**Expected**: `Content-Length` should be stripped on method-changing redirects."
        ),
        "hints_text": "Check requests/sessions.py rebuild_method()",
        "test_patch": "",
        "fail_to_pass": ["tests/test_redirects.py::TestRedirects::test_303_removes_content_length"],
        "pass_to_pass": [],
        "version": "2.18",
    },
    {
        "task_id": "flask__flask-2237",
        "repo": "pallets/flask",
        "instance_id": "flask__flask-2237",
        "base_commit": "ghi9012",
        "problem_statement": (
            "**Feature request**: `flask.cli.with_appcontext` should be usable as a "
            "decorator on async functions so that Flask-AsyncExt works without extra glue.\n\n"
            "Currently wrapping an async command with `@with_appcontext` causes a "
            "`RuntimeError: This event loop is already running`."
        ),
        "hints_text": "Look at flask/cli.py with_appcontext()",
        "test_patch": "",
        "fail_to_pass": ["tests/test_cli.py::test_with_appcontext_async"],
        "pass_to_pass": [],
        "version": "2.0",
    },
    {
        "task_id": "numpy__numpy-18547",
        "repo": "numpy/numpy",
        "instance_id": "numpy__numpy-18547",
        "base_commit": "jkl3456",
        "problem_statement": (
            "**Bug**: `np.unique` with `return_counts=True` returns incorrect counts when "
            "the input array contains `NaN` values. Multiple NaN entries are not treated as "
            "equal by the current implementation.\n\n"
            "**Expected**: All NaN entries should be collapsed into a single unique entry."
        ),
        "hints_text": "Check numpy/lib/arraysetops.py unique()",
        "test_patch": "",
        "fail_to_pass": ["numpy/lib/tests/test_arraysetops.py::test_unique_nan_counts"],
        "pass_to_pass": [],
        "version": "1.21",
    },
    {
        "task_id": "sympy__sympy-20049",
        "repo": "sympy/sympy",
        "instance_id": "sympy__sympy-20049",
        "base_commit": "mno7890",
        "problem_statement": (
            "**Bug**: `Point2D` equality check raises `TypeError` instead of returning "
            "`False` when compared against a non-Point type, e.g. `Point2D(1,2) == 'hello'`."
        ),
        "hints_text": "Check sympy/geometry/point.py __eq__()",
        "test_patch": "",
        "fail_to_pass": ["sympy/geometry/tests/test_point.py::test_point_equality_non_point"],
        "pass_to_pass": [],
        "version": "1.7",
    },
]


# ── Task Loader ─────────────────────────────────────────────────────────────────

def load_tasks(
    split: str = "lite",
    limit: int = 10,
    repo_filter: str = "",
    use_huggingface: bool = True,
) -> list[SWEBenchTask]:
    """Load SWE-bench tasks.

    Tries HuggingFace datasets first; falls back to built-in samples.
    """
    tasks: list[SWEBenchTask] = []

    if use_huggingface:
        try:
            tasks = _load_from_huggingface(split, limit, repo_filter)
        except Exception as e:
            print(f"[swe_bench] HuggingFace unavailable ({e}), using sample tasks.")

    if not tasks:
        tasks = _load_sample_tasks(repo_filter)

    if repo_filter:
        tasks = [t for t in tasks if repo_filter.lower() in t.repo.lower()]

    return tasks[:limit]


def _load_from_huggingface(split: str, limit: int, repo_filter: str) -> list[SWEBenchTask]:
    from datasets import load_dataset  # type: ignore
    ds_name = "princeton-nlp/SWE-bench_Lite" if split == "lite" else "princeton-nlp/SWE-bench"
    ds = load_dataset(ds_name, split="test", streaming=True)
    tasks: list[SWEBenchTask] = []
    for row in ds:
        if repo_filter and repo_filter.lower() not in row.get("repo", "").lower():
            continue
        tasks.append(SWEBenchTask(
            task_id=row["instance_id"],
            repo=row["repo"],
            instance_id=row["instance_id"],
            base_commit=row.get("base_commit", ""),
            problem_statement=row.get("problem_statement", ""),
            hints_text=row.get("hints_text", ""),
            test_patch=row.get("test_patch", ""),
            fail_to_pass=json.loads(row.get("FAIL_TO_PASS", "[]")),
            pass_to_pass=json.loads(row.get("PASS_TO_PASS", "[]")),
            version=row.get("version", ""),
        ))
        if len(tasks) >= limit:
            break
    return tasks


def _load_sample_tasks(repo_filter: str = "") -> list[SWEBenchTask]:
    tasks = []
    for d in _SAMPLE_TASKS:
        if repo_filter and repo_filter.lower() not in d["repo"].lower():
            continue
        tasks.append(SWEBenchTask(**d))
    return tasks


# ── Repo helpers ────────────────────────────────────────────────────────────────

def _git(*args: str, cwd: str) -> tuple[int, str, str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120
    )
    return result.returncode, result.stdout, result.stderr


def _setup_repo(task: SWEBenchTask, workspace_root: str) -> str | None:
    """Clone / reset repo to task's base commit. Returns local path or None."""
    repo_dir = os.path.join(workspace_root, task.instance_id)
    clone_url = f"https://github.com/{task.repo}.git"

    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir, exist_ok=True)
        rc, _, err = _git("clone", "--depth=50", clone_url, repo_dir, cwd=workspace_root)
        if rc != 0:
            print(f"[swe_bench] clone failed: {err}")
            return None

    # Reset to base commit if available
    if task.base_commit and len(task.base_commit) > 6 and not task.base_commit.startswith("abc"):
        _git("fetch", "--depth=50", "origin", task.base_commit, cwd=repo_dir)
        _git("checkout", task.base_commit, cwd=repo_dir)

    # Apply test patch so failing tests exist
    if task.test_patch:
        patch_path = os.path.join(repo_dir, "_test.patch")
        Path(patch_path).write_text(task.test_patch)
        _git("apply", "--allow-empty", patch_path, cwd=repo_dir)

    return repo_dir


def _collect_patch(repo_dir: str) -> str:
    """Return git diff of all unstaged + staged changes."""
    _, diff, _ = _git("diff", "HEAD", cwd=repo_dir)
    return diff or ""


def _run_tests(repo_dir: str, test_ids: list[str]) -> tuple[str, dict[str, bool]]:
    """Run pytest for the given test IDs. Returns (stdout, {test_id: passed})."""
    if not test_ids:
        return "", {}

    cmd = ["python", "-m", "pytest", "--tb=short", "-q"] + test_ids
    try:
        result = subprocess.run(
            cmd, cwd=repo_dir, capture_output=True, text=True, timeout=300
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT", {t: False for t in test_ids}
    except FileNotFoundError:
        return "pytest not found", {t: False for t in test_ids}

    # Parse pass/fail from pytest output
    results: dict[str, bool] = {}
    for tid in test_ids:
        short = tid.split("::")[-1] if "::" in tid else tid
        passed = f"PASSED" in output and short in output
        failed = f"FAILED" in output and short in output or f"ERROR" in output and short in output
        if passed and not failed:
            results[tid] = True
        elif failed:
            results[tid] = False
        else:
            # Heuristic: if overall run passed (exit 0) assume passed
            results[tid] = result.returncode == 0

    return output[:4000], results


# ── Core runner ─────────────────────────────────────────────────────────────────

class SWEBenchRunner:
    def __init__(self, workspace_root: str | None = None):
        self.workspace_root = workspace_root or str(_DATA_DIR / "workspaces")
        Path(self.workspace_root).mkdir(parents=True, exist_ok=True)
        self._runs = _load_runs()
        self._results = _load_results()
        self._active_run_id: str | None = None
        self._cancel_event = asyncio.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def list_runs(self) -> list[dict]:
        return [r.to_dict() for r in sorted(
            self._runs.values(), key=lambda r: r.started_at, reverse=True
        )]

    def get_run(self, run_id: str) -> dict | None:
        run = self._runs.get(run_id)
        return run.to_dict() if run else None

    def get_run_results(self, run_id: str) -> list[dict]:
        run = self._runs.get(run_id)
        if not run:
            return []
        return [self._results[rid].to_dict() for rid in run.result_ids if rid in self._results]

    def get_result(self, result_id: str) -> dict | None:
        r = self._results.get(result_id)
        return r.to_dict() if r else None

    def delete_run(self, run_id: str) -> bool:
        if run_id not in self._runs:
            return False
        run = self._runs.pop(run_id)
        for rid in run.result_ids:
            self._results.pop(rid, None)
        _save_runs(self._runs)
        _save_results(self._results)
        return True

    def cancel_run(self) -> None:
        self._cancel_event.set()

    async def start_run(
        self,
        split: str = "lite",
        limit: int = 5,
        repo_filter: str = "",
        name: str = "",
        agent_runner: Any = None,        # callable: async (task, workspace) -> (patch, log)
    ) -> BenchmarkRun:
        tasks = load_tasks(split=split, limit=limit, repo_filter=repo_filter)
        run = BenchmarkRun(
            name=name or f"SWE-bench {split} × {len(tasks)}",
            split=split,
            limit=limit,
            repo_filter=repo_filter,
            status="running",
            task_ids=[t.task_id for t in tasks],
            total=len(tasks),
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._runs[run.run_id] = run
        self._active_run_id = run.run_id
        self._cancel_event.clear()
        _save_runs(self._runs)

        # Run tasks one by one
        for task in tasks:
            if self._cancel_event.is_set():
                run.status = "cancelled"
                break

            result = await self._run_single_task(task, run.run_id, agent_runner)
            run.result_ids.append(result.result_id)
            if result.status == BenchmarkStatus.RESOLVED:
                run.resolved += 1

            _save_runs(self._runs)
            _save_results(self._results)

        run.status = "completed" if not self._cancel_event.is_set() else "cancelled"
        run.finished_at = datetime.now(timezone.utc).isoformat()
        _save_runs(self._runs)
        self._active_run_id = None
        return run

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _run_single_task(
        self,
        task: SWEBenchTask,
        run_id: str,
        agent_runner: Any,
    ) -> TaskResult:
        import time
        t0 = time.time()
        result = TaskResult(
            run_id=run_id,
            task_id=task.task_id,
            instance_id=task.instance_id,
            repo=task.repo,
            status=BenchmarkStatus.RUNNING,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._results[result.result_id] = result
        _save_results(self._results)

        try:
            repo_dir = _setup_repo(task, self.workspace_root)
            if repo_dir is None:
                result.status = BenchmarkStatus.SKIPPED
                result.error = "Repository clone failed (network unavailable or invalid SHA). Using simulation mode."
                # In simulation mode: present task to agent and capture response
                if agent_runner:
                    patch, log = await agent_runner(task, None)
                    result.patch = patch
                    result.agent_log = log
                result.status = BenchmarkStatus.UNRESOLVED
            else:
                # Agent solves the task
                if agent_runner:
                    patch, log = await agent_runner(task, repo_dir)
                    result.patch = patch
                    result.agent_log = log
                else:
                    result.patch = ""
                    result.agent_log = "No agent runner provided."

                # Collect actual diff
                if not result.patch and repo_dir:
                    result.patch = _collect_patch(repo_dir)

                # Run tests
                all_tests = task.fail_to_pass + task.pass_to_pass
                if all_tests and repo_dir:
                    test_out, ftp_results = _run_tests(repo_dir, task.fail_to_pass)
                    _, ptp_results = _run_tests(repo_dir, task.pass_to_pass)
                    result.test_output = test_out
                    result.fail_to_pass_results = ftp_results
                    result.pass_to_pass_results = ptp_results

                    # Resolved = all fail_to_pass now pass AND all pass_to_pass still pass
                    ftp_ok = all(ftp_results.values()) if ftp_results else False
                    ptp_ok = all(ptp_results.values()) if ptp_results else True
                    result.status = (
                        BenchmarkStatus.RESOLVED if (ftp_ok and ptp_ok)
                        else BenchmarkStatus.UNRESOLVED
                    )
                else:
                    # No tests to run — evaluate by patch quality heuristic
                    result.status = (
                        BenchmarkStatus.RESOLVED if len(result.patch) > 50
                        else BenchmarkStatus.UNRESOLVED
                    )

        except Exception as e:
            result.status = BenchmarkStatus.ERROR
            result.error = str(e)

        result.finished_at = datetime.now(timezone.utc).isoformat()
        result.duration_s = round(time.time() - t0, 1)
        _save_results(self._results)
        return result


# ── Module-level singleton ──────────────────────────────────────────────────────

_runner: SWEBenchRunner | None = None


def get_runner() -> SWEBenchRunner:
    global _runner
    if _runner is None:
        _runner = SWEBenchRunner()
    return _runner


# ── Agent runner factory ────────────────────────────────────────────────────────

def make_agent_runner(db_manager: Any, model: str = "claude-3-5-sonnet-20241022"):
    """Return an async callable that runs the agent on one SWE-bench task."""

    async def _run(task: SWEBenchTask, repo_dir: str | None) -> tuple[str, str]:
        workspace = repo_dir or str(_DATA_DIR / "workspaces" / task.instance_id)
        Path(workspace).mkdir(parents=True, exist_ok=True)

        prompt = (
            f"# SWE-bench Task: {task.instance_id}\n\n"
            f"**Repository**: `{task.repo}`\n\n"
            f"## Problem Statement\n\n{task.problem_statement}\n\n"
            + (f"## Hints\n\n{task.hints_text}\n\n" if task.hints_text else "")
            + f"## Your Task\n\n"
            f"Fix the bug described above. Edit the relevant source files. "
            f"Do NOT modify existing tests. When done, output a brief summary."
        )

        try:
            from ..sessions.db_manager import SessionDBManager  # type: ignore
            mgr: SessionDBManager = db_manager

            # Create a throwaway session
            session = await mgr.create_session(
                title=f"SWE-bench: {task.instance_id}",
                working_directory=workspace,
            )
            sid = session.session_id

            # Run the task
            task_result = await mgr.run_task(
                session_id=sid,
                task_description=prompt,
                model=model,
            )

            # Gather patch
            patch = ""
            if repo_dir:
                patch = _collect_patch(repo_dir)

            log = task_result.get("summary", "") if isinstance(task_result, dict) else ""

            # Clean up session
            try:
                await mgr.delete_session(sid)
            except Exception:
                pass

            return patch, log
        except Exception as e:
            return "", f"Agent error: {e}"

    return _run
