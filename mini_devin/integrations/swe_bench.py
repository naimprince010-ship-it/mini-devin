"""
SWE-bench Integration for Plodder
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
import textwrap
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ── Persistence ────────────────────────────────────────────────────────────────
_DATA_DIR = Path(os.environ.get("PLODDER_DATA") or os.environ.get("MINI_DEVIN_DATA", "data")) / "swe_bench"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_RUNS_FILE = _DATA_DIR / "runs.json"
_RESULTS_FILE = _DATA_DIR / "results.json"
_PATCH_EXCLUDED_ROOTS = (".plodder/",)
_PATCH_EXCLUDED_FILES = {"PLAN.md"}


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
    agent_log: str = ""       # summary + timeline/actions captured during agent execution
    agent_session_id: str = ""
    agent_task_id: str = ""
    workspace: str = ""
    attempt_count: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    test_output: str = ""     # stdout of test run
    fail_to_pass_results: dict[str, bool] = field(default_factory=dict)
    pass_to_pass_results: dict[str, bool] = field(default_factory=dict)
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, BenchmarkStatus) else str(self.status)
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
        "task_id": "plodder_fixture__string_utils_casefold",
        "repo": "plodder-fixtures/string-utils",
        "instance_id": "plodder_fixture__string_utils_casefold",
        "base_commit": "local:string-utils-v1",
        "problem_statement": (
            "**Bug**: `contains_word(text, word)` is supposed to be case-insensitive, "
            "but it currently checks the original strings directly.\n\n"
            "**Expected behavior**: `contains_word('Hello World', 'hello')` should return `True`."
        ),
        "hints_text": "Look at string_utils.py and keep the implementation simple.",
        "test_patch": "",
        "fail_to_pass": ["tests/test_string_utils.py::test_contains_word_is_case_insensitive"],
        "pass_to_pass": [],
        "version": "fixture-v1",
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


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _setup_fixture_repo(task: SWEBenchTask, workspace_root: str) -> str:
    """Create a small real Git repo for fallback benchmark smoke tests."""
    repo_dir = Path(workspace_root) / task.instance_id
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
    repo_dir.mkdir(parents=True, exist_ok=True)

    _write_file(
        repo_dir / "string_utils.py",
        """
        def contains_word(text: str, word: str) -> bool:
            \"\"\"Return whether word appears inside text, ignoring case.\"\"\"
            return word in text
        """,
    )
    _write_file(
        repo_dir / "tests" / "test_string_utils.py",
        """
        from string_utils import contains_word


        def test_contains_word_is_case_insensitive():
            assert contains_word("Hello World", "hello") is True
        """,
    )
    _write_file(
        repo_dir / "README.md",
        """
        # Plodder benchmark fixture

        This tiny repository is generated locally so benchmark smoke tests have
        a real failing test and a meaningful patch to inspect when HuggingFace
        SWE-bench data is unavailable.
        """,
    )

    _git("init", cwd=str(repo_dir))
    _git("config", "user.email", "benchmark@plodder.local", cwd=str(repo_dir))
    _git("config", "user.name", "Plodder Benchmark", cwd=str(repo_dir))
    _git("add", ".", cwd=str(repo_dir))
    _git("commit", "-m", "Initial failing fixture", cwd=str(repo_dir))
    return str(repo_dir)


def _current_head(repo_dir: str) -> str:
    rc, out, _ = _git("rev-parse", "HEAD", cwd=repo_dir)
    return out.strip() if rc == 0 else "HEAD"


def _setup_repo(task: SWEBenchTask, workspace_root: str) -> str | None:
    """Clone / reset repo to task's base commit. Returns local path or None."""
    if task.base_commit.startswith("local:") or task.repo.startswith("plodder-fixtures/"):
        return _setup_fixture_repo(task, workspace_root)

    repo_dir = os.path.join(workspace_root, task.instance_id)
    clone_url = f"https://github.com/{task.repo}.git"

    if os.path.exists(repo_dir) and not os.path.isdir(os.path.join(repo_dir, ".git")):
        shutil.rmtree(repo_dir, ignore_errors=True)

    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        Path(workspace_root).mkdir(parents=True, exist_ok=True)
        rc, _, err = _git("clone", "--depth=50", clone_url, repo_dir, cwd=workspace_root)
        if rc != 0:
            print(f"[swe_bench] clone failed: {err}")
            shutil.rmtree(repo_dir, ignore_errors=True)
            return None

    # Reset to base commit if available
    if task.base_commit and len(task.base_commit) > 6 and not task.base_commit.startswith("abc"):
        rc, _, err = _git("fetch", "--depth=50", "origin", task.base_commit, cwd=repo_dir)
        if rc != 0:
            print(f"[swe_bench] fetch failed for {task.instance_id}: {err}")
            return None
        rc, _, err = _git("checkout", task.base_commit, cwd=repo_dir)
        if rc != 0:
            print(f"[swe_bench] checkout failed for {task.instance_id}: {err}")
            return None

    # Apply test patch so failing tests exist
    if task.test_patch:
        patch_path = os.path.join(repo_dir, "_test.patch")
        Path(patch_path).write_text(task.test_patch)
        rc, _, err = _git("apply", "--allow-empty", patch_path, cwd=repo_dir)
        if rc != 0:
            print(f"[swe_bench] test patch apply failed for {task.instance_id}: {err}")
            return None

    return repo_dir


def _collect_patch(repo_dir: str, base_ref: str = "HEAD") -> str:
    """Return git diff of tracked, staged, and untracked text changes."""
    parts: list[str] = []
    _, diff, _ = _git(
        "diff",
        "--binary",
        base_ref,
        "--",
        ".",
        ":!.plodder",
        ":!PLAN.md",
        cwd=repo_dir,
    )
    if diff:
        parts.append(diff)

    _, untracked, _ = _git("ls-files", "--others", "--exclude-standard", cwd=repo_dir)
    for rel in [line.strip() for line in untracked.splitlines() if line.strip()]:
        rel_posix = rel.replace("\\", "/")
        if rel_posix in _PATCH_EXCLUDED_FILES or rel_posix.startswith(_PATCH_EXCLUDED_ROOTS):
            continue
        path = Path(repo_dir) / rel
        if not path.is_file() or path.stat().st_size > 200_000:
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rc, out, err = _git("diff", "--no-index", "--", os.devnull, rel, cwd=repo_dir)
        if out:
            parts.append(out)
        elif rc not in (0, 1) and err:
            parts.append(f"# Failed to diff untracked file {rel}: {err}")

    return "\n".join(part.rstrip() for part in parts if part).strip()


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


def _trim_text(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n\n...[truncated]...\n\n" + text[-limit // 2 :]


def _collect_agent_timeline(workspace: str, limit: int = 80) -> str:
    """Read the agent JSONL timeline and format a compact debug log."""
    try:
        from ..orchestrator.event_stream import EventStream

        events = EventStream(workspace).to_export_list(max_lines=limit)
    except Exception as e:
        return f"timeline unavailable: {e}"

    lines: list[str] = []
    for event in events[-limit:]:
        kind = event.get("kind") or event.get("type") or "event"
        title = event.get("title") or event.get("tool") or ""
        summary = event.get("summary") or event.get("text") or event.get("output") or ""
        line = f"[{kind}]"
        if title:
            line += f" {title}"
        if summary:
            line += f": {str(summary).strip()}"
        lines.append(line[:2000])
    return _trim_text("\n".join(lines))


def _format_agent_log(
    *,
    status: str,
    summary: str,
    commands: list[str],
    files: list[str],
    timeline: str,
    error: str = "",
) -> str:
    sections = [
        f"status: {status or 'unknown'}",
        f"summary:\n{summary or '(empty)'}",
    ]
    if error:
        sections.append(f"error:\n{error}")
    sections.append("commands_executed:\n" + ("\n".join(f"- {c}" for c in commands) if commands else "(none)"))
    sections.append("files_modified:\n" + ("\n".join(f"- {f}" for f in files) if files else "(none)"))
    sections.append("timeline:\n" + (timeline or "(empty)"))
    return _trim_text("\n\n".join(sections))


def _normalise_agent_output(raw: Any, repo_dir: str | None, base_ref: str = "HEAD") -> dict[str, Any]:
    if isinstance(raw, dict):
        out = dict(raw)
    elif isinstance(raw, tuple):
        patch = raw[0] if len(raw) > 0 else ""
        log = raw[1] if len(raw) > 1 else ""
        out = {"patch": patch, "agent_log": log}
    else:
        out = {"patch": "", "agent_log": str(raw or "")}

    if repo_dir and not out.get("patch"):
        out["patch"] = _collect_patch(repo_dir, base_ref=base_ref)
    return out


async def _call_agent_runner(
    agent_runner: Any,
    task: SWEBenchTask,
    repo_dir: str | None,
    retry_feedback: str,
) -> Any:
    """Call newer retry-aware agent runners while preserving older callables."""
    try:
        return await agent_runner(task, repo_dir, retry_feedback)
    except TypeError as e:
        if retry_feedback:
            raise
        try:
            return await agent_runner(task, repo_dir)
        except TypeError:
            raise e


def _build_retry_feedback(
    *,
    attempt_no: int,
    patch: str,
    agent_log: str,
    test_output: str,
    fail_to_pass_results: dict[str, bool],
    pass_to_pass_results: dict[str, bool],
) -> str:
    return _trim_text(
        "\n\n".join(
            [
                f"## Retry Context\nThe previous benchmark attempt #{attempt_no} did not resolve the task.",
                "Use the failure details below, inspect the repository again, fix the root cause, and rerun the relevant tests.",
                "### Previous Test Output\n" + (test_output or "(empty)"),
                "### Fail-to-Pass Results\n" + json.dumps(fail_to_pass_results, indent=2, default=str),
                "### Pass-to-Pass Results\n" + json.dumps(pass_to_pass_results, indent=2, default=str),
                "### Previous Patch\n" + (patch or "(empty)"),
                "### Previous Agent Log\n" + (agent_log or "(empty)"),
            ]
        ),
        limit=16000,
    )


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
        retry_limit: int = 1,
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

            result = await self._run_single_task(
                task,
                run.run_id,
                agent_runner,
                retry_limit=retry_limit,
            )
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
        retry_limit: int = 1,
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
                    agent_out = _normalise_agent_output(await agent_runner(task, None), None)
                    result.patch = agent_out.get("patch", "")
                    result.agent_log = agent_out.get("agent_log", "")
                    result.agent_session_id = agent_out.get("agent_session_id", "")
                    result.agent_task_id = agent_out.get("agent_task_id", "")
                    result.workspace = agent_out.get("workspace", "")
                result.status = BenchmarkStatus.UNRESOLVED
            else:
                result.workspace = repo_dir
                base_ref = _current_head(repo_dir)
                retry_feedback = ""
                max_attempts = 1 + max(0, retry_limit if agent_runner else 0)
                for attempt_no in range(1, max_attempts + 1):
                    if agent_runner:
                        agent_out = _normalise_agent_output(
                            await _call_agent_runner(agent_runner, task, repo_dir, retry_feedback),
                            repo_dir,
                            base_ref=base_ref,
                        )
                        result.patch = agent_out.get("patch", "")
                        result.agent_log = agent_out.get("agent_log", "")
                        result.agent_session_id = agent_out.get("agent_session_id", "")
                        result.agent_task_id = agent_out.get("agent_task_id", "")
                        result.workspace = agent_out.get("workspace", repo_dir)
                    else:
                        result.patch = ""
                        result.agent_log = "No agent runner provided."

                    if not result.patch and repo_dir:
                        result.patch = _collect_patch(repo_dir, base_ref=base_ref)

                    all_tests = task.fail_to_pass + task.pass_to_pass
                    if all_tests and repo_dir:
                        test_out, ftp_results = _run_tests(repo_dir, task.fail_to_pass)
                        _, ptp_results = _run_tests(repo_dir, task.pass_to_pass)
                        result.test_output = test_out
                        result.fail_to_pass_results = ftp_results
                        result.pass_to_pass_results = ptp_results

                        ftp_ok = all(ftp_results.values()) if ftp_results else False
                        ptp_ok = all(ptp_results.values()) if ptp_results else True
                        result.status = (
                            BenchmarkStatus.RESOLVED if (ftp_ok and ptp_ok)
                            else BenchmarkStatus.UNRESOLVED
                        )
                    else:
                        result.status = (
                            BenchmarkStatus.RESOLVED if len(result.patch) > 50
                            else BenchmarkStatus.UNRESOLVED
                        )

                    result.attempt_count = attempt_no
                    result.attempts.append(
                        {
                            "attempt": attempt_no,
                            "status": result.status.value,
                            "agent_session_id": result.agent_session_id,
                            "agent_task_id": result.agent_task_id,
                            "patch_chars": len(result.patch or ""),
                            "test_output": result.test_output,
                            "fail_to_pass_results": result.fail_to_pass_results,
                            "pass_to_pass_results": result.pass_to_pass_results,
                            "agent_log": result.agent_log,
                        }
                    )
                    _save_results(self._results)

                    if result.status == BenchmarkStatus.RESOLVED or attempt_no >= max_attempts:
                        break

                    retry_feedback = _build_retry_feedback(
                        attempt_no=attempt_no,
                        patch=result.patch,
                        agent_log=result.agent_log,
                        test_output=result.test_output,
                        fail_to_pass_results=result.fail_to_pass_results,
                        pass_to_pass_results=result.pass_to_pass_results,
                    )

                result.finished_at = datetime.now(timezone.utc).isoformat()
                result.duration_s = round(time.time() - t0, 1)
                _save_results(self._results)
                return result
                # Agent solves the task
                if agent_runner:
                    agent_out = _normalise_agent_output(
                        await agent_runner(task, repo_dir),
                        repo_dir,
                        base_ref=base_ref,
                    )
                    result.patch = agent_out.get("patch", "")
                    result.agent_log = agent_out.get("agent_log", "")
                    result.agent_session_id = agent_out.get("agent_session_id", "")
                    result.agent_task_id = agent_out.get("agent_task_id", "")
                    result.workspace = agent_out.get("workspace", repo_dir)
                else:
                    result.patch = ""
                    result.agent_log = "No agent runner provided."

                # Collect actual diff
                if not result.patch and repo_dir:
                    result.patch = _collect_patch(repo_dir, base_ref=base_ref)

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

def make_agent_runner(db_manager: Any, model: str = "auto"):
    """Return an async callable that runs the agent on one SWE-bench task."""

    async def _run(task: SWEBenchTask, repo_dir: str | None, retry_feedback: str = "") -> tuple[str, str]:
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
        if retry_feedback:
            prompt += (
                "\n\n## Previous Attempt Failed\n\n"
                f"{retry_feedback}\n\n"
                "Continue from the current workspace state. Fix the actual root cause, "
                "run the relevant tests again, and finish only when they pass or you have a clear blocker."
            )

        try:
            mgr = db_manager
            # Create a throwaway session
            session = await mgr.create_session(
                working_directory=workspace,
                model=model,
            )
            sid = session.session_id
            created_task_id = ""

            created_task = await mgr.create_task(
                session_id=sid,
                description=prompt,
            )
            created_task_id = created_task.task_id
            task_result = await mgr.run_task(
                session_id=sid,
                task_id=created_task_id,
            )

            # Gather patch
            patch = ""
            if repo_dir:
                patch = _collect_patch(repo_dir)

            if isinstance(task_result, dict):
                status = str(task_result.get("status") or "")
                summary = str(task_result.get("summary") or "")
                commands = task_result.get("commands_executed") or []
                files = task_result.get("files_modified") or []
                error = str(task_result.get("error_message") or "")
            else:
                status = str(getattr(task_result, "status", "") or "")
                summary = str(getattr(task_result, "summary", "") or "")
                commands = list(getattr(task_result, "commands_executed", []) or [])
                files = list(getattr(task_result, "files_modified", []) or [])
                error = str(getattr(task_result, "error_message", "") or "")

            timeline = _collect_agent_timeline(workspace)
            log = _format_agent_log(
                status=status,
                summary=summary,
                commands=commands,
                files=files,
                timeline=timeline,
                error=error,
            )

            # Clean up session
            try:
                await mgr.delete_session(sid)
            except Exception:
                pass

            return {
                "patch": patch,
                "agent_log": log,
                "agent_session_id": sid,
                "agent_task_id": created_task_id,
                "workspace": workspace,
            }
        except Exception as e:
            return {
                "patch": _collect_patch(repo_dir) if repo_dir else "",
                "agent_log": f"Agent error: {e}",
                "workspace": workspace,
            }

    return _run
