"""
Agent Orchestrator for Plodder

This module implements the main agent that orchestrates task execution
using a state machine approach with planning, execution, and verification phases.
"""
from __future__ import annotations  # Enable forward references for type hints

import asyncio
import importlib.util
import json
import os
import time
import uuid
import re
import inspect
import shlex
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

_DEFAULT_AGENT_MAX_ITERATIONS = int(os.environ.get("DEFAULT_MAX_ITERATIONS", "200"))

from ..core.llm_client import LLMClient, create_llm_client
from ..core.tool_interface import ToolRegistry
from ..schemas.state import (
    AgentPhase,
    AgentState,
    TaskGoal,
    TaskState,
    TaskStatus,
    TaskType,
    StepStatus,
)
from ..skills.playbook import discover_repo_root, load_playbook_markdown
from ..schemas.verification import (
    VerificationSuite,
    create_auto_verification_suite,
    create_python_verification_suite,
)
from ..tools.terminal import create_terminal_tool
from ..tools.editor import create_editor_tool
from ..tools.browser import (
    create_search_tool,
    create_fetch_tool,
    create_interactive_tool,
    create_playwright_tool,
    create_advanced_browser_tools,
    create_citation_store,
)
from ..safety.guards import SafetyGuard, SafetyPolicy, SafetyViolation
from ..artifacts.logger import ArtifactLogger, create_artifact_logger
from ..memory import (
    SymbolIndex,
    VectorStore,
    RetrievalManager,
    WorkingMemory,
    create_symbol_index,
    create_vector_store,
    create_retrieval_manager,
    create_working_memory,
    ConversationMemory,
    ConversationEntryType,
    Importance,
    TaskSummary,
    create_conversation_memory,
)

# Selective imports will be handled locally within methods to avoid circular dependencies
from ..core.parallel_executor import (
    ParallelExecutor,
    BatchToolCaller,
    ToolCall,
    ToolCallResult,
    ParallelExecutionResult,
    create_parallel_executor,
    create_batch_caller,
)
from ..reliability.post_mortem import (
    DiagnosticTriggerRecord,
    FailureStreakRecord,
    PostMortemPayload,
    RecoveryPathRecord,
    format_post_mortem,
    infer_recovery_path_summary,
)
from ..reliability.self_correction import (
    SelfCorrectionEngine,
    ErrorType,
    error_fingerprint,
    format_system_correction_block,
    gather_workspace_diagnostics_sync,
    incremental_recovery_hint,
)
from ..learning.teacher_review import maybe_log_teacher_review
from ..reliability.ops_telemetry import (
    FileOpsTelemetryCollector,
    telemetry_config_from_env,
)

from .planner import Planner
from .session_events import (
    build_governance_signal,
    estimate_llm_cost_usd,
    load_session_events,
    normalize_governance_signals,
)
from .activity_loop import (
    AgentActivityState,
    build_activity_meta,
    classify_action_type,
    validate_action_pre_flight,
)
from .session_worklog import SessionWorklog, load_worklog, save_worklog
from .terminal_recovery import terminal_recovery_hint
from .workspace_sidecar import WorkspaceSidecar
from .standard_events import AgentEventKind, AgentStreamEvent, append_standard_event
from ..orchestration.runtime_contracts import (
    emit_step_completed,
    emit_step_started,
    governance_budget_signals_enabled,
    governance_loop_signals_enabled,
    governance_retry_signals_enabled,
    governance_telemetry_enabled,
    runtime_contracts_enabled,
)


import sys as _sys_for_prompt


def _make_system_prompt() -> str:
    """Build system prompt with OS-specific environment notes."""
    if _sys_for_prompt.platform == "win32":
        os_note = (
            "- The agent is running on **Windows** with **PowerShell**.\n"
            "- Use `python` (not `python3`), `pip` (not `pip3`).\n"
            "- For directory listings use `Get-ChildItem -Force` or plain `dir`; do **not** use bash/cmd flags like `ls -la`, `ll`, `dir /a`, or `dir /b`.\n"
            "- Common Linux commands are auto-translated by the terminal tool: "
            "`ls`, `cat`, `mkdir -p`, `rm -rf`, `grep`, `touch`, `which`, `cp`, `mv` all work.\n"
            "- Chain commands with `;` not `&&`. Use relative paths only.\n"
            "- `pytest` may not be installed globally — prefer `python -m pytest` or `python -m unittest`."
        )
    else:
        os_note = (
            "- The agent runs on **Linux/bash**.\n"
            "- Use standard Linux commands (`python3`, `pip3`, etc.).\n"
            "- Never use Windows drive paths (`C:\\\\`, `G:\\\\`)."
        )
    has_github_token = bool(os.environ.get("GITHUB_TOKEN", "").strip())
    github_note = (
        "\n## Git/GitHub Workflow (always follow when repo work is requested)\n"
        "- On a cloned repo, **avoid committing large features directly to the default branch** when review is expected.\n"
        "- Preferred sequence: inspect issue/task context → create/switch to a feature branch → implement with ``editor``/``terminal`` → run verification/tests → commit with clear message → push/PR when auth permits.\n"
        "- If explicit GitHub tools are unavailable, perform the same flow with terminal git commands and report exact next commands needed from the user.\n"
        "- If user gives an issue number like ``#123``, treat it as the source spec and align implementation + commit/PR text to that issue.\n"
        "- Before merge-ready claims, verify branch status and CI/test outputs; do not claim merge-ready without evidence.\n"
        "- Do not push or merge unless asked (or unless task explicitly requires it).\n"
        "- **PR review comments**: If a PR has review comments (``changes_requested`` in ``get_pr_status``), call ``github`` **get_pr_comments** (``action: get_pr_comments``, ``pr_number: N``) to read every inline and thread comment before making any code changes. Address each reviewer comment explicitly.\n"
        "- **CI/CD failures**: If a workflow run fails (visible in ``get_pr_status`` → ``check_runs`` with ``conclusion: failure``), call ``github`` **get_action_logs** (``action: get_action_logs``, ``run_id: <id>``) to fetch the exact failed-job log text. Read the error output, fix the root cause in code, then push again. Do not guess at CI failures without reading the logs.\n"
    )
    if has_github_token:
        github_note += (
            "\n## GitHub API tools (``GITHUB_TOKEN`` detected)\n"
            "- Preferred sequence with GitHub tools: ``github`` **get_repo_context** / **get_issue** → ``git`` **checkout_branch** (or ``github`` **create_branch**) → implement + verify → ``git``/**github** **commit** → **create_pr**.\n"
            'For ``base_branch``, omit it or set ``"default"`` so the repo default (``main`` or ``master``) is used.\n'
            "- Use **get_pr_status** with ``pr_number`` to check **mergeable**, **mergeable_state**, and combined CI status before requesting merge.\n"
            "- Use **merge_pr** only when instructions explicitly allow merging.\n"
            "- Pass ``repo_path`` as the workspace root (use ``.`` only if that is repo root).\n"
            "- **Auto git push**: if ``auto_git_commit`` + push is enabled, Plodder skips ``git push`` on remote default branch unless ``PLODDER_GIT_PUSH_DEFAULT_BRANCH=1`` is set (local commits still run).\n"
        )
    return _SYSTEM_PROMPT_TEMPLATE.format(os_env_note=os_note, github_note=github_note)


def _runtime_context_block(working_directory: str | None) -> str:
    """Inject exact Python path and OS so the model runs commands that work on this machine."""
    exe = _sys_for_prompt.executable.replace("\\", "/")
    plat = _sys_for_prompt.platform
    wd = working_directory or "."
    root = Path(wd).resolve()
    has_package_json = (root / "package.json").is_file()
    has_python_markers = any(
        (root / name).exists()
        for name in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")
    )
    looks_node_only = has_package_json and not has_python_markers
    if looks_node_only:
        test_hint = (
            "This looks like a Node/JS workspace (`package.json` present, no Python project markers). "
            "Prefer `npm test`, `npm run build`, or `node <entry>` verification. Do not run `pytest` unless you first confirm Python tests exist."
        )
    elif plat == "win32":
        test_hint = (
            f"Use `{exe} -m pytest` or `{exe} -m unittest discover` (not bare `pytest` if it is not on PATH). "
            f"If pytest is missing, run `{exe} -m pip install pytest` or install the repo's test requirements, then retry once."
        )
    else:
        test_hint = (
            "Use `python3 -m pytest` or `python3 -m unittest discover` when `pytest` is missing from PATH. "
            "If pytest is missing, run `python3 -m pip install pytest` or install the repo's test requirements, then retry once."
        )
    return f"""## Runtime context (use these exact values in terminal commands)
- **Platform**: `{plat}`
- **Python executable**: `{exe}`
- **Workspace cwd** for terminal: `{wd}`
- **Tests**: {test_hint}
- **Windows listing rule**: if Platform is `win32`, use `Get-ChildItem -Force` or `editor list_directory`; never use `ls -la`, `ll`, `dir /a`, or `dir /b`.
- **Unit tests you write** must assert behavior that matches your implementation (avoid contradictory expected values)."""


_SYSTEM_PROMPT_TEMPLATE = """
## CRITICAL RULES — READ FIRST
- **If the workspace root already contains `.git`**, this folder **is already a Git checkout** (e.g. the UI cloned a GitHub URL at session start). **Do NOT run `git clone` into `.`** or you will get "destination path already exists" / non-empty directory errors. Start with `editor` `list_directory` / `read_file` (e.g. README) or `terminal` from the repo root.
- **NEVER describe or narrate actions without calling a tool.** If you say "I will create a file", you MUST immediately call the `editor` tool to do it.
- **NEVER write fake outputs.** Do not write "The tests passed" unless you actually ran tests via `terminal` using the **Python executable from Runtime context** and saw exit code 0 in the output.
- **NEVER say TASK COMPLETE unless you have used at least one tool** AND run verification (tests, a minimal run, or a targeted file/content check) with real tool output showing success.
- **Do NOT just write a plan as text and stop.** After your brief plan, immediately call the first tool.
- **CRITICAL — No full-file rewrites for edits**: Do NOT use `write_file` to rewrite an existing file just to make a small change. You MUST use `editor` `str_replace` or `apply_patch` for any edit to a file that already exists. Only use `write_file` when creating a **brand-new** file that does not yet exist.
- **CRITICAL — Plan before complex code**: Before writing any code for a task that spans more than one file or step, you MUST create or update `PLAN.md` at the workspace root outlining every step. Think before you act. Execute the plan one step at a time to avoid infinite loops and wasted iterations.

## Think → Act → Observe (required)
1. **Think** — Before tool calls, your assistant text must include a short rationale prefixed with **Think:** and tied to a step in workspace `PLAN.md` (e.g. STEP-2).
2. **Act** — Every `terminal` and `editor` tool call **must** include **`plan_step`** (string, e.g. `"STEP-2"`) matching the plan step you are executing.
3. **Observe** — Read each tool result fully (stdout/stderr, exit code, and **Observe (filesystem)**). On non-zero exit codes, diagnose and self-correct with a new tool call; do not give up after one failure.

## Anti-loop guard (strict)
- Do not run the exact same terminal command repeatedly without new input/state change.
- If a command returns the same output twice, switch strategy (read file, inspect config, check different signal) instead of retrying blindly.
- If blocked by auth/permissions/secrets, stop retrying and ask the user for the missing prerequisite.

## Workflow
1. **Brief plan** (2-3 lines max): State what you will do; align with `PLAN.md` at the workspace root (created/updated for each task).
2. **ACT immediately**: Call the first tool right away — do not wait.
3. **Continue**: After each tool result, call the next tool needed.
4. **Verify once**: Use the cheapest proof that matches the task. For simple file/content tasks, one `editor read_file` or one focused `terminal` command is enough. Do not verify the same fact twice.
5. **TASK COMPLETE**: As soon as verification output confirms success (or you document why verification is N/A), stop and write TASK COMPLETE. Do not take extra tool calls after success.

## Autonomous Coding Agent Operating Policy
- **Task vs question**: If the user asks an informational/meta question, answer directly without editing files. Only enter tool-heavy coding mode when the user requests an action such as build, fix, edit, clone, run, test, open, deploy, commit, or check.
- **Plan before coding**: For coding work, create/read `PLAN.md` first, write a short plan, then execute the first concrete step. Keep the plan synced as work progresses.
- **Inspect repository structure**: Before editing, inspect the repo narrowly: current directory, `git status`/`git diff` when available, package/config files, and only the files implicated by the task.
- **Use terminal carefully**: Prefer narrow, non-destructive commands. Never run destructive commands (`rm -rf`, reset/checkout, forced deletes, migrations, deploys, pushes) unless explicitly requested or clearly safe for the task.
- **Minimize diffs**: Make the smallest code change that solves the problem. Prefer `str_replace`/`apply_patch` over rewriting whole files; preserve unrelated user changes.
- **Follow existing style**: Match the repo's conventions, naming, formatting, framework patterns, test style, and dependency choices. Do not introduce new architecture or packages unless necessary.
- **Risky edits need reasoning**: Before schema changes, auth/payment/security changes, migrations, deploys, dependency installs, destructive file operations, or broad refactors, explain why the edit is needed and what you will touch.
- **Verify after modifications**: After code edits, run the most relevant focused tests/build/typecheck/lint. If no automated test exists, run a targeted command or file/content check and say what was verified.
- **Retry intelligently**: If tests or commands fail, read the error, make a focused fix, and retry. Do not give up after the first failure unless blocked by missing secrets, permissions, or external services.

## High-Ticket Web Architect (premium product + data)
When the task is **marketing sites, SaaS, dashboards, or CRUD apps** in TypeScript/React—and the repo is or should be a **modern web stack**—operate as a **senior web architect** shipping **$2000+ tier** quality: hierarchy, rhythm, accessibility, performance, and trustworthy data—not generic tutorial UI.

- **Consistency**: Drive **color, spacing, and typography** from a **single source of truth**—prefer
  **Tailwind** `tailwind.config.*` (or shared CSS variables / design tokens). Avoid scattered magic values.
- **Component-driven order**: Build **layout shells** first → **atomic** UI (buttons, cards, inputs) →
  **pages/routes** that compose them. Do not dump an entire screen into one file.
- **Micro-interactions**: Add **hover**, **focus-visible** (keyboard), and short **transitions** on
  interactive elements so the UI feels intentional, not static.
- **Premium stack**: Prefer **shadcn/ui** (Radix-based), **Radix UI** primitives, and **Lucide** icons
  when the project stack supports them, instead of bespoke complex CSS.
- **State-driven UI**: For lists, dashboards, and forms, implement **loading**, **error**, and **empty**
  states—not only the happy path.
- **Live preview by default**: If you are building a local web app, do not stop at writing files. Install dependencies once if needed, start the dev server in the correct sandbox mode, detect the listening port, and attach **`live_preview`** so the app can be inspected while work is in progress.
- **Visual QA**: After substantive UI edits, use **`browser_playwright`** (or the session’s Playwright
  observe tool) to capture the relevant route and **check alignment, contrast, and obvious layout issues**
  before claiming the UI is done.
### Default stack (use unless the repo clearly forbids it)
- **Next.js 15 — App Router**: Server Components by default; add ``'use client'`` only where hooks/events/browser APIs are required. Use **``loading.tsx``**, **``error.tsx``**, and **Route Handlers** where appropriate. Prefer **Server Actions** or server-only fetch for mutations that touch the DB—never ship secrets to the client bundle.
- **Drizzle ORM**: Central schema (e.g. ``db/schema``), typed queries, **drizzle-kit** migrations; avoid raw SQL in UI layers; keep schema changes reviewable and incremental.
- **Supabase**: ``@supabase/supabase-js``; **anon key + RLS** on the client when data is user-scoped; **service role only server-side** (Route Handlers / server actions / edge)—never commit keys or paste service role into client code. Design tables and policies as if production RLS is on.
- **UI — shadcn/ui + Magic UI**: **shadcn/ui** (Radix + Tailwind) for app shell, forms, tables, dialogs. **Magic UI** ([magicui.design](https://magicui.design)) for **marketing polish**—animated heroes, bento grids, marquee, border beams—**composed** on top of shadcn primitives, not duplicate design systems. **Lucide** icons. One Tailwind token source (``tailwind.config`` / CSS variables).

### Premium bar (non-negotiables)
- **Consistency**: Color, spacing, type scale, radius, and shadows from **tokens**—no one-off hex/radius litter.
- **Build order**: Layout shells → atomic components → pages/routes; **no monolithic 800-line screens**.
- **Motion**: Purposeful **hover**, **focus-visible**, short **transitions**; respect **prefers-reduced-motion**.
- **State-driven UI**: **Loading**, **error**, and **empty** states for every data view and form.
- **Visual QA**: After substantive UI work, **`browser_playwright`** on the real route—alignment, contrast, obvious responsive breaks—before claiming done.

### Git-aware context (save tokens; stay accurate)
When ``.git`` exists in the workspace:
- **Start narrow**: Use ``git`` **status** / **diff** (or terminal ``git diff --stat``, ``git log -3 --oneline``) to see **what changed** and **which files matter**—do **not** re-read the whole repo or paste huge directory trees into reasoning.
- **Read selectively**: Open **only** files implicated by the task, diff, or error traces; use ``editor`` **search** to jump to symbols instead of bulk ``read_file`` across unrelated modules.
- **Edit efficiently**: Prefer ``str_replace`` / ``apply_patch`` over full ``write_file`` when the file already exists and the change is local.
- **Summarize tool output**: Long stdout listings—extract paths and errors; do not echo entire trees back in assistant text.

### Clean architecture
- **Boundaries**: UI (``app/``, ``components/``) → **server** data loaders / actions → **Drizzle** in a dedicated layer (e.g. ``lib/db``, ``server/db``)—no DB clients or ORM calls scattered in presentational components.
- **Validation**: **Zod** (or equivalent) at system boundaries (forms, webhooks, route bodies).
- **Features at scale**: **Feature folders** (e.g. ``features/inventory/``) with colocated components + server helpers when the app grows; keep shared UI in ``components/ui``.
- **Env**: Document required vars in ``.env.example``; never commit real secrets.

### Other stacks
For **React/Vite/Next without** the full stack above, still apply **Tailwind tokens**, **shadcn where supported**, state-driven UI, and Git-aware reading—adapt file paths to the repo’s conventions.

## Browser Strategy
- **Prefer persistent browser tools for real interaction**: Use **`browser_navigate`**, **`browser_click`**, **`browser_type`**, **`browser_scroll`**, and **`browser_screenshot`** when you need a login/session/cookie state that persists across steps.
- **Observe before acting**: Start with **`browser_navigate`** or **`browser_screenshot`**, then read the returned screenshot, accessibility tree, console lines, and interactive element list before the next browser action.
- **Selector first, coordinates second**: Prefer stable CSS selectors for **`browser_click`** / **`browser_type`**. Only fall back to raw `x`/`y` clicks when the element cannot be targeted reliably by selector and the screenshot / accessibility data makes the target unambiguous.
- **Keep the loop tight**: After each browser action, inspect the new OBSERVATION. Do not issue a chain of blind clicks or types without checking what changed on the page.
- **Forms and login flows**: Fill one field at a time, then re-check the page if the UI reacts. Use **`clear_first`** when replacing existing values, and use **`submit: true`** on **`browser_type`** when pressing Enter is the natural submit action for a form.
- **Modal / overlay handling**: If a click is blocked or the target disappears, assume a modal, cookie banner, or overlay may be intercepting input. Capture a fresh observation, look for close/accept/dismiss controls first, and use **`browser_playwright`** when you need Escape, JS inspection, or deeper debugging.
- **Multi-step navigation**: After submit / next / continue actions, verify the new URL, title, and visible interactive elements before proceeding. Do not assume navigation succeeded just because a click completed.
- **Use the right tool for the job**: Use **`browser_playwright`** for richer DOM/debug tasks (evaluate JS, DOM outline, network/page-error debugging). Use the advanced browser tools for human-like step-by-step interaction on persistent pages.
- **For apps you are building, prefer live preview first**: Once the local dev server is running, call **`live_preview`** to probe likely ports and set the active one before doing browser QA. Use browser snapshots after the preview is attached or when the app is not running yet.
- **Recover methodically**: If a browser action fails, capture a fresh **`browser_screenshot`** or **`browser_playwright`** snapshot, then retry with a more specific selector, a wait, or a corrected target. Do not repeat the exact same failing click.

## Tool Usage
- `terminal` — Run shell commands; prefer `python -m pip`, `python -m pytest`, `python -m unittest` per Runtime context. For long installs (`npm install`, `npx create-*`) pass **`timeout_seconds`** up to **300** (default 30 is too short).
- `editor` with `read_file` — Read a file
- `editor` with `write_file` — Write/create a file
- `editor` with `search` — Search patterns in files
- `editor` with `list_directory` — List directory contents
- `editor` with `str_replace` — Replace exact text in a file (token-efficient, preferred for edits)
- `editor` with `apply_patch` — Apply a unified diff patch
- `browser_search` — Search the web
- `browser_fetch` — Fetch a web page (HTTP; no JS console)
- `browser_playwright` — **Preferred for UI bugs / DOM debugging**: real Chromium, DOM outline, console + network + page errors, screenshots, JS evaluation
- `browser_navigate` — Open a URL in the persistent Chromium session; use this to begin a multi-step browser task
- `browser_click` — Human-like click with selector-first targeting; use `x`/`y` only as a fallback after inspecting the screenshot / accessibility tree
- `browser_type` — Human-like typing into a selector in the persistent Chromium session; supports `clear_first` and `submit` for form workflows
- `browser_scroll` — Gradual viewport scroll in the persistent Chromium session
- `browser_screenshot` — Refresh browser state for the LLM: screenshot + accessibility tree + interactive element map
- `browser_interactive` — Legacy Selenium (use Playwright when available)
- `live_preview` — For apps you are building locally: probe localhost ports after `npm run dev` / framework dev server startup, then attach the active port so the Browser pane shows the running app. Common fallback ports include `3001`, `3002`, `4173`, `5001`, `5002`, `5173`, and `8000`.
- `github` — GitHub: branches, commits, PRs, PR status (CI), merge (see GitHub section below when token is set)
- `monitor` — Check app health, fetch cloud/docker logs, register for continuous monitoring
- `env_parity` — Generate Dockerfile/.env.example/docker-compose; diff local vs production env
- `memory` — **Persistent long-term memory across ALL sessions.** Actions: `save` (key + value), `read` (returns every stored entry), `delete` (key). Store coding style, preferences, recurring project facts here so they survive restarts.
- `deploy_vercel` — Deploy a project to Vercel (requires `VERCEL_TOKEN` in env). Pass `working_directory` pointing at the project root. Returns the live production URL on success.
{github_note}
## Long-Term Memory Usage
- **ALWAYS call `memory` with `action: read` at the very beginning of every task** to load any stored user preferences or project notes before doing any work.
- When the user says "remember this", "save this for later", or "always do X", call `memory` with `action: save`, choose a short descriptive `key`, and store the preference as the `value`.
- Surface remembered preferences naturally — e.g. "I remember you prefer tabs over spaces" — when they are relevant to the current task.
- Never invent memories; only store what the user explicitly asked to persist.

## Deployment (Vercel)
- If the user asks to deploy, make live, or publish a frontend app, use the `deploy_vercel` tool with `working_directory` set to the project root.
- If `VERCEL_TOKEN` is not set, tell the user and instruct them to add it to the environment.
- After a successful deploy, always report the live URL prominently.

## Important Rules
- Read `PLAN.md` at the repo root when starting; keep steps and `plan_step` references in sync.
- Read a file before editing it (to avoid overwriting changes)
- Always write COMPLETE file content when using `write_file`
- After writing a file, verify with `read_file`
- For browser work, prefer an **observe → act → observe** loop instead of multiple blind interactions in a row.
- For local website/app tasks, after the dev server is running, attempt **`live_preview`** attachment before declaring the UI ready.
- If **auto-verify (ruff)** reports issues on a `.py` file, you may set **`apply_ruff_fix`: true** on the next `editor` `write_file` / `str_replace` / `apply_patch` for that file to run **`ruff check --fix`** after the edit, or run the same via `terminal`.
- If a command fails, read the error and fix it — do NOT give up
- When done: write **TASK COMPLETE** followed by a short summary of actual results.

## Environment (critical)
{os_env_note}
- Use **relative paths only** in all tool calls — never hardcoded absolute paths.
- Prefer **`editor` `write_file`** to create files; it creates parent directories automatically.
- When `terminal` commands fail, fix the command and retry — do NOT give up after one attempt."""

SYSTEM_PROMPT = _make_system_prompt()


async def _git_current_branch(cwd: str) -> str | None:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    name = (out or b"").decode(errors="replace").strip()
    return name or None


async def _git_origin_default_branch(cwd: str) -> str | None:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "symbolic-ref",
        "--quiet",
        "--short",
        "refs/remotes/origin/HEAD",
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    if proc.returncode == 0 and out:
        full = out.decode(errors="replace").strip()
        if "/" in full:
            return full.rsplit("/", 1)[-1]
        return full or None
    proc2 = await asyncio.create_subprocess_exec(
        "git",
        "remote",
        "show",
        "origin",
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out2, _ = await proc2.communicate()
    if proc2.returncode != 0 or not out2:
        return None
    for line in out2.decode(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("HEAD branch:"):
            return line.split(":", 1)[1].strip() or None
    return None


class Agent:
    """
    The main agent that orchestrates task execution.

    The agent follows a state machine with these phases:
    - INTAKE: Receive and understand the task
    - EXPLORE: Explore the codebase to understand context
    - PLAN: Create an execution plan
    - EXECUTE: Execute plan steps using tools
    - VERIFY: Verify changes work correctly
    - REPAIR: Fix issues if verification fails
    - COMPLETE: Task completed successfully
    - BLOCKED: Waiting for user input
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        observation_llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        working_directory: str | None = None,
        max_iterations: int = _DEFAULT_AGENT_MAX_ITERATIONS,
        verbose: bool = True,
        auto_verify: bool = True,
        max_repair_iterations: int = 3,
        safety_policy: SafetyPolicy | None = None,
        artifact_dir: str = "runs",
        enable_parallel_execution: bool = True,
        max_parallel_tools: int = 5,
        callbacks: dict[str, Any] | None = None,
        use_sandbox: bool = False,
        auto_git_commit: bool = False,
        git_push: bool = False,
        session_id: str | None = None,
    ):
        self.llm = llm_client or create_llm_client()
        self._observation_llm_override = observation_llm_client
        self._observation_llm: LLMClient | None = None
        self._context_focus_path: str | None = None
        # Tool instances carry session-scoped state such as working_directory and
        # bridge callbacks, so each Agent needs an isolated registry. Reusing the
        # global registry made new sessions write into an older session workspace.
        self.registry = tool_registry or ToolRegistry()
        self.working_directory = working_directory
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.auto_verify = auto_verify
        self.max_repair_iterations = max_repair_iterations
        self.callbacks = callbacks or {}

        # Self-correction specific parameters
        self.max_immediate_retries = 3

        # Initialize state (session_id ties agent to DB session for local bridge, etc.)
        sid = session_id if session_id else str(uuid.uuid4())
        self.state = AgentState(
            agent_id=str(uuid.uuid4()),
            session_id=sid,
            phase=AgentPhase.INTAKE,
        )

        # Verification components (lazy initialized)
        self._verification_runner = None
        self._git_manager = None
        self._repair_loop = None

        # Safety guard (ensure policy iteration ceiling is not below the agent limit)
        self.safety_guard = SafetyGuard(safety_policy or SafetyPolicy())
        if self.safety_guard.policy.max_iterations_per_task < self.max_iterations:
            self.safety_guard.policy.max_iterations_per_task = self.max_iterations

        # Artifact logger (lazy initialized per task)
        self.artifact_dir = artifact_dir
        self._artifact_logger: ArtifactLogger | None = None

        # Memory system (Phase 4)
        self._symbol_index: SymbolIndex | None = None
        self._vector_store: VectorStore | None = None
        self._retrieval_manager: RetrievalManager | None = None
        self._working_memory: WorkingMemory | None = None
        self._memory_indexed = False

        # Reviewer agent (Phase 9A)
        self._reviewer_agent: ReviewerAgent | None = None

        # Planner agent (Phase 9B)
        self._planner_agent: PlannerAgent | None = None

        # Conversation memory for cross-session learning (Phase 18)
        self._conversation_memory: ConversationMemory | None = None
        self._use_conversation_memory: bool = True

        # Parallel execution (Phase 19)
        self._enable_parallel_execution = enable_parallel_execution
        self._max_parallel_tools = max_parallel_tools
        self._parallel_executor: ParallelExecutor | None = None
        self._batch_caller: BatchToolCaller | None = None

        # Auto git commit after task completion
        self.auto_git_commit = auto_git_commit
        self.git_push = git_push

        # Self-correction (Phase 24)
        self._correction_engine = SelfCorrectionEngine(
            max_immediate_retries=self.max_immediate_retries
        )
        self._consecutive_failures = 0
        # Last failed tool call before a self-correction retry (tool_name, args, stderr_snippet)
        self._last_failed_tool: tuple[str, dict[str, Any], str] | None = None
        # Short lines appended when self-correction succeeds (fed into save_task_summary)
        self._session_self_correct_lessons: list[str] = []

        # Docker Sandbox (optional isolated execution)
        self.use_sandbox = use_sandbox
        self._sandbox = None  # Initialized lazily per task
        self._pex_terminal: Any | None = (
            None  # ProcessExecutionSandbox for host-bash execution
        )
        # Last (prompt, completion) totals when we wrote think/observe — for per-event deltas in JSONL
        self._last_event_llm_tokens: tuple[int, int] | None = None
        # Same tool + identical error text repeated (forces workspace snapshot injection)
        self._same_error_streak_tool: str | None = None
        self._same_error_streak_fp: str | None = None
        self._same_error_streak_n: int = 0
        self._no_progress_inspection_streak: int = 0
        # How many times the progress guard fired without real progress (escalating hard-stop)
        self._progress_guard_fired_count: int = 0
        self._progress_guard_force_stop: bool = False
        # Observation-style post-mortem (forced diagnostic + recovery path)
        self._post_mortem_diagnostic_triggers: list[DiagnosticTriggerRecord] = []
        self._post_mortem_failure_streaks: list[FailureStreakRecord] = []
        self._post_mortem_recovery_paths: list[RecoveryPathRecord] = []
        self._pending_diagnostic_fp: str | None = None
        # OpenHands-style activity loop state (Action → Observation stream metadata)
        self._activity_state = AgentActivityState()
        self._current_activity_context: dict[str, Any] | None = None
        self._workspace_sidecar: WorkspaceSidecar | None = None
        self._last_terminal_command: str | None = None
        self._governance_telemetry_enabled = governance_telemetry_enabled()
        self._governance_emit_budget_signals = governance_budget_signals_enabled()
        self._governance_emit_retry_signals = governance_retry_signals_enabled()
        self._governance_emit_loop_signals = governance_loop_signals_enabled()
        self._governance_ops_collector: FileOpsTelemetryCollector | None = None
        try:
            _raw_token_budget = int(
                os.environ.get("PLODDER_GOVERNANCE_TOKEN_BUDGET", "0") or "0"
            )
        except ValueError:
            _raw_token_budget = 0
        self._governance_token_budget = max(0, _raw_token_budget)

        # EventStream / AgentController (optional; enabled for :meth:`run_simple`)
        self._backbone_stream: Any = None
        self._backbone_runtime: Any = None
        self._backbone_controller: Any = None
        self._use_backbone_dispatch: bool = False

        # Proactive clarification support
        self._clarification_event: asyncio.Event | None = None
        self._clarification_answer: str | None = None

        # Register default tools
        self._register_default_tools()

        # Set system prompt (inject workspace so models stop inventing Windows paths)
        wd = self.working_directory or os.getcwd()
        _spec_key = (
            (os.environ.get("AGENT_SPECIALIZATION") or "default").strip().lower()
        )
        _spec_extra = ""
        try:
            from ..agents.specialized_prompts import get_specialization_system_suffix

            _spec_extra = get_specialization_system_suffix(_spec_key)
        except Exception:
            pass
        _sys = SYSTEM_PROMPT + f"\n\n**Workspace root (authoritative):** `{wd}`\n"
        if _spec_extra:
            _sys = _sys + "\n" + _spec_extra + "\n"
        try:
            from .tool_function_registry import canonical_tools_prompt_block

            _sys = _sys + "\n\n" + canonical_tools_prompt_block()
        except Exception:
            pass
        self._system_prompt_base = _sys
        self.llm.set_system_prompt(_sys)

    def set_primary_llm_model(self, model: str) -> None:
        """
        Swap the primary :class:`LLMClient` (e.g. user picked another model in the UI).

        Keeps the in-memory conversation so chat context continues; ``create_llm_client``
        resolves ``auto`` the same way as session creation.
        """
        new_llm = create_llm_client(model=model)
        new_llm.conversation = list(self.llm.conversation)
        self.llm = new_llm
        if self._artifact_logger is not None:
            try:
                self._artifact_logger.set_model(self.llm.config.model)
            except Exception:
                pass

    def _get_observation_llm(self) -> LLMClient:
        """Gemini Flash (default) for observation/diagnostics; falls back to primary LLM if misconfigured."""
        if self._observation_llm_override is not None:
            return self._observation_llm_override
        if self._observation_llm is None:
            try:
                om = (
                    os.environ.get("LLM_MODEL_OBSERVATION") or "gemini/gemini-2.0-flash"
                ).strip()
                if not om:
                    om = "gemini/gemini-2.0-flash"
                temp = float(
                    (os.environ.get("LLM_OBSERVATION_TEMPERATURE") or "0.1").strip()
                    or 0.1
                )
                self._observation_llm = create_llm_client(model=om, temperature=temp)
            except ValueError:
                self._observation_llm = self.llm
        return self._observation_llm

    @staticmethod
    def _head_tail_truncate(text: str, cap: int, note: str) -> str:
        """Bound ``text`` to ``cap`` chars while keeping BOTH the head and the tail
        (signatures / exports often live near the end of a file), with a single
        elision marker in the middle. Head-only truncation loses the file's tail.
        """
        if cap <= 0 or len(text) <= cap:
            return text
        head = int(cap * 0.7)
        tail = cap - head
        omitted = len(text) - cap
        return (
            text[:head]
            + f"\n\n...({note}; {omitted} chars omitted)...\n\n"
            + text[-tail:]
        )

    @staticmethod
    def _compact_task_description_for_prompt(description: str) -> str:
        """Reduce very large user task prompts for LLM input while preserving intent.

        This only affects prompt rendering; the original task text stays unchanged in
        state/database for auditability.
        """
        raw = (description or "").strip()
        if not raw:
            return ""

        compact_on = os.environ.get(
            "PLODDER_TASK_PROMPT_COMPACT", "true"
        ).strip().lower() not in ("0", "false", "no")
        if not compact_on:
            return raw

        max_chars = int(
            (os.environ.get("PLODDER_TASK_PROMPT_COMPACT_MAX_CHARS") or "1200").strip()
            or "1200"
        )
        if len(raw) <= max_chars:
            return raw

        lines = [
            re.sub(r"\s+", " ", ln).strip()
            for ln in raw.splitlines()
            if ln and ln.strip()
        ]
        lead: list[str] = []
        structured: list[str] = []
        marker_re = re.compile(r"^(?:\d+[\).:]|[-*])\s+")

        for ln in lines:
            low = ln.lower()
            is_structured = bool(marker_re.match(ln)) or any(
                key in low
                for key in (
                    "requirement",
                    "success criteria",
                    "constraint",
                    "must",
                    "do not",
                )
            )
            if is_structured:
                structured.append(marker_re.sub("", ln).strip())
            elif len(lead) < 3:
                lead.append(ln)

        if not lead:
            lead.append(re.sub(r"\s+", " ", raw).strip())

        key_items = [s for s in structured if s][:8]
        compact_parts: list[str] = [lead[0]]
        if key_items:
            compact_parts.append("\nKey constraints:")
            compact_parts.extend(f"- {item}" for item in key_items)

        compact = "\n".join(compact_parts).strip()
        if len(compact) > max_chars:
            compact = Agent._head_tail_truncate(
                compact, max_chars, "task prompt compacted"
            )
        return (
            compact
            + "\n\n_(auto-compacted for token efficiency; full task kept in session data.)_"
        )

    @staticmethod
    def _compact_acceptance_criteria(criteria: list[str] | None) -> list[str]:
        """Limit acceptance criteria verbosity for prompt efficiency."""
        items = criteria or []
        max_items = int(
            (os.environ.get("PLODDER_TASK_CRITERIA_MAX_ITEMS") or "8").strip() or "8"
        )
        max_chars = int(
            (os.environ.get("PLODDER_TASK_CRITERION_MAX_CHARS") or "220").strip()
            or "220"
        )

        out: list[str] = []
        for c in items:
            txt = re.sub(r"\s+", " ", str(c or "")).strip()
            if not txt:
                continue
            if len(txt) > max_chars:
                txt = txt[: max_chars - 1] + "…"
            out.append(txt)
            if len(out) >= max_items:
                break
        return out

    def _ephemeral_dedup_text(self, key: str, raw: str, head_chars: int) -> str:
        """Token-saving: when ``raw`` is unchanged from the previous model call, return a
        compact head + pointer instead of the full body. The agent can still
        ``editor read_file`` for the complete content on demand. Disable via
        ``PLODDER_EPHEMERAL_DEDUP=false``.
        """
        dedup_on = os.environ.get(
            "PLODDER_EPHEMERAL_DEDUP", "true"
        ).strip().lower() not in ("0", "false", "no")
        last = getattr(self, "_ephemeral_last_hashes", None)
        if last is None:
            last = {}
            self._ephemeral_last_hashes = last
        import hashlib

        digest = hashlib.md5(raw.encode("utf-8", "replace")).hexdigest()
        if dedup_on and last.get(key) == digest and len(raw) > head_chars:
            return (
                raw[:head_chars]
                + "\n…(unchanged since last turn — full content omitted to save tokens; "
                + "use `editor read_file` if you need the rest)…"
            )
        last[key] = digest
        return raw

    def _workspace_context_ephemeral_messages(self) -> list[dict[str, Any]]:
        """Inject PLAN.md + current focus file into every completion (token-aware)."""
        chunks: list[str] = []
        wd = self.working_directory
        if not wd:
            return []
        root = Path(wd).resolve()
        plan = root / "PLAN.md"
        if plan.is_file():
            try:
                cap_p = int(
                    (os.environ.get("PLODDER_PLAN_MD_MAX_CHARS") or "8000").strip()
                    or "8000"
                )
                pt = plan.read_text(encoding="utf-8", errors="replace")
                pt = self._head_tail_truncate(
                    pt, cap_p, "PLAN.md middle elided for prompt"
                )
                pt = self._ephemeral_dedup_text("plan", pt, 1500)
                chunks.append(
                    "## PLAN.md (workspace root)\n```markdown\n" + pt + "\n```"
                )
            except OSError:
                pass

        rel = (self._context_focus_path or "").strip().replace("\\", "/")
        if rel:
            candidate = (root / rel).resolve()
            under_root = False
            try:
                candidate.relative_to(root)
                under_root = True
            except ValueError:
                under_root = False
            if under_root and candidate.is_file():
                mode = (
                    (os.environ.get("PLODDER_FOCUS_FILE_MODE") or "eager")
                    .strip()
                    .lower()
                )
                if mode == "lazy":
                    # Cursor-style on-demand: don't inline the body; the agent reads it
                    # via `editor read_file` only when it actually needs the content.
                    chunks.append(
                        f"## Current focus file: `{rel}`\n"
                        "_(not inlined to save tokens — use `editor read_file` to view its contents.)_"
                    )
                else:
                    try:
                        cap_f = int(
                            (
                                os.environ.get("PLODDER_CONTEXT_FILE_MAX_CHARS")
                                or "16000"
                            ).strip()
                            or "16000"
                        )
                        body = candidate.read_text(encoding="utf-8", errors="replace")
                        body = self._head_tail_truncate(
                            body, cap_f, "middle elided for prompt"
                        )
                        body = self._ephemeral_dedup_text(f"focus:{rel}", body, 2000)
                        chunks.append(
                            f"## Current focus file: `{rel}`\n```\n{body}\n```"
                        )
                    except OSError:
                        pass

        try:
            cap_w = int(
                (
                    os.environ.get("PLODDER_WORKING_MEMORY_PROMPT_TOKENS") or "1200"
                ).strip()
                or "1200"
            )
            wm = self.get_memory_context(max_tokens=cap_w)
            if wm.strip():
                chunks.append("## Working memory snapshot\n```text\n" + wm + "\n```")
        except Exception:
            pass

        if self._use_conversation_memory:
            try:
                lesson_limit = int(
                    (os.environ.get("PLODDER_RECENT_LESSONS_LIMIT") or "5").strip()
                    or "5"
                )
                lessons = [
                    lesson.strip()
                    for lesson in self.get_recent_lessons(limit=lesson_limit)
                    if lesson.strip()
                ]
                if lessons:
                    lesson_block = "\n".join(f"- {lesson[:360]}" for lesson in lessons)
                    chunks.append("## Recent lessons learned\n" + lesson_block)
            except Exception:
                pass

        if not chunks:
            return []
        return [
            {
                "role": "user",
                "content": "\n\n".join(chunks)
                + "\n\n_(Workspace context — refreshed each model call.)_",
            }
        ]

    def _seed_task_working_memory(
        self, task: TaskState, resumed_plan: list[str] | None = None
    ) -> None:
        """Load the live task goal, constraints, and any resumed plan into working memory."""
        try:
            goal = (task.goal.description or "").strip()
            if goal:
                self.add_to_memory(goal, item_type="goal", priority="critical")

            criteria = [
                c.strip()
                for c in (task.goal.acceptance_criteria or [])
                if c and c.strip()
            ]
            if criteria:
                self.add_to_memory(
                    "Acceptance criteria:\n"
                    + "\n".join(f"- {c}" for c in criteria[:12]),
                    item_type="constraint",
                    priority="high",
                )

            constraint_bits: list[str] = []
            if task.constraints.require_tests:
                constraint_bits.append("tests required")
            if task.constraints.require_lint:
                constraint_bits.append("lint required")
            if task.constraints.allowed_directories:
                constraint_bits.append(
                    "allowed directories: "
                    + ", ".join(task.constraints.allowed_directories[:10])
                )
            if constraint_bits:
                self.add_to_memory(
                    "Task constraints: " + "; ".join(constraint_bits),
                    item_type="constraint",
                    priority="high",
                )

            if self.working_directory:
                self.add_to_memory(
                    f"Workspace root: {self.working_directory}",
                    item_type="context",
                    priority="medium",
                )

            if resumed_plan:
                self.add_to_memory(
                    "Resumed plan steps:\n"
                    + "\n".join(f"- {step}" for step in resumed_plan[:20]),
                    item_type="plan",
                    priority="high",
                )
        except Exception as e:
            self._log(f"Warning: failed to seed working memory: {e}")

    def _build_task_retrieval_context(self, task: TaskState) -> str:
        """Collect a compact retrieval block from code search and long-term memory."""
        parts: list[str] = []
        queries = self._task_retrieval_queries(task)
        if not queries:
            return ""

        try:
            if self.working_directory and not self._memory_indexed:
                self.index_workspace()
            code_blocks: list[str] = []
            for label, query in queries[:3]:
                hits = self.search_code(query, max_results=3)
                if not hits:
                    continue
                lines: list[str] = []
                for hit in hits[:3]:
                    if hasattr(hit, "to_dict"):
                        data = hit.to_dict()
                        file_path = (
                            data.get("location", {}).get("file_path")
                            or data.get("file_path")
                            or ""
                        )
                        name = (
                            data.get("name")
                            or data.get("symbol_name")
                            or data.get("title")
                            or ""
                        )
                        kind = data.get("symbol_type") or data.get("type") or "symbol"
                        detail = (
                            data.get("docstring")
                            or data.get("summary")
                            or data.get("content")
                            or ""
                        )
                        preview = str(detail).replace("\n", " ")[:260]
                        loc = file_path or data.get("path") or ""
                        lines.append(f"- [{kind}] {name} {loc}: {preview}".strip())
                    else:
                        text = str(hit)
                        lines.append(f"- {text[:320]}")
                code_blocks.append(f"{label}:\n" + "\n".join(lines))
            if code_blocks:
                parts.append(
                    "Relevant code search results:\n" + "\n\n".join(code_blocks)
                )
        except Exception as e:
            self._log(f"Code retrieval skipped: {e}")

        if self._use_conversation_memory:
            try:
                mem_block = self.get_context_from_memory(
                    "\n".join(q for _, q in queries[:2])
                )
                if mem_block.strip():
                    parts.append("Relevant past memory:\n" + mem_block)
            except Exception as e:
                self._log(f"Memory retrieval skipped: {e}")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _task_retrieval_queries(task: TaskState) -> list[tuple[str, str]]:
        """Return prioritized retrieval queries tailored to the task type."""
        description = (task.goal.description or "").strip()
        criteria = " ".join(
            c.strip() for c in (task.goal.acceptance_criteria or []) if c and c.strip()
        )
        combined = " ".join(part for part in (description, criteria) if part)
        ttype = getattr(task, "task_type", None)
        task_type = getattr(ttype, "value", str(ttype or "general")).lower()

        hint_map: dict[str, list[str]] = {
            "bug_fix": [
                "bug fix error traceback failing test regression",
                "debug failure root cause",
            ],
            "feature": [
                "feature implementation endpoint route component api",
                "new functionality",
            ],
            "refactor": [
                "refactor cleanup restructure simplify similar code",
                "shared helper extraction",
            ],
            "testing": [
                "test pytest unit integration assertion",
                "verification test coverage",
            ],
            "documentation": [
                "documentation readme comments guide explanation",
                "docs update",
            ],
            "code_review": ["review validate lint audit issues", "diff review"],
            "exploration": ["explore inspect understand symbols", "workspace overview"],
        }
        hints = hint_map.get(task_type, ["implementation verify"])

        queries: list[tuple[str, str]] = []
        if combined:
            queries.append(("task goal", combined))
        for hint in hints:
            if combined:
                queries.append((hint, f"{hint} {combined}".strip()))
            else:
                queries.append((hint, hint))
        return queries

    async def _emit_file_changed_ui(
        self, relative_path: str, file_content: str | None = None
    ) -> None:
        """Notify UI (WebSocket) with full file text after a successful workspace edit."""
        if "on_file_changed" not in self.callbacks:
            return
        rp = str(relative_path or "").strip().replace("\\", "/")
        if not rp:
            return
        text = file_content
        if text is None:
            if not self.working_directory:
                return
            root = Path(self.working_directory).resolve()
            candidate = (root / rp).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                return
            if not candidate.is_file():
                return
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return
        cap = int(os.environ.get("PLODDER_WS_FILE_CHANGED_MAX_CHARS", "500000"))
        if len(text) > cap:
            text = text[:cap] + "\n\n...(truncated for Plodder UI sync)\n"
        await self._trigger_callback("on_file_changed", rp, text)

    async def _prepare_messages_for_llm_turn(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from mini_devin.core.context_condenser import condense_chat_messages
        from mini_devin.core.llm_client import _openai_non_system_window_valid

        base = self.llm.get_conversation_for_api()
        if os.environ.get("LLM_CONTEXT_CONDENSER", "true").lower() not in (
            "0",
            "false",
            "no",
        ):
            try:
                condensed = await condense_chat_messages(
                    base,
                    summarizer=self._get_observation_llm(),
                )
                ns = [m for m in condensed if m.get("role") != "system"]
                if _openai_non_system_window_valid(ns):
                    base = condensed
                elif condensed is not base:
                    self._log(
                        "Context condenser returned OpenAI-unsafe message ordering; "
                        "using un-condensed history for this turn."
                    )
            except Exception as _ce:
                self._log(f"Context condenser skipped: {_ce}")
        ephemeral = self._workspace_context_ephemeral_messages()
        return base, ephemeral

    def _emit_terminal_line(self, line: str) -> None:
        cb = self.callbacks.get("on_command_output")
        if cb:
            cb(line)

    def _register_default_tools(self) -> None:
        """Register the default tools (terminal, editor, browser)."""
        # Only register if not already registered
        if not self.registry.get("terminal"):
            agent = self
            terminal = create_terminal_tool(
                working_directory=self.working_directory,
                bridge_session_id=self.state.session_id,
                on_output_line=lambda ln: agent._emit_terminal_line(ln),
            )
            self.registry.register(terminal)

        if not self.registry.get("editor"):
            editor = create_editor_tool(working_directory=self.working_directory)
            self.registry.register(editor)

        # Browser tools
        if not self.registry.get("browser_search"):
            browser_search = create_search_tool()
            self.registry.register(browser_search)

        if not self.registry.get("browser_fetch"):
            browser_fetch = create_fetch_tool()
            self.registry.register(browser_fetch)

        if not self.registry.get("browser_interactive"):
            browser_interactive = create_interactive_tool()
            self.registry.register(browser_interactive)

        if not self.registry.get("browser_playwright"):
            agent = self
            browser_pw = create_playwright_tool(
                on_browser_event=lambda p: agent._emit_playwright_browser_event(p),
            )
            self.registry.register(browser_pw)

        advanced_browser_names = (
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_scroll",
            "browser_screenshot",
        )
        if any(not self.registry.get(name) for name in advanced_browser_names):
            agent = self
            for browser_tool in create_advanced_browser_tools(
                on_browser_event=lambda p: agent._emit_playwright_browser_event(p),
            ):
                if not self.registry.get(browser_tool.name):
                    self.registry.register(browser_tool)

        # Citation store for tracking web references
        if not hasattr(self, "_citation_store"):
            self._citation_store = create_citation_store()

        # Git workflow tool
        if not self.registry.get("git"):
            from ..tools.git import create_git_tool

            git_tool = create_git_tool()
            self.registry.register(git_tool)

        # GitHub tool
        if not self.registry.get("github"):
            from ..tools.github import create_github_tool

            github_tool = create_github_tool()
            self.registry.register(github_tool)

        # Long-term memory tool
        if not self.registry.get("memory"):
            from ..tools.memory import create_memory_tool

            self.registry.register(create_memory_tool())

        # Vercel deploy tool
        if not self.registry.get("deploy_vercel"):
            from ..tools.deploy import create_deploy_tool

            self.registry.register(create_deploy_tool())

    def _ensure_workspace_sidecar(self) -> None:
        wd = self.working_directory
        if not wd or self._workspace_sidecar is not None:
            return
        self._workspace_sidecar = WorkspaceSidecar(wd)
        self._workspace_sidecar.start()

    def stop_workspace_sidecar(self) -> None:
        if self._workspace_sidecar:
            self._workspace_sidecar.stop()
            self._workspace_sidecar = None

    async def cleanup(self) -> None:
        """Release background resources (file watcher, etc.)."""
        self.stop_workspace_sidecar()
        seen_sessions: set[int] = set()
        for tool_name in (
            "browser_playwright",
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_scroll",
            "browser_screenshot",
        ):
            tool = self.registry.get(tool_name)
            if tool is None:
                continue
            session_obj = getattr(tool, "_session", tool)
            session_key = id(session_obj)
            if session_key in seen_sessions:
                continue
            seen_sessions.add(session_key)
            close_fn = getattr(tool, "close", None)
            if callable(close_fn):
                try:
                    await close_fn()
                except Exception as exc:
                    self._log(f"browser cleanup skipped for {tool_name}: {exc}")

    def bootstrap_ide_experience(self) -> None:
        """
        Emit greeting + live workspace index to the event stream (OpenHands-style IDE open).
        Call once after the session is bound to a workspace.
        """
        wd = self.working_directory
        if not wd:
            return
        self._ensure_workspace_sidecar()
        snap = ""
        if self._workspace_sidecar:
            snap = self._workspace_sidecar.get_snapshot_text(
                max_lines=600, max_chars=12_000
            )
        wl = load_worklog(wd)
        resume_note = ""
        if wl and wl.current_plan:
            resume_note = (
                f"\n\n**Saved progress**: {len(wl.current_plan)} plan step(s) on disk "
                f"(last task `{wl.last_task_id}`, cursor step index {wl.current_step_idx}). "
                "Starting the same task id will restore the plan."
            )
        greeting = (
            "Welcome — this workspace is wired for an autonomous coding loop. "
            "The tree below is refreshed by a background watcher; use **terminal** as your primary "
            "sensor (build, test, git) and **editor** for precise file work. Describe a task when ready."
        )
        preview_note = ""
        if self._should_use_process_execution_terminal():
            preview_note = (
                "\n\n**Browser Live Preview**: When you run a dev server (`npm run dev`, Vite, etc.) it "
                "usually listens on **127.0.0.1**. After it is listening, call the **live_preview** tool: "
                "`action: probe`, then **`set_active_port`** with one of the reported ports so the **Browser** "
                "tab iframe loads your app (same-origin proxy through this API)."
            )
        text = (
            greeting
            + preview_note
            + "\n\n## Workspace index (sidecar)\n\n```text\n"
            + snap
            + "\n```"
            + resume_note
        )
        self._append_session_event(
            AgentStreamEvent(
                kind=AgentEventKind.STATUS,
                role="agent",
                text=text[:12000],
                legacy_type="ide_bootstrap",
                meta={"bootstrap": True, "repo_root": str(Path(wd).resolve())},
            )
        )

    def _persist_worklog(self, task: TaskState) -> None:
        wd = self.working_directory
        if not wd:
            return
        finished: list[str] = []
        steps = getattr(self, "_plan_steps", None) or []
        idx = int(getattr(self, "_current_step_idx", 0))
        if getattr(self, "_plan_sent", False) and steps:
            finished = [steps[i] for i in range(min(idx, len(steps)))]
        log = SessionWorklog(
            session_id=self.state.session_id,
            last_task_id=task.task_id,
            current_plan=list(steps),
            finished_steps=finished,
            remaining_tasks=[],
            current_step_idx=idx,
        )
        save_worklog(wd, log)

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM function calling."""
        schemas = []

        # Terminal tool schema — OS-aware description
        import sys as _sys

        _on_windows = _sys.platform == "win32"
        _lp_terminal_tail = (
            " After a dev UI server is running in the background, call **live_preview** "
            "(`probe`, then `set_active_port`) so the **Browser** tab can show it."
        )
        if _on_windows:
            _terminal_desc = (
                "**Primary interface**: run shell commands first to observe the real world "
                "(directory listings, builds, tests, git). "
                "Execute a shell command via PowerShell on Windows. "
                "Common Linux commands are auto-translated (python3→python, ls, cat, mkdir -p, rm -rf, grep, touch, etc.). "
                "Use relative paths from the workspace. Do NOT use absolute Windows paths like C:\\\\. "
                "Prefer `python` over `python3`. For listings use `Get-ChildItem -Force` or plain `dir`; avoid `ls -la`, `ll`, `dir /a`, and `dir /b`. "
                "Chain commands with `;` instead of `&&` when possible."
            )
        else:
            _terminal_desc = (
                "**Primary interface**: run shell commands first to observe the real world "
                "(ls, git status, builds, tests). "
                "Execute a shell command in bash under the task workspace. "
                "Use relative paths (./...) from the workspace. "
                "Do not use Windows-style paths (C:\\\\, D:\\\\)."
            )
        if self._should_use_process_execution_terminal():
            _terminal_desc = _terminal_desc + _lp_terminal_tail
        schemas.append(
            {
                "name": "terminal",
                "description": _terminal_desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute",
                        },
                        "working_directory": {
                            "type": "string",
                            "description": "Working directory for the command (default: current directory)",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": (
                                "Max seconds before the command is killed (default 30, max 300). "
                                "Use 180–300 for `npm install`, `npx create-*`, `pip install -r`, builds, etc."
                            ),
                        },
                        "plan_step": {
                            "type": "string",
                            "description": 'Plan step id from workspace PLAN.md (e.g. "STEP-2"). Required for traceability.',
                        },
                    },
                    "required": ["command"],
                },
            }
        )

        # Editor tool schema (multi-action)
        schemas.append(
            {
                "name": "editor",
                "description": """Perform file operations. Supports multiple actions:
- read_file: Read a file's contents
- write_file: Write/create a file with full content
- str_replace: Replace an exact string in a file (PREFERRED for edits — token-efficient, precise)
- search: Search for patterns in files
- list_directory: List directory contents
- apply_patch: Apply a unified diff patch
- revert_file: Undo all uncommitted changes to a file and restore its last known good state (git checkout)

PREFER str_replace over write_file when editing existing files.

Optional **`apply_ruff_fix`**: set to true on `write_file` / `str_replace` / `apply_patch` for a `.py` file to run `ruff check --fix` on that path after a successful edit, then re-run `ruff check` (requires ruff config in the repo). Use when auto-verify reports ruff violations.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "read_file",
                                "write_file",
                                "str_replace",
                                "search",
                                "list_directory",
                                "apply_patch",
                                "revert_file",
                            ],
                            "description": "The action to perform",
                        },
                        "path": {
                            "type": "string",
                            "description": "File or directory path",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write (for write_file action)",
                        },
                        "old_str": {
                            "type": "string",
                            "description": "Exact string to find and replace (for str_replace action — must be unique in file)",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "Replacement string (for str_replace action — use empty string to delete)",
                        },
                        "allow_multiple": {
                            "type": "boolean",
                            "description": "Replace all occurrences (for str_replace action, default false)",
                        },
                        "plan_step": {
                            "type": "string",
                            "description": 'Plan step id from workspace PLAN.md (e.g. "STEP-2"). Required for traceability.',
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (for search action)",
                        },
                        "patch": {
                            "type": "string",
                            "description": "Unified diff patch (for apply_patch action)",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Start line for reading (for read_file action)",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "End line for reading (for read_file action)",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Recursive listing (for list_directory action)",
                        },
                        "apply_ruff_fix": {
                            "type": "boolean",
                            "description": (
                                "If true, after a successful write_file/str_replace/apply_patch on a .py file, "
                                "run `python -m ruff check --fix` on that file, then `ruff check` again. "
                                "Ignored for non-Python paths or if the edit failed."
                            ),
                        },
                    },
                    "required": ["action", "path"],
                },
            }
        )

        # Browser search tool schema
        schemas.append(
            {
                "name": "browser_open",
                "description": (
                    "Open a URL in the user's Plodder Browser tab. Use this when the user asks to open, show, "
                    "visit, or navigate to a public/local URL visibly. This does not require backend network access; "
                    "the frontend browser tab loads the URL for the user."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Absolute URL to open, e.g. https://example.com or http://localhost:3000",
                        },
                        "note": {
                            "type": "string",
                            "description": "Optional short note to show in the browser activity feed",
                        },
                    },
                    "required": ["url"],
                },
            }
        )

        schemas.append(
            {
                "name": "browser_search",
                "description": "Search the web for information. Use this to find documentation, solutions to errors, or research topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10)",
                        },
                    },
                    "required": ["query"],
                },
            }
        )

        # Browser fetch tool schema
        schemas.append(
            {
                "name": "browser_fetch",
                "description": "Fetch and read the content of a web page. Use this to read documentation, articles, or any web content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch",
                        },
                        "extract_content": {
                            "type": "boolean",
                            "description": "Extract clean text content (default: true)",
                        },
                    },
                    "required": ["url"],
                },
            }
        )

        # Browser interactive tool schema
        schemas.append(
            {
                "name": "browser_interactive",
                "description": "Interact with web pages that require JavaScript or complex interactions. Use for forms, login pages, or dynamic content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "navigate",
                                "click",
                                "type",
                                "scroll",
                                "screenshot",
                                "get_text",
                                "get_html",
                                "wait",
                            ],
                            "description": "The browser action to perform",
                        },
                        "url": {
                            "type": "string",
                            "description": "URL to navigate to (for navigate action)",
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for the element (for click, type, get_text actions)",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type (for type action)",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "top", "bottom"],
                            "description": "Scroll direction (for scroll action)",
                        },
                        "seconds": {
                            "type": "number",
                            "description": "Seconds to wait (for wait action)",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        schemas.append(
            {
                "name": "browser_playwright",
                "description": (
                    "Headless Chromium (Playwright). Use for JS sites, UI debugging, and before/after checks. "
                    "Returns DOM outline (headings + body structure), console lines, XHR/fetch/document network log, "
                    "uncaught page errors, failed requests, and optional screenshots. "
                    "Actions: navigate (url, wait_until: domcontentloaded|networkidle, full_page), debug_snapshot "
                    "(settle_ms ms wait then dump), screenshot, click (selector|x,y), type/fill, wait_for "
                    "(selector|url|network_idle), get_text, get_html, evaluate (script), find_elements, reload, etc. "
                    "Requires: pip install playwright && playwright install chromium, or BROWSERLESS_API_KEY."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "navigate",
                                "screenshot",
                                "click",
                                "type",
                                "fill",
                                "select",
                                "scroll",
                                "hover",
                                "wait_for",
                                "get_text",
                                "get_html",
                                "evaluate",
                                "highlight",
                                "find_elements",
                                "pdf",
                                "press",
                                "go_back",
                                "go_forward",
                                "reload",
                                "debug_snapshot",
                            ],
                            "description": "Playwright action to run",
                        },
                        "url": {
                            "type": "string",
                            "description": "For navigate / wait_for(url pattern)",
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for click, type, fill, etc.",
                        },
                        "text": {"type": "string", "description": "For type action"},
                        "value": {"type": "string", "description": "For fill/select"},
                        "script": {
                            "type": "string",
                            "description": "JavaScript for evaluate action",
                        },
                        "wait_until": {
                            "type": "string",
                            "description": "navigate load state: domcontentloaded | load | networkidle",
                        },
                        "full_page": {
                            "type": "boolean",
                            "description": "Full-page screenshot",
                        },
                        "network_idle": {
                            "type": "boolean",
                            "description": "For wait_for: wait until network idle",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout ms for waits",
                        },
                        "settle_ms": {
                            "type": "integer",
                            "description": "For debug_snapshot: extra wait before capture (default 400)",
                        },
                        "direction": {
                            "type": "string",
                            "description": "scroll: up|down|top|bottom",
                        },
                        "amount": {"type": "number", "description": "scroll pixels"},
                        "key": {
                            "type": "string",
                            "description": "press: key name e.g. Enter",
                        },
                        "delay": {
                            "type": "integer",
                            "description": "type: ms between keys",
                        },
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                    "required": ["action"],
                },
            }
        )

        schemas.append(
            {
                "name": "browser_navigate",
                "description": (
                    "Navigate the persistent Chromium session to a URL, preserving login/session state. "
                    "Use this to start a browser task, then inspect the returned screenshot, accessibility tree, "
                    "and interactive element map before choosing the next action."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Destination URL"},
                        "wait_until": {
                            "type": "string",
                            "description": "Playwright load state: domcontentloaded | load | networkidle",
                        },
                    },
                    "required": ["url"],
                },
            }
        )

        schemas.append(
            {
                "name": "browser_click",
                "description": (
                    "Click inside the persistent Chromium session using a CSS selector or x/y coordinates. "
                    "Prefer selectors; use x/y only when the target is visually unambiguous from the latest "
                    "screenshot/accessibility data. Uses human-like mouse movement and returns a fresh screenshot "
                    "plus accessibility tree."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "Preferred CSS selector target for the click",
                        },
                        "x": {"type": "number", "description": "Viewport x coordinate"},
                        "y": {"type": "number", "description": "Viewport y coordinate"},
                    },
                    "required": [],
                },
            }
        )

        schemas.append(
            {
                "name": "browser_type",
                "description": (
                    "Type text into a selector inside the persistent Chromium session using human-like "
                    "keystroke pacing. Supports optional Enter submit for forms. Use after inspecting the "
                    "latest browser observation and return a fresh screenshot plus accessibility tree."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for the input element",
                        },
                        "text": {"type": "string", "description": "Text to enter"},
                        "clear_first": {
                            "type": "boolean",
                            "description": "If true, clear the field before typing",
                        },
                        "submit": {
                            "type": "boolean",
                            "description": "If true, press Enter after typing to submit the current form",
                        },
                    },
                    "required": ["selector", "text"],
                },
            }
        )

        schemas.append(
            {
                "name": "browser_scroll",
                "description": (
                    "Scroll the persistent Chromium page with gradual wheel movement. Returns a fresh "
                    "screenshot plus accessibility tree so you can reassess what became visible."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "top", "bottom"],
                            "description": "Scroll direction",
                        },
                        "amount": {
                            "type": "integer",
                            "description": "Approximate scroll distance in pixels",
                        },
                    },
                    "required": ["direction"],
                },
            }
        )

        schemas.append(
            {
                "name": "browser_screenshot",
                "description": (
                    "Capture the current page state from the persistent Chromium session, including a "
                    "screenshot, accessibility tree, and interactive element map. Use this when you need a "
                    "fresh observation before clicking, typing, or retrying."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "full_page": {
                            "type": "boolean",
                            "description": "If true, capture a full-page screenshot",
                        },
                    },
                    "required": [],
                },
            }
        )

        # Ask-user tool for proactive clarification
        schemas.append(
            {
                "name": "ask_user",
                "description": (
                    "Ask the user a clarifying question when the task is ambiguous or blocked. "
                    "Optionally provide multiple-choice options so the user can click instead of type. "
                    "Use ONLY when you genuinely cannot proceed without human input."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The specific question to ask the user",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of choices to present as clickable buttons, e.g. "
                                "['Use PostgreSQL', 'Use SQLite', 'Let me decide later']. "
                                "When provided, the user clicks a button instead of typing."
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional additional context or explanation to show the user",
                        },
                    },
                    "required": ["question"],
                },
            }
        )

        schemas.append(
            {
                "name": "live_preview",
                "description": (
                    "Detect a local dev server (e.g. Vite on 5173) and register it for the Plodder **Browser** tab Live Preview. "
                    "Use **probe** after `npm run dev` / `vite` in terminal; then **set_active_port** with a listening port. "
                    "The UI reverse-proxies through the API so the iframe is same-origin. "
                    "HMR WebSockets may not tunnel through this proxy—reload still works when files change. "
                    "**Not for external sites**: Live Preview cannot open arbitrary public URLs (e.g. user says 'open example.com'). "
                    "Use **browser_playwright** or **browser_fetch** with the full https URL instead. "
                    "On hosted platforms, **PORT** (often 8080) is usually this API — probing it does not load a third-party domain. "
                    "**Host reachability**: The dev server must listen on **127.0.0.1** where this API process can see it "
                    "(set `USE_PROCESS_EXECUTION_SANDBOX=1` for a host-side terminal when needed). "
                    "A terminal confined to an isolated one-shot Docker exec typically cannot register those ports here without extra port forwarding."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["probe", "set_active_port"],
                            "description": "probe=scan ports on 127.0.0.1; set_active_port=register port for Browser iframe",
                        },
                        "ports": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "For probe: candidate ports (default from LIVE_PREVIEW_PROBE_PORTS env)",
                        },
                        "port": {
                            "type": "integer",
                            "description": "For set_active_port: must be listening and in LIVE_PREVIEW_ALLOWED_PORTS",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        # Project memory tool schema
        schemas.append(
            {
                "name": "project_memory",
                "description": (
                    "Long-term vector memory for a project. Store architecture decisions, "
                    "tech stack choices, constraints, and lessons learned that persist "
                    "across sessions. Search memory semantically to recall past decisions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "remember",
                                "search",
                                "list",
                                "get_context",
                                "create_project",
                                "list_projects",
                                "delete_entry",
                            ],
                            "description": (
                                "remember=store a new memory entry; "
                                "search=semantic search over project memory; "
                                "list=list all entries for a project; "
                                "get_context=get formatted context string for a task; "
                                "create_project=create a new project; "
                                "list_projects=show all projects; "
                                "delete_entry=remove a memory entry."
                            ),
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project identifier",
                        },
                        "project_name": {
                            "type": "string",
                            "description": "Human-readable project name (for create_project)",
                        },
                        "project_description": {
                            "type": "string",
                            "description": "Project description (for create_project)",
                        },
                        "tech_stack": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tech stack list (for create_project)",
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "architecture",
                                "decision",
                                "constraint",
                                "api_contract",
                                "lesson",
                                "milestone",
                                "user_preference",
                                "code_snippet",
                                "context",
                            ],
                            "description": "Memory category",
                        },
                        "title": {
                            "type": "string",
                            "description": "Short title for the memory entry",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full content of the memory entry",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for filtering",
                        },
                        "importance": {
                            "type": "integer",
                            "description": "Importance 1–10 (10=critical, default 5)",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query for 'search' and 'get_context' actions",
                        },
                        "entry_id": {
                            "type": "string",
                            "description": "Entry ID for delete_entry",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results for search (default 5)",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        # Project plan tool schema
        schemas.append(
            {
                "name": "project_plan",
                "description": (
                    "Hierarchical project planner. Decompose a large goal into milestones, "
                    "create a persistent plan, execute each milestone as an isolated sub-agent, "
                    "and resume across sessions. Perfect for month-long projects."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "create",
                                "get",
                                "list",
                                "execute",
                                "retry_milestone",
                                "delete",
                            ],
                            "description": (
                                "create=decompose goal into milestones (LLM-powered); "
                                "get=show plan status; "
                                "list=list all plans; "
                                "execute=run remaining milestones as sub-agents; "
                                "retry_milestone=reset a failed milestone; "
                                "delete=remove a plan."
                            ),
                        },
                        "plan_id": {
                            "type": "string",
                            "description": "Plan ID for get/execute/retry/delete",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project ID for create/list",
                        },
                        "goal": {
                            "type": "string",
                            "description": "High-level project goal for 'create'",
                        },
                        "milestones": {
                            "type": "array",
                            "description": "Manual milestone list (optional, skips LLM decomposition)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "acceptance_criteria": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "depends_on_indexes": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                    "estimated_hours": {"type": "number"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "milestone_id": {
                            "type": "string",
                            "description": "Milestone ID for retry_milestone",
                        },
                        "working_dir": {
                            "type": "string",
                            "description": "Working directory for execution",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        # UI Test tool schema
        schemas.append(
            {
                "name": "ui_test",
                "description": (
                    "Run a structured browser-based UI test suite against a live URL using Playwright. "
                    "Each step can assert elements exist, check text, click buttons, fill forms, "
                    "verify URL changes, check for JS errors, or capture screenshots for visual regression. "
                    "Returns a full pass/fail report per step."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "suite_name": {
                            "type": "string",
                            "description": "Name for this test suite",
                        },
                        "url": {
                            "type": "string",
                            "description": "Starting URL to test",
                        },
                        "steps": {
                            "type": "array",
                            "description": "List of test steps to execute",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "navigate",
                                            "assert_element",
                                            "assert_text",
                                            "assert_url",
                                            "assert_title",
                                            "click",
                                            "fill",
                                            "select",
                                            "press_key",
                                            "wait",
                                            "wait_for_selector",
                                            "screenshot",
                                            "evaluate",
                                            "assert_no_js_errors",
                                            "hover",
                                            "scroll",
                                        ],
                                        "description": "Step type",
                                    },
                                    "selector": {
                                        "type": "string",
                                        "description": "CSS selector",
                                    },
                                    "url": {
                                        "type": "string",
                                        "description": "URL for navigate/assert_url steps",
                                    },
                                    "text": {
                                        "type": "string",
                                        "description": "Expected text for assert_text/assert_title",
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "Value for fill/select/press_key",
                                    },
                                    "script": {
                                        "type": "string",
                                        "description": "JavaScript for evaluate step",
                                    },
                                    "expected": {
                                        "description": "Expected JS return value for evaluate"
                                    },
                                    "screenshot_name": {
                                        "type": "string",
                                        "description": "Name for screenshot baseline",
                                    },
                                    "set_baseline": {
                                        "type": "boolean",
                                        "description": "Set this screenshot as new baseline",
                                    },
                                    "threshold_percent": {
                                        "type": "number",
                                        "description": "Visual diff threshold (default 0.5%)",
                                    },
                                    "ms": {
                                        "type": "integer",
                                        "description": "Milliseconds for wait step",
                                    },
                                    "timeout_ms": {
                                        "type": "integer",
                                        "description": "Timeout in ms (default 10000)",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Human-readable step description",
                                    },
                                },
                                "required": ["type"],
                            },
                        },
                        "threshold_percent": {
                            "type": "number",
                            "description": "Default visual regression threshold (default 0.5%)",
                        },
                    },
                    "required": ["suite_name", "url", "steps"],
                },
            }
        )

        # Visual regression tool schema
        schemas.append(
            {
                "name": "visual_regression",
                "description": (
                    "Manage screenshot baselines and run pixel-diff comparisons for visual regression testing. "
                    "Capture baselines, compare new screenshots against them, and get diff images."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "set_baseline",
                                "compare",
                                "list_baselines",
                                "get_history",
                                "delete_baseline",
                            ],
                            "description": (
                                "set_baseline=save screenshot as new baseline; "
                                "compare=compare screenshot against baseline; "
                                "list_baselines=show all stored baselines; "
                                "get_history=show test history for a page; "
                                "delete_baseline=remove a baseline."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": "Page/component name (e.g. 'homepage', 'dashboard')",
                        },
                        "screenshot_b64": {
                            "type": "string",
                            "description": "Base64-encoded PNG for set_baseline or compare",
                        },
                        "url": {
                            "type": "string",
                            "description": "URL associated with this screenshot (metadata only)",
                        },
                        "threshold_percent": {
                            "type": "number",
                            "description": "Max % changed pixels before fail (default 0.5)",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        # Monitor tool schema
        schemas.append(
            {
                "name": "monitor",
                "description": (
                    "Self-healing monitor: check app health, fetch cloud/docker logs, and manage "
                    "continuous monitoring with auto-heal on crash."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "status",
                                "health_check",
                                "fetch_logs",
                                "register",
                                "start",
                                "stop",
                            ],
                            "description": (
                                "status=show monitor state; health_check=one-shot HTTP check; "
                                "fetch_logs=get logs from platform; register=add an app to continuous monitoring; "
                                "start/stop=control the monitor loop."
                            ),
                        },
                        "url": {
                            "type": "string",
                            "description": "App URL for health_check",
                        },
                        "platform": {
                            "type": "string",
                            "enum": ["digitalocean", "docker", "generic"],
                            "description": "Platform for fetch_logs or register",
                        },
                        "config": {
                            "type": "object",
                            "description": (
                                "Platform config dict. DO: {do_token, app_id}. Docker: {container_name}."
                            ),
                        },
                        "lines": {
                            "type": "integer",
                            "description": "Log lines to fetch (default 50)",
                        },
                        "name": {
                            "type": "string",
                            "description": "App name for register",
                        },
                        "health_url": {
                            "type": "string",
                            "description": "Health check URL for register",
                        },
                        "interval": {
                            "type": "integer",
                            "description": "Poll interval seconds (default 60)",
                        },
                        "failure_threshold": {
                            "type": "integer",
                            "description": "Failures before heal (default 3)",
                        },
                        "platform_config": {
                            "type": "object",
                            "description": "Platform-specific config for register",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        # Environment parity tool schema
        schemas.append(
            {
                "name": "env_parity",
                "description": (
                    "Ensure local and production environments are identical. "
                    "Generate Dockerfiles, .env.example, docker-compose.yml, and diff environments."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "diff",
                                "generate_dockerfile",
                                "generate_env_example",
                                "generate_docker_compose",
                            ],
                            "description": (
                                "diff=compare local .env vs production; "
                                "generate_dockerfile=create optimized Dockerfile; "
                                "generate_env_example=create .env.example from .env; "
                                "generate_docker_compose=create local dev compose file."
                            ),
                        },
                        "project_root": {
                            "type": "string",
                            "description": "Project root directory (default: workspace)",
                        },
                        "project_type": {
                            "type": "string",
                            "enum": ["auto", "python", "node", "fullstack"],
                            "description": "Project type for Dockerfile generation (default: auto-detect)",
                        },
                        "frontend_dir": {
                            "type": "string",
                            "description": "Frontend subdirectory (default: frontend)",
                        },
                        "requirements_file": {
                            "type": "string",
                            "description": "Python requirements file (default: requirements.txt)",
                        },
                        "port": {
                            "type": "integer",
                            "description": "App port (default: auto-detect from .env)",
                        },
                        "health_path": {
                            "type": "string",
                            "description": "Healthcheck path (default: /health)",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Output file path (optional)",
                        },
                        "env_file": {
                            "type": "string",
                            "description": "Source .env file for diff (default: .env)",
                        },
                        "source_env_file": {
                            "type": "string",
                            "description": "Source .env file for .env.example generation",
                        },
                        "include_current_values": {
                            "type": "boolean",
                            "description": "Include non-secret values in .env.example",
                        },
                        "include_redis": {
                            "type": "boolean",
                            "description": "Include Redis in docker-compose",
                        },
                        "include_postgres": {
                            "type": "boolean",
                            "description": "Include Postgres in docker-compose",
                        },
                        "production_env": {
                            "type": "object",
                            "description": "Production env vars dict for diff comparison",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        # Git tool schema
        schemas.append(
            {
                "name": "git",
                "description": "Perform structured git workflow actions like status, checkout_branch, add, commit, push, and diff.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "status",
                                "checkout_branch",
                                "add",
                                "commit",
                                "push",
                                "diff",
                            ],
                            "description": "The git action to perform",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Local repository path (default: .)",
                        },
                        "branch_name": {
                            "type": "string",
                            "description": "Branch name for checkout_branch / push",
                        },
                        "create": {
                            "type": "boolean",
                            "description": "For checkout_branch: create the branch with git checkout -b",
                        },
                        "base_branch": {
                            "type": "string",
                            "description": "Optional starting point when creating a branch",
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files for add / commit. Empty means all changes.",
                        },
                        "commit_message": {
                            "type": "string",
                            "description": "Commit message for commit",
                        },
                        "remote": {
                            "type": "string",
                            "description": "Remote name for push (default: origin)",
                        },
                        "set_upstream": {
                            "type": "boolean",
                            "description": "For push: include -u when pushing the branch",
                        },
                        "allow_default_branch_push": {
                            "type": "boolean",
                            "description": "Explicitly allow push to the remote default branch",
                        },
                        "staged_only": {
                            "type": "boolean",
                            "description": "For diff: show only staged changes",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        # GitHub tool schema
        schemas.append(
            {
                "name": "github",
                "description": "Perform GitHub operations like fetching issue/PR context, creating branches, committing, and creating pull requests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "get_repo_context",
                                "list_issues",
                                "get_issue",
                                "list_pull_requests",
                                "get_pull_request",
                                "create_branch",
                                "commit",
                                "create_pr",
                                "automated_workflow",
                                "get_pr_status",
                                "merge_pr",
                            ],
                            "description": "The GitHub action to perform",
                        },
                        "branch_name": {
                            "type": "string",
                            "description": "Name of the branch (for create_branch, create_pr, automated_workflow)",
                        },
                        "base_branch": {
                            "type": "string",
                            "description": 'Base branch for PR / branch-off; omit or use "default" for the GitHub repo default (main/master)',
                        },
                        "commit_message": {
                            "type": "string",
                            "description": "Commit message (for commit)",
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of files to commit. Empty means all changes.",
                        },
                        "pr_title": {
                            "type": "string",
                            "description": "Pull Request title",
                        },
                        "pr_description": {
                            "type": "string",
                            "description": "Pull Request description",
                        },
                        "task_description": {
                            "type": "string",
                            "description": "Task description (for automated_workflow)",
                        },
                        "issue_number": {
                            "type": "integer",
                            "description": "Issue number for get_issue",
                        },
                        "state": {
                            "type": "string",
                            "description": "State filter for list_issues / list_pull_requests (default: open)",
                        },
                        "include_comments": {
                            "type": "boolean",
                            "description": "Include issue or pull request comments in the response",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results for list actions",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Local repository path (default: .)",
                        },
                        "pr_number": {
                            "type": "integer",
                            "description": "Pull request number (get_pr_status, merge_pr)",
                        },
                        "merge_method": {
                            "type": "string",
                            "enum": ["squash", "merge", "rebase"],
                            "description": "For merge_pr (default: squash)",
                        },
                        "draft": {
                            "type": "boolean",
                            "description": "For create_pr: open as draft (requires GitHub + token scopes)",
                        },
                        "assignees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "GitHub usernames to assign after create_pr",
                        },
                        "linked_issues": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Issue numbers; appends Closes #n to PR body",
                        },
                    },
                    "required": ["action"],
                },
            }
        )

        return self._filter_tool_schemas(schemas)

    # Tool groups eligible for task-aware gating. Core tools (terminal, editor, ask_user,
    # git, project_memory) are intentionally absent here and always retained. Each gated
    # group is only sent to the model when its keywords appear anywhere in the live
    # conversation (task goal + tool outputs), so a tool re-appears the moment it becomes
    # relevant. This trims the per-call tool schema (~8-10k tokens) for tasks that never
    # touch the web/ops/test surfaces. Disable with PLODDER_TOOL_GATING=false.
    _GATED_TOOL_GROUPS: dict[str, dict[str, Any]] = {
        "browser": {
            "tools": {
                "browser_open",
                "browser_search",
                "browser_fetch",
                "browser_interactive",
                "browser_playwright",
                "browser_navigate",
                "browser_click",
                "browser_type",
                "browser_scroll",
                "browser_screenshot",
                "live_preview",
            },
            "keywords": (
                "http://",
                "https://",
                "www.",
                "browser",
                "website",
                "web page",
                "webpage",
                "scrape",
                "crawl",
                "screenshot",
                "url",
                "localhost",
                "navigate",
                "frontend",
                "preview",
                "render",
                "dom",
                "selenium",
                "playwright",
                "html page",
                "site ",
                "search the web",
                "fetch page",
                "live preview",
            ),
        },
        "uitest": {
            "tools": {"ui_test", "visual_regression"},
            "keywords": (
                "ui test",
                "e2e",
                "end-to-end",
                "visual regression",
                "screenshot diff",
                "baseline",
                "pixel diff",
                "assert element",
                "regression test",
            ),
        },
        "ops": {
            "tools": {"monitor", "env_parity"},
            "keywords": (
                "deploy",
                "docker",
                "dockerfile",
                "compose",
                "monitor",
                "health check",
                "production",
                "digitalocean",
                "kubernetes",
                "container",
                "env parity",
                "self-heal",
                "fetch logs",
            ),
        },
        "plan": {
            "tools": {"project_plan"},
            "keywords": (
                "milestone",
                "roadmap",
                "multi-week",
                "long-term plan",
                "decompose the",
                "project plan",
                "break into phases",
                "sub-agent",
            ),
        },
        "github": {
            "tools": {"github"},
            "keywords": (
                "github",
                "pull request",
                "create pr",
                "open a pr",
                "merge pr",
                "git push",
                "issue #",
                "create a branch",
                "create_pr",
            ),
        },
    }

    def _tool_gating_context_text(self) -> str:
        """Lowercased blob of the live conversation used for task-aware tool gating.

        The system prompt is deliberately excluded: it documents every tool (browser,
        github, docker, playwright, ...) so including it would keep all groups permanently
        active and defeat gating. Only the real task + tool outputs drive relevance.
        """
        parts: list[str] = []
        try:
            for msg in self.llm.conversation:
                if getattr(msg, "role", None) == "system":
                    continue
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    parts.append(content)
                for tc in getattr(msg, "tool_calls", None) or []:
                    parts.append(getattr(tc, "name", "") or "")
                    args = getattr(tc, "arguments", None)
                    if isinstance(args, str):
                        parts.append(args)
        except Exception:
            return ""
        return " ".join(parts).lower()

    def _filter_tool_schemas(
        self, schemas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Drop heavy, task-irrelevant tool schemas to cut per-call prompt tokens.

        Conservative + reversible: a group is only removed when none of its keywords appear
        in the conversation. Core tools are always kept. Controlled by PLODDER_TOOL_GATING
        (default on); benchmark harnesses can set it to ``false`` to send every tool.
        """
        if os.environ.get("PLODDER_TOOL_GATING", "true").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return schemas

        # Map each gated tool name -> group key (built once).
        name_to_group = getattr(self, "_tool_name_to_group", None)
        if name_to_group is None:
            name_to_group = {}
            for group, spec in self._GATED_TOOL_GROUPS.items():
                for tool_name in spec["tools"]:
                    name_to_group[tool_name] = group
            self._tool_name_to_group = name_to_group

        blob = self._tool_gating_context_text()
        active_groups = {
            group
            for group, spec in self._GATED_TOOL_GROUPS.items()
            if any(kw in blob for kw in spec["keywords"])
        }

        kept: list[dict[str, Any]] = []
        for schema in schemas:
            group = name_to_group.get(schema.get("name", ""))
            if group is None or group in active_groups:
                kept.append(schema)

        trimmed = len(schemas) - len(kept)
        if trimmed and getattr(self, "_last_tool_gating_kept", None) != len(kept):
            self._last_tool_gating_kept = len(kept)
            self._log(
                f"Tool gating: sending {len(kept)}/{len(schemas)} tool schemas "
                f"(trimmed {trimmed}; active groups: {sorted(active_groups) or 'none'})."
            )
        return kept

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        thought: str | None = None,
        activity_source: str = "agent",
        _inline_backbone: bool = False,
    ) -> str:
        """Execute a tool and return the result as a string."""
        import time

        start_time = time.time()
        call_id = str(uuid.uuid4())[:8]
        arg_dict = dict(arguments) if isinstance(arguments, dict) else {}

        # ── ask_user: pause and request clarification from user ──
        if name == "ask_user":
            question = arguments.get("question", "Can you clarify?")
            options = arguments.get("options")  # optional list of choice strings
            context = arguments.get("context", "")  # optional extra context

            on_clarification = self.callbacks.get("on_clarification_needed")
            if on_clarification:
                # Pass full payload so frontend can render option buttons
                on_clarification(
                    {
                        "question": question,
                        "options": options or [],
                        "context": context,
                    }
                )
            # Wait up to 5 minutes for user answer
            self._clarification_event = asyncio.Event()
            self._clarification_answer = None
            try:
                await asyncio.wait_for(self._clarification_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                # Auto-select first option if available, otherwise best guess
                if options:
                    return f"User did not answer within 5 minutes. Proceeding with option: {options[0]}"
                return (
                    "User did not answer within 5 minutes. Proceeding with best guess."
                )
            answer = self._clarification_answer or "(no answer)"
            self._clarification_event = None
            return f"User answered: {answer}"

        if name == "live_preview":
            from ..api.live_preview_state import (
                allowed_ports,
                live_preview_probe_hints,
                live_preview_set_port_warning,
                probe_local_ports_sync,
                set_session_preview_port,
            )

            action = str(arguments.get("action", "probe") or "probe").lower()
            sid = self.state.session_id
            if action == "probe":
                ports = arguments.get("ports")
                if not isinstance(ports, list) or not ports:
                    raw = (
                        os.environ.get("LIVE_PREVIEW_PROBE_PORTS")
                        or "5173,3000,3001,3002,4173,4200,5000,5001,5002,8000,8080"
                    ).strip()
                    ports = []
                    for x in raw.split(","):
                        x = x.strip()
                        if x.isdigit():
                            ports.append(int(x))
                try:
                    candidates = [int(p) for p in ports]
                except (TypeError, ValueError):
                    candidates = [
                        5173,
                        3000,
                        3001,
                        3002,
                        4173,
                        5000,
                        5001,
                        5002,
                        8000,
                        8080,
                    ]
                listening = probe_local_ports_sync(candidates)
                payload: dict[str, Any] = {
                    "listening_ports": listening,
                    "allowed_ports": sorted(allowed_ports()),
                    "next_step": (
                        "Call live_preview with action set_active_port and port=<one of listening_ports> "
                        "only if that port is **your workspace dev server** (e.g. Vite). "
                        "Do not use Live Preview to open external domains."
                    ),
                }
                payload.update(live_preview_probe_hints(listening_ports=listening))
                return json.dumps(payload, indent=2)
            if action == "set_active_port":
                try:
                    p = int(arguments.get("port", 0))
                except (TypeError, ValueError):
                    return json.dumps({"ok": False, "error": "invalid port"})
                ok = await set_session_preview_port(sid, p)
                if not ok:
                    return json.dumps(
                        {
                            "ok": False,
                            "error": f"Port {p} not allowed or not accepting TCP on 127.0.0.1",
                            "allowed_ports": sorted(allowed_ports()),
                        },
                        indent=2,
                    )
                out_ok: dict[str, Any] = {
                    "ok": True,
                    "active_port": p,
                    "browser_iframe": f"/api/sessions/{sid}/live-preview/",
                }
                w = live_preview_set_port_warning(p)
                if w:
                    out_ok["warning"] = w
                return json.dumps(out_ok, indent=2)
            return f"Error: unknown live_preview action '{action}'"

        if name == "browser_open":
            url = str(arg_dict.get("url", "") or "").strip()
            if not re.match(r"^https?://", url, re.IGNORECASE):
                return (
                    "Error: browser_open requires an absolute http:// or https:// URL."
                )
            from urllib.parse import urlparse

            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            if host in {"localhost", "127.0.0.1", "::1"}:
                return (
                    "Error: browser_open cannot load localhost/127.0.0.1 from the remote Plodder Browser. "
                    "Use live_preview for a local dev server, or browser_playwright/browser_fetch for a public URL."
                )
            payload = {
                "event_type": "navigate",
                "url": url,
                "query": str(
                    arg_dict.get("note", "") or "Opened in Plodder Browser tab"
                ),
            }
            await self._trigger_callback("on_browser_event", payload)
            if self.working_directory:
                self._append_session_event(
                    AgentStreamEvent(
                        kind=AgentEventKind.TOOL_CALL,
                        tool_name="browser_open",
                        tool_args=self._sanitize_tool_args_for_log(arg_dict),
                        legacy_type="act",
                        meta={"visible_browser": True},
                    )
                )
                self._append_session_event(
                    AgentStreamEvent(
                        kind=AgentEventKind.OBSERVATION,
                        tool_name="browser_open",
                        legacy_type="observe",
                        output=f"Opened URL in Plodder Browser tab: {url}",
                        meta={"visible_browser": True, "url": url},
                    )
                )
            return (
                f"Opened URL in the user's Plodder Browser tab: {url}\n"
                "If the site blocks iframe embedding, the Browser tab still shows an external-open link."
            )

        def _is_high_risk_terminal_command(cmd: str) -> tuple[bool, str]:
            c = (cmd or "").strip().lower()
            if not c:
                return False, ""
            risky_patterns = [
                (r"\brm\s+-rf\b", "destructive recursive delete"),
                (r"\brm\s+-r\b", "recursive delete"),
                (r"\bdel\s+/[a-z]*f", "forced file delete"),
                (r"\brmdir\s+/[a-z]*s", "recursive directory delete"),
                (r"\bgit\s+reset\s+--hard\b", "destructive git reset"),
                (r"\bgit\s+clean\s+-[a-z]*f", "destructive git clean"),
                (r"\bmkfs\b", "filesystem format command"),
                (r"\bdd\s+if=", "raw disk write command"),
            ]
            for pat, reason in risky_patterns:
                if re.search(pat, c):
                    return True, reason
            return False, ""

        async def _require_human_approval(question: str, context: str) -> bool:
            on_clarification = self.callbacks.get("on_clarification_needed")
            if on_clarification:
                on_clarification(
                    {
                        "question": question,
                        "options": ["Approve", "Reject"],
                        "context": context,
                    }
                )
            self._clarification_event = asyncio.Event()
            self._clarification_answer = None
            try:
                await asyncio.wait_for(self._clarification_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                self._clarification_event = None
                self._clarification_answer = None
                return False
            answer = (self._clarification_answer or "").strip().lower()
            self._clarification_event = None
            self._clarification_answer = None
            return answer in {"approve", "approved", "yes", "y", "allow", "continue", "proceed"}

        # Human-in-the-loop gate for risky actions.
        if name == "terminal":
            cmd = str(arg_dict.get("command", "") or "")
            risky, reason = _is_high_risk_terminal_command(cmd)
            if risky:
                approved = await _require_human_approval(
                    "This command looks risky. Approve execution?",
                    f"Reason: {reason}\nCommand:\n{cmd}",
                )
                if not approved:
                    return f"Blocked by human approval gate: {reason}. Command was not executed."

        if name in {"editor", "atomic_edit"}:
            action = str(arg_dict.get("action", "") or "").lower()
            delete_actions = {"delete_file", "remove_file", "unlink", "delete"}
            if action in delete_actions:
                target_path = str(arg_dict.get("path", "") or "")
                approved = await _require_human_approval(
                    "Agent requested file deletion. Approve?",
                    f"Action: {action}\nPath: {target_path or '(missing path)'}",
                )
                if not approved:
                    return "Blocked by human approval gate: file deletion was rejected."

        tool = self.registry.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        if name == "editor" and not arg_dict:
            return (
                "Error: malformed editor tool call. Missing required arguments. "
                "Use one of these forms: "
                '{"action":"write_file","path":"index.html","content":"..."}; '
                '{"action":"read_file","path":"PLAN.md"}; '
                '{"action":"list_directory","path":"."}.'
            )
        if (
            name == "editor"
            and arg_dict.get("action") in (None, "read_file")
            and not arg_dict.get("path")
        ):
            return (
                "Error: malformed editor read_file call. Missing `path`. "
                "If you need to create a file, use action=write_file with path and full content."
            )
        state_before_snapshot = self._activity_state.to_meta_snapshot()
        action_type = classify_action_type(name, arg_dict)
        step = self._activity_state.bump_step()
        self._activity_state.record_tool(name, action_type)
        act_meta = build_activity_meta(
            thought=thought or "",
            source=activity_source
            if activity_source in ("agent", "user", "system")
            else "agent",
            action_type=action_type,
            step=step,
            tool_name=name,
        )
        if isinstance(act_meta.get("activity"), dict):
            act_meta["activity"]["state_before"] = state_before_snapshot
        self._current_activity_context = act_meta

        ok_pre, pre_msg, viol = validate_action_pre_flight(
            name,
            arg_dict,
            is_windows=(os.name == "nt"),
            command_safety_check=self._check_command_safety,
        )
        if not ok_pre:
            if viol is not None and getattr(viol, "blocked", False):
                self._update_phase(AgentPhase.BLOCKED)
                err_out = f"BLOCKED: {pre_msg}. Task moved to BLOCKED state."
            else:
                err_out = (
                    f"Error: Command failed sanity check (not executed): {pre_msg}\n"
                    "Fix the command syntax or paths, then retry."
                )
            if self.working_directory:
                obs_act = dict(act_meta.get("activity") or {})
                obs_act["validation_failed"] = True
                obs_act["preflight_message"] = pre_msg
                self._append_session_event(
                    AgentStreamEvent(
                        kind=AgentEventKind.OBSERVATION,
                        tool_name=name,
                        legacy_type="observe",
                        output=err_out[:8000],
                        meta={"preflight_failed": True, "activity": obs_act},
                    )
                )
            self._current_activity_context = None
            return err_out

        try:
            if self.working_directory and not _inline_backbone:
                self._append_session_event(
                    AgentStreamEvent(
                        kind=AgentEventKind.TOOL_CALL,
                        tool_name=name,
                        tool_call_id=call_id,
                        tool_args=self._sanitize_tool_args_for_log(arguments),
                        legacy_type="act",
                        meta=dict(act_meta),
                    )
                )

            if (
                self._backbone_controller is not None
                and not _inline_backbone
                and getattr(self, "_use_backbone_dispatch", False)
                and not self.use_sandbox
            ):
                return await self._backbone_controller.dispatch_and_wait(
                    name,
                    arg_dict,
                )

            if name == "terminal":
                command = arguments.get("command", "")
                self._last_terminal_command = str(command)
                plan_step = arguments.get("plan_step")

                fs_before = self._workspace_path_set()

                # Emit command to shell stream
                on_cmd_output = self.callbacks.get("on_command_output")
                on_cmd_start = self.callbacks.get("on_command_start")
                if on_cmd_start:
                    on_cmd_start(command)  # fires a "command" typed line with ts
                elif on_cmd_output:
                    on_cmd_output(f"$ {command}")

                from ..schemas.tools import TerminalInput

                raw_timeout = arguments.get("timeout_seconds", 30)
                try:
                    timeout_seconds = int(raw_timeout)
                except (TypeError, ValueError):
                    timeout_seconds = 30
                timeout_seconds = max(1, min(300, timeout_seconds))

                def _terminal_artifact(
                    out: str, success: bool, exit_c: int | None
                ) -> None:
                    if self._artifact_logger:
                        duration_ms = int((time.time() - start_time) * 1000)
                        self._artifact_logger.log_tool_call(
                            call_id=call_id,
                            tool_name="terminal",
                            arguments=arguments,
                            result=out,
                            duration_ms=duration_ms,
                            success=success,
                        )
                        self._artifact_logger.add_command_executed(command)

                # ── ProcessExecutionSandbox first (USE_PROCESS_EXECUTION_SANDBOX) ──
                # Avoid touching Docker when the daemon/socket is absent.
                if self._should_use_process_execution_terminal():
                    work_dir = arguments.get("working_directory", ".")
                    workdir_ps = None if work_dir in (".", "", None) else work_dir
                    pex = self._get_pex_terminal()
                    dev_server = bool(
                        re.search(
                            r"(?i)\b("
                            r"npm\s+run\s+(dev|start)|"
                            r"yarn\s+(dev|start)|"
                            r"pnpm\s+(dev|start)|"
                            r"next\s+dev|vite\b|astro\s+dev|parcel\b|"
                            r"flask\s+run|"
                            r"python(?:\d+(?:\.\d+)?)?\s+[^;&\n]*\.py\b|"
                            r"node\s+[^;&\n]*(server|app|index)\.js\b"
                            r")",
                            command,
                        )
                    )
                    try:
                        if dev_server and callable(
                            getattr(pex, "run_dev_server_in_workspace", None)
                        ):
                            runner_command = command
                            if workdir_ps not in (None, ".", ""):
                                runner_command = (
                                    f"cd {shlex.quote(str(workdir_ps))} && {command}"
                                )
                            dev_res = await asyncio.to_thread(
                                pex.run_dev_server_in_workspace,
                                {},
                                ["sh", "-c", runner_command],
                                timeout_sec=timeout_seconds,
                            )
                            output_parts = []
                            if dev_res.get("stdout"):
                                output_parts.append(f"STDOUT:\n{dev_res['stdout']}")
                                if on_cmd_output:
                                    for line in str(dev_res["stdout"]).splitlines():
                                        on_cmd_output(line)
                            if dev_res.get("stderr"):
                                output_parts.append(f"STDERR:\n{dev_res['stderr']}")
                                if on_cmd_output:
                                    for line in str(dev_res["stderr"]).splitlines():
                                        on_cmd_output(f"[stderr] {line}")
                            if (
                                dev_res.get("ok")
                                and int(dev_res.get("port", 0) or 0) > 0
                            ):
                                from ..api.live_preview_state import (
                                    live_preview_set_port_warning,
                                    set_session_preview_port,
                                )

                                started_port = int(dev_res["port"])
                                ok = await set_session_preview_port(
                                    self.state.session_id, started_port
                                )
                                lp_payload: dict[str, Any]
                                if ok:
                                    lp_payload = {
                                        "ok": True,
                                        "active_port": started_port,
                                        "browser_iframe": f"/api/sessions/{self.state.session_id}/live-preview/",
                                    }
                                    warning = live_preview_set_port_warning(
                                        started_port
                                    )
                                    if warning:
                                        lp_payload["warning"] = warning
                                else:
                                    lp_payload = {
                                        "ok": False,
                                        "error": f"Port {started_port} was not reachable for live preview attachment.",
                                    }
                                output_parts.append(
                                    f"LIVE_PREVIEW:\n{json.dumps(lp_payload, indent=2)}"
                                )
                            exit_code = int(dev_res.get("exit_code", 0) or 0)
                            output_parts.append(f"Exit code: {exit_code}")
                            if on_cmd_output:
                                on_cmd_output(f"Exit code: {exit_code}")
                            output = "\n".join(output_parts)
                            _terminal_artifact(output, exit_code == 0, exit_code)
                            return self._attach_filesystem_observe(
                                output,
                                fs_before,
                                tool="terminal",
                                exit_code=exit_code,
                                plan_step=plan_step,
                            )

                        sb_res = await asyncio.to_thread(
                            pex.exec_shell,
                            command,
                            workdir=workdir_ps,
                            timeout_sec=timeout_seconds,
                        )
                    except Exception as e:
                        err = f"Error executing terminal command (process sandbox): {e}"
                        _terminal_artifact(err, False, None)
                        return self._attach_filesystem_observe(
                            err,
                            fs_before,
                            tool="terminal",
                            exit_code=None,
                            plan_step=plan_step,
                        )
                    output_parts = []
                    if sb_res.stdout:
                        output_parts.append(f"STDOUT:\n{sb_res.stdout}")
                        if on_cmd_output:
                            for line in sb_res.stdout.splitlines():
                                on_cmd_output(line)
                    if getattr(sb_res, "stderr", None):
                        output_parts.append(f"STDERR:\n{sb_res.stderr}")
                        if on_cmd_output:
                            for line in sb_res.stderr.splitlines():
                                on_cmd_output(f"[stderr] {line}")
                    if getattr(sb_res, "timed_out", False):
                        output_parts.append(
                            "(process sandbox) Command timed out and was terminated."
                        )
                    output_parts.append(f"Exit code: {sb_res.exit_code}")
                    if on_cmd_output:
                        on_cmd_output(f"Exit code: {sb_res.exit_code}")
                    output = "\n".join(output_parts)
                    _terminal_artifact(output, sb_res.exit_code == 0, sb_res.exit_code)
                    return self._attach_filesystem_observe(
                        output,
                        fs_before,
                        tool="terminal",
                        exit_code=sb_res.exit_code,
                        plan_step=plan_step,
                    )

                # ── Docker Sandbox Execution (optional; skipped when host-process execution runs) ──
                if self.use_sandbox and self._sandbox is not None:
                    try:
                        if not self._sandbox.is_running():
                            started = await self._sandbox.start()
                            if not started:
                                self.use_sandbox = False

                        if self._sandbox.is_running():
                            sandbox_result = await self._sandbox.execute(
                                command,
                                working_dir=arguments.get("working_directory"),
                                on_output_line=on_cmd_output,
                            )
                            output_parts = []
                            if sandbox_result.stdout:
                                output_parts.append(f"STDOUT:\n{sandbox_result.stdout}")
                                if on_cmd_output and not getattr(
                                    sandbox_result, "streamed_live", False
                                ):
                                    for line in sandbox_result.stdout.splitlines():
                                        on_cmd_output(line)
                            if sandbox_result.stderr:
                                output_parts.append(f"STDERR:\n{sandbox_result.stderr}")
                                if on_cmd_output and not getattr(
                                    sandbox_result, "streamed_live", False
                                ):
                                    for line in sandbox_result.stderr.splitlines():
                                        on_cmd_output(line)
                            output_parts.append(
                                f"Exit code: {sandbox_result.exit_code}"
                            )
                            if on_cmd_output:
                                on_cmd_output(f"Exit code: {sandbox_result.exit_code}")
                            output = "\n".join(output_parts)
                            _terminal_artifact(
                                output,
                                sandbox_result.exit_code == 0,
                                sandbox_result.exit_code,
                            )
                            return self._attach_filesystem_observe(
                                output,
                                fs_before,
                                tool="terminal",
                                exit_code=sandbox_result.exit_code,
                                plan_step=plan_step,
                            )
                    except Exception as sandbox_err:
                        self._log(f"Sandbox error, falling back to host: {sandbox_err}")

                # ── Regular Host Execution (terminal tool) ──
                input_data = TerminalInput(
                    command=command,
                    working_directory=arguments.get("working_directory", "."),
                    timeout_seconds=timeout_seconds,
                )

                result = None
                try:
                    result = await tool.execute(input_data)
                except Exception as e:
                    err = f"Error executing terminal command: {str(e)}"
                    _terminal_artifact(err, False, None)
                    return self._attach_filesystem_observe(
                        err,
                        fs_before,
                        tool="terminal",
                        exit_code=None,
                        plan_step=plan_step,
                    )

                output_parts = []
                if result.stdout:
                    output_parts.append(f"STDOUT:\n{result.stdout}")
                    if on_cmd_output:
                        for line in result.stdout.splitlines():
                            on_cmd_output(line)

                if result.stderr:
                    output_parts.append(f"STDERR:\n{result.stderr}")
                    if on_cmd_output:
                        for line in result.stderr.splitlines():
                            on_cmd_output(line)

                output_parts.append(f"Exit code: {result.exit_code}")
                if on_cmd_output:
                    on_cmd_output(f"Exit code: {result.exit_code}")

                output = "\n".join(output_parts)
                _terminal_artifact(output, result.exit_code == 0, result.exit_code)
                return self._attach_filesystem_observe(
                    output,
                    fs_before,
                    tool="terminal",
                    exit_code=result.exit_code,
                    plan_step=plan_step,
                )

            elif name == "editor":
                fs_before = self._workspace_path_set()
                plan_step = arguments.get("plan_step")
                ed_args = dict(arguments)
                ed_args.pop("plan_step", None)
                apply_ruff_fix = bool(ed_args.pop("apply_ruff_fix", False))
                action = ed_args.get("action", "read_file")
                file_path = ed_args.get("path", "")

                if action == "read_file":
                    from ..schemas.tools import ReadFileInput, FileRange

                    line_range = None
                    if ed_args.get("start_line"):
                        line_range = FileRange(
                            start_line=ed_args["start_line"],
                            end_line=ed_args.get("end_line"),
                        )
                    input_data = ReadFileInput(
                        path=file_path,
                        line_range=line_range,
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        output = f"File: {file_path} ({result.total_lines} lines)\n\n{result.content}"
                    else:
                        output = f"Error: {result.error_message}"

                elif action == "write_file":
                    content = ed_args.get("content", "")

                    # Safety check for file edits
                    violation = self._check_file_edit_safety(file_path, content)
                    if violation and violation.blocked:
                        self._update_phase(AgentPhase.BLOCKED)
                        return f"BLOCKED: {violation.message}. Task moved to BLOCKED state."

                    # Safety check for dependency files
                    abs_target = (
                        Path(self.working_directory or ".") / file_path
                    ).resolve()
                    dep_violation = self._check_dependency_safety(
                        file_path,
                        change_type="create" if not abs_target.exists() else "bump",
                    )
                    if dep_violation and dep_violation.blocked:
                        self._update_phase(AgentPhase.BLOCKED)
                        return f"BLOCKED: {dep_violation.message}. Task moved to BLOCKED state."

                    from ..schemas.tools import WriteFileInput

                    input_data = WriteFileInput(
                        path=file_path,
                        content=content,
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        output = f"Successfully wrote {result.bytes_written} bytes to {result.path}"
                        if self._artifact_logger:
                            self._artifact_logger.add_file_modified(file_path)
                    else:
                        output = f"Error: {result.error_message}"

                elif action == "str_replace":
                    old_str = ed_args.get("old_str", "")
                    new_str = ed_args.get("new_str", "")
                    allow_multiple = ed_args.get("allow_multiple", False)

                    dep_violation = self._check_dependency_safety(file_path)
                    if dep_violation and dep_violation.blocked:
                        self._update_phase(AgentPhase.BLOCKED)
                        return f"BLOCKED: {dep_violation.message}. Task moved to BLOCKED state."

                    from ..schemas.tools import StrReplaceInput

                    input_data = StrReplaceInput(
                        path=file_path,
                        old_str=old_str,
                        new_str=new_str,
                        allow_multiple=allow_multiple,
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        output = f"str_replace: {result.replacements_made} replacement(s) in {result.path}"
                        if self._artifact_logger:
                            self._artifact_logger.add_file_modified(file_path)
                    else:
                        output = f"Error: {result.error_message}"

                elif action == "search":
                    from ..schemas.tools import SearchInput

                    input_data = SearchInput(
                        pattern=ed_args.get("pattern", ""),
                        path=file_path,
                        file_pattern=ed_args.get("file_pattern"),
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        if not result.matches:
                            output = "No matches found"
                        else:
                            matches_str = "\n".join(
                                [
                                    f"{m.file_path}:{m.line_number}: {m.line_content}"
                                    for m in result.matches[:50]
                                ]
                            )
                            output = (
                                f"Found {result.total_matches} matches:\n{matches_str}"
                            )
                    else:
                        output = f"Error: {result.error_message}"

                elif action == "list_directory":
                    from ..schemas.tools import ListDirectoryInput

                    input_data = ListDirectoryInput(
                        path=file_path,
                        recursive=ed_args.get("recursive", False),
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        entries_str = "\n".join(
                            [
                                f"{'[DIR] ' if e.is_directory else ''}{e.name}"
                                for e in result.entries[:100]
                            ]
                        )
                        output = f"Directory: {file_path}\n{result.total_directories} directories, {result.total_files} files\n\n{entries_str}"
                    else:
                        output = f"Error: {result.error_message}"

                elif action == "apply_patch":
                    patch = ed_args.get("patch", "")

                    # Safety check for patch (estimate lines changed)
                    violation = self._check_file_edit_safety(file_path, patch)
                    if violation and violation.blocked:
                        self._update_phase(AgentPhase.BLOCKED)
                        return f"BLOCKED: {violation.message}. Task moved to BLOCKED state."

                    from ..schemas.tools import ApplyPatchInput

                    input_data = ApplyPatchInput(
                        path=file_path,
                        patch=patch,
                    )
                    result = await tool.execute(input_data)
                    if result.status.value == "success":
                        output = f"Patch applied: {result.hunks_applied} hunks applied, {result.hunks_failed} failed"
                        if self._artifact_logger:
                            self._artifact_logger.add_file_modified(file_path)
                    else:
                        output = f"Error: {result.error_message}"

                elif action == "revert_file":
                    import subprocess
                    try:
                        subprocess.run(
                            ["git", "checkout", "HEAD", "--", file_path],
                            cwd=self.working_directory or ".",
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        output = f"Successfully reverted {file_path} to its last committed state."
                        if self._artifact_logger:
                            self._artifact_logger.add_file_modified(file_path)
                    except subprocess.CalledProcessError as e:
                        output = f"Error reverting {file_path}: {e.stderr}"

                else:
                    output = f"Error: Unknown editor action '{action}'"

                written_paths: list[str] = []
                if (
                    action in ("write_file", "str_replace", "apply_patch", "revert_file")
                    and "Error" not in output
                    and "BLOCKED" not in output
                ):
                    written_paths = [file_path]

                extra_verify = ""
                if written_paths:
                    fix_block = ""
                    if (
                        apply_ruff_fix
                        and file_path.endswith(".py")
                        and "Error" not in output
                        and "BLOCKED" not in output
                    ):
                        fix_block = await self._run_ruff_fix_file(file_path)
                    extra_verify = fix_block + await self._auto_ruff_check_file(
                        file_path
                    )

                out_final = self._attach_filesystem_observe(
                    output + extra_verify,
                    fs_before,
                    tool="editor",
                    exit_code=None,
                    plan_step=plan_step,
                    written_paths=written_paths or None,
                )

                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="editor",
                        arguments=arguments,
                        result=out_final[:5000],  # Truncate for logging
                        duration_ms=duration_ms,
                        success="Error" not in output and "BLOCKED" not in output,
                    )

                if (
                    file_path
                    and action
                    in ("read_file", "write_file", "str_replace", "apply_patch")
                    and "Error" not in out_final
                    and "BLOCKED" not in out_final
                ):
                    self._context_focus_path = str(file_path).replace("\\", "/")

                if (
                    file_path
                    and action in ("write_file", "str_replace", "apply_patch")
                    and "Error" not in out_final
                    and "BLOCKED" not in out_final
                ):
                    await self._emit_file_changed_ui(str(file_path).replace("\\", "/"))

                return out_final

            elif name == "browser_search":
                query = arguments.get("query", "")
                max_results = arguments.get("max_results", 10)

                # Create a simple input object for the search tool
                class SearchInput:
                    def __init__(self, q, mr):
                        self.query = q
                        self.max_results = mr

                input_data = SearchInput(query, max_results)
                result = await tool.execute(input_data)

                if result.success:
                    search_response = result.data
                    results_str = "\n".join(
                        [
                            f"{i+1}. [{r.title}]({r.url})\n   {r.snippet[:200]}..."
                            for i, r in enumerate(search_response.results[:max_results])
                        ]
                    )
                    output = f"Search results for '{query}':\n\n{results_str}"

                    # Add citations for search results
                    if hasattr(self, "_citation_store"):
                        for r in search_response.results:
                            self._citation_store.add(
                                url=r.url,
                                title=r.title,
                                snippet=r.snippet,
                            )
                else:
                    output = f"Search failed: {result.message}"

                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="browser_search",
                        arguments=arguments,
                        result=output[:5000],
                        duration_ms=duration_ms,
                        success=result.success,
                    )

                return output

            elif name == "browser_fetch":
                url = arguments.get("url", "")
                extract_content = arguments.get("extract_content", True)

                # Create a simple input object for the fetch tool
                class FetchInput:
                    def __init__(self, u, ec):
                        self.url = u
                        self.extract_content = ec
                        self.use_cache = True

                input_data = FetchInput(url, extract_content)
                result = await tool.execute(input_data)

                if result.success:
                    fetch_response = result.data
                    page = fetch_response.page

                    # Truncate content if too long
                    content = page.content
                    if len(content) > 10000:
                        content = content[:10000] + "\n\n[Content truncated...]"

                    output = f"Fetched: {page.title}\nURL: {page.url}\nWords: {page.word_count}\n\n{content}"

                    # Add citation
                    if hasattr(self, "_citation_store"):
                        self._citation_store.add(
                            url=page.url,
                            title=page.title,
                            snippet=page.content[:500] if page.content else "",
                        )
                else:
                    output = f"Fetch failed: {result.message}"

                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="browser_fetch",
                        arguments=arguments,
                        result=output[:5000],
                        duration_ms=duration_ms,
                        success=result.success,
                    )

                return output

            elif name == "browser_interactive":
                action = arguments.get("action", "navigate")

                # Create a simple input object for the interactive tool
                class InteractiveInput:
                    def __init__(self, args):
                        self.action = args.get("action", "navigate")
                        self.url = args.get("url", "")
                        self.selector = args.get("selector", "")
                        self.text = args.get("text", "")
                        self.direction = args.get("direction", "down")
                        self.seconds = args.get("seconds", 1)
                        self.selector_type = args.get("selector_type", "css")
                        self.clear_first = args.get("clear_first", True)

                input_data = InteractiveInput(arguments)
                result = await tool.execute(input_data)

                if result.success:
                    response = result.data
                    page_state = response.page_state

                    output_parts = [f"Browser action '{action}' completed."]
                    if page_state:
                        output_parts.append(f"URL: {page_state.url}")
                        output_parts.append(f"Title: {page_state.title}")
                        if page_state.text:
                            text = page_state.text
                            if len(text) > 5000:
                                text = text[:5000] + "\n\n[Content truncated...]"
                            output_parts.append(f"\nPage text:\n{text}")

                    output = "\n".join(output_parts)
                    # Stream screenshot / URL to dashboard Browser tab (tool_completed only sends strings)
                    if page_state:
                        shot = getattr(page_state, "screenshot_base64", None) or None
                        ev = (
                            "screenshot"
                            if shot
                            else (
                                "navigate"
                                if str(action).lower() == "navigate"
                                else "other"
                            )
                        )
                        await self._trigger_callback(
                            "on_browser_event",
                            {
                                "event_type": ev,
                                "url": page_state.url,
                                "query": None,
                                "screenshot_base64": shot,
                            },
                        )
                else:
                    output = f"Browser action failed: {result.message}"

                # Log to artifacts
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="browser_interactive",
                        arguments=arguments,
                        result=output[:5000],
                        duration_ms=duration_ms,
                        success=result.success,
                    )

                return output

            elif name == "browser_playwright":
                result = await tool.execute(arguments)
                if result.success:
                    msg = result.message or ""
                    if self._artifact_logger:
                        duration_ms = int((time.time() - start_time) * 1000)
                        self._artifact_logger.log_tool_call(
                            call_id=call_id,
                            tool_name="browser_playwright",
                            arguments=arguments,
                            result=msg[:12000],
                            duration_ms=duration_ms,
                            success=True,
                        )
                    return msg
                err = result.error or result.message or "browser_playwright failed"
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="browser_playwright",
                        arguments=arguments,
                        result=err[:8000],
                        duration_ms=duration_ms,
                        success=False,
                        error=err,
                    )
                return err

            elif name in {
                "browser_navigate",
                "browser_click",
                "browser_type",
                "browser_scroll",
                "browser_screenshot",
            }:
                result = await tool.execute(arguments)
                output = result.message or ""
                if not result.success:
                    err = result.error or output or f"{name} failed"
                    if not output.startswith("Error:"):
                        output = f"Error: {err}"
                self._append_browser_observation(
                    name,
                    output,
                    error=result.error if not result.success else None,
                    browser_meta=self._browser_observation_meta(result.data),
                    plan_step=arguments.get("plan_step"),
                )
                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name=name,
                        arguments=arguments,
                        result=output[:12000],
                        duration_ms=duration_ms,
                        success=result.success,
                        error=result.error if not result.success else None,
                    )
                return output

            elif name == "github":
                action = arguments.get("action", "")

                from ..tools.github import GitHubToolInput, GitHubAction

                _pn = arguments.get("pr_number")
                pr_num: int | None
                try:
                    pr_num = int(_pn) if _pn is not None else None
                except (TypeError, ValueError):
                    pr_num = None
                _assign = arguments.get("assignees")
                if _assign is not None and not isinstance(_assign, list):
                    _assign = None
                _linked = arguments.get("linked_issues")
                _linked_nums: list[int] | None = None
                if isinstance(_linked, list):
                    tmp: list[int] = []
                    for x in _linked:
                        try:
                            tmp.append(int(x))
                        except (TypeError, ValueError):
                            pass
                    if tmp:
                        _linked_nums = tmp
                input_data = GitHubToolInput(
                    action=GitHubAction(action),
                    branch_name=arguments.get("branch_name"),
                    base_branch=arguments.get("base_branch"),
                    commit_message=arguments.get("commit_message"),
                    files=arguments.get("files"),
                    pr_title=arguments.get("pr_title"),
                    pr_description=arguments.get("pr_description"),
                    task_description=arguments.get("task_description"),
                    pr_number=pr_num,
                    merge_method=arguments.get("merge_method") or "squash",
                    draft=bool(arguments.get("draft", False)),
                    assignees=[str(x) for x in _assign] if _assign else None,
                    linked_issues=_linked_nums,
                    repo_path=arguments.get("repo_path", self.working_directory or "."),
                )

                result = await tool.execute(input_data)

                if result.success:
                    output = f"GitHub action '{action}' completed successfully.\n{result.message}"
                    if result.pr_url:
                        output += f"\nPR URL: {result.pr_url}"
                else:
                    output = f"GitHub action failed: {result.message}"

                if self._artifact_logger:
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._artifact_logger.log_tool_call(
                        call_id=call_id,
                        tool_name="github",
                        arguments=arguments,
                        result=output[:5000],
                        duration_ms=duration_ms,
                        success=result.success,
                    )

                return output

            elif name == "project_memory":
                action = arguments.get("action", "list_projects")
                from ..integrations.project_memory import (
                    get_project_memory,
                    MemoryCategory,
                )

                pm = get_project_memory()

                if action == "create_project":
                    pname = arguments.get("project_name", "")
                    if not pname:
                        return "Error: project_name is required"
                    proj = pm.create_project(
                        name=pname,
                        description=arguments.get("project_description", ""),
                        tech_stack=arguments.get("tech_stack", []),
                        project_id=arguments.get("project_id"),
                    )
                    return f"Project created: {proj.name} (id={proj.id})"

                elif action == "list_projects":
                    projects = pm.list_projects()
                    if not projects:
                        return "No projects yet. Use create_project to start one."
                    lines = ["Projects:"]
                    for p in projects:
                        lines.append(f"  • [{p.id}] {p.name} — {p.description[:80]}")
                    return "\n".join(lines)

                elif action == "remember":
                    pid = arguments.get("project_id", "")
                    if not pid:
                        return "Error: project_id is required"
                    title = arguments.get("title", "")
                    content = arguments.get("content", "")
                    if not title or not content:
                        return "Error: title and content are required"
                    cat_str = arguments.get("category", "context")
                    try:
                        cat = MemoryCategory(cat_str)
                    except ValueError:
                        cat = MemoryCategory.CONTEXT
                    entry = pm.add_entry(
                        project_id=pid,
                        category=cat,
                        title=title,
                        content=content,
                        tags=arguments.get("tags", []),
                        importance=int(arguments.get("importance", 5)),
                        session_id=getattr(self, "session_id", None),
                    )
                    return f"Memory stored: [{entry.category.upper()}] {entry.title} (id={entry.id})"

                elif action == "search":
                    pid = arguments.get("project_id", "")
                    query = arguments.get("query", "")
                    if not pid or not query:
                        return "Error: project_id and query are required"
                    results = pm.search(
                        pid,
                        query,
                        top_k=int(arguments.get("top_k", 5)),
                    )
                    if not results:
                        return "No relevant memories found."
                    lines = [f"Found {len(results)} relevant memories:"]
                    for r in results:
                        e = r["entry"]
                        lines.append(
                            f"  [{e['category'].upper()}] {e['title']} (score={r['score']})\n"
                            f"    {e['content'][:200]}"
                        )
                    return "\n".join(lines)

                elif action == "get_context":
                    pid = arguments.get("project_id", "")
                    query = arguments.get("query", "")
                    if not pid:
                        return "Error: project_id is required"
                    ctx = pm.get_context_for_task(pid, query)
                    return ctx

                elif action == "list":
                    pid = arguments.get("project_id", "")
                    if not pid:
                        return "Error: project_id is required"
                    entries = pm.list_entries(pid)
                    if not entries:
                        return f"No memories for project '{pid}' yet."
                    lines = [f"Memories for {pid} ({len(entries)} entries):"]
                    for e in sorted(entries, key=lambda x: -x.importance):
                        lines.append(
                            f"  [{e.category.upper()} ★{e.importance}] {e.title}: {e.content[:120]}"
                        )
                    return "\n".join(lines)

                elif action == "delete_entry":
                    eid = arguments.get("entry_id", "")
                    if not eid:
                        return "Error: entry_id is required"
                    removed = pm.delete_entry(eid)
                    return f"Entry {'deleted' if removed else 'not found'}: {eid}"

                else:
                    return f"Unknown project_memory action: {action}"

            elif name == "project_plan":
                action = arguments.get("action", "list")
                from ..integrations.hierarchical_planner import get_planner

                planner = get_planner()

                if action == "create":
                    pid = arguments.get("project_id", "")
                    goal = arguments.get("goal", "")
                    if not pid or not goal:
                        return "Error: project_id and goal are required"
                    # Optionally get project context from memory
                    try:
                        from ..integrations.project_memory import get_project_memory

                        pm = get_project_memory()
                        ctx = (
                            pm.get_context_for_task(pid, goal, max_tokens=400)
                            if pm.get_project(pid)
                            else ""
                        )
                    except Exception:
                        ctx = ""
                    plan = await planner.create_plan(
                        project_id=pid,
                        goal=goal,
                        milestones=arguments.get("milestones"),
                        working_dir=arguments.get(
                            "working_dir", self.working_directory or "."
                        ),
                        project_context=ctx,
                    )
                    lines = [
                        f"Plan created: {plan.id}",
                        f"Goal: {plan.goal}",
                        f"Milestones ({len(plan.milestones)}):",
                    ]
                    for m in plan.milestones:
                        lines.append(
                            f"  {m.index + 1}. {m.name} (~{m.estimated_hours}h) — {m.description[:100]}"
                        )
                    return "\n".join(lines)

                elif action == "get":
                    plan_id = arguments.get("plan_id", "")
                    if not plan_id:
                        return "Error: plan_id is required"
                    plan = planner.get_plan(plan_id)
                    if not plan:
                        return f"Plan '{plan_id}' not found"
                    prog = plan.progress()
                    lines = [
                        f"Plan: {plan.id} | Status: {plan.status}",
                        f"Goal: {plan.goal}",
                        f"Progress: {prog['completed']}/{prog['total']} milestones ({prog['percent']}%)",
                        "",
                    ]
                    for m in plan.milestones:
                        r = plan.results.get(m.id)
                        status = r.status.value if r else "pending"
                        icon = {
                            "completed": "✅",
                            "failed": "❌",
                            "running": "🔄",
                            "pending": "⏳",
                        }.get(status, "⏳")
                        lines.append(f"  {icon} {m.index + 1}. {m.name}")
                        if r and r.error:
                            lines.append(f"      Error: {r.error[:120]}")
                    return "\n".join(lines)

                elif action == "list":
                    pid = arguments.get("project_id")
                    plans = planner.list_plans(pid)
                    if not plans:
                        return "No project plans found."
                    lines = ["Project plans:"]
                    for p in plans:
                        prog = p.progress()
                        lines.append(
                            f"  [{p.id}] {p.goal[:60]} — {prog['completed']}/{prog['total']} done ({p.status})"
                        )
                    return "\n".join(lines)

                elif action == "execute":
                    plan_id = arguments.get("plan_id", "")
                    if not plan_id:
                        return "Error: plan_id is required"
                    plan = planner.get_plan(plan_id)
                    if not plan:
                        return f"Plan '{plan_id}' not found"
                    # Get session manager from callbacks
                    sm = self.callbacks.get("session_manager")
                    cm = self.callbacks.get("connection_manager")
                    if not sm or not cm:
                        return (
                            "Cannot execute plan from within an agent tool — "
                            "use the POST /api/project-plans/{plan_id}/execute endpoint instead."
                        )
                    updated = await planner.execute_plan(plan_id, sm, cm)
                    prog = updated.progress()
                    return (
                        f"Execution finished. Status: {updated.status}. "
                        f"{prog['completed']}/{prog['total']} milestones completed."
                    )

                elif action == "retry_milestone":
                    plan_id = arguments.get("plan_id", "")
                    mid = arguments.get("milestone_id", "")
                    if not plan_id or not mid:
                        return "Error: plan_id and milestone_id are required"
                    ok = planner.retry_milestone(plan_id, mid)
                    return f"Milestone reset to pending: {ok}"

                elif action == "delete":
                    plan_id = arguments.get("plan_id", "")
                    if not plan_id:
                        return "Error: plan_id is required"
                    ok = planner.delete_plan(plan_id)
                    return f"Plan {'deleted' if ok else 'not found'}: {plan_id}"

                else:
                    return f"Unknown project_plan action: {action}"

            elif name == "ui_test":
                from ..tools.browser.ui_tester import UITestRunner, steps_from_spec

                suite_name = arguments.get("suite_name", "UI Test")
                url = arguments.get("url", "")
                raw_steps = arguments.get("steps", [])
                threshold = float(arguments.get("threshold_percent", 0.5))

                if not url:
                    return "Error: url is required for ui_test"
                if not raw_steps:
                    return "Error: steps list is required for ui_test"

                steps = steps_from_spec(raw_steps)
                runner = UITestRunner(
                    working_dir=self.working_directory or ".",
                    headless=True,
                )
                result = await runner.run(
                    suite_name=suite_name,
                    start_url=url,
                    steps=steps,
                    threshold_percent=threshold,
                )
                # Build a readable report
                lines = [result.summary(), ""]
                for sr in result.steps:
                    icon = "✅" if sr.passed else "❌"
                    err = f" — {sr.error}" if sr.error else ""
                    lines.append(
                        f"  {icon} [{sr.step_type}] {sr.description}{err} ({sr.duration_ms}ms)"
                    )
                    if sr.diff:
                        lines.append(
                            f"       Visual diff: {sr.diff.get('changed_percent')}% changed "
                            f"(threshold {sr.diff.get('threshold_percent')}%)"
                        )
                if result.js_errors:
                    lines.append(f"\nJS errors: {'; '.join(result.js_errors[:3])}")
                return "\n".join(lines)

            elif name == "visual_regression":
                action = arguments.get("action", "list_baselines")
                vr_name = arguments.get("name", "")
                b64 = arguments.get("screenshot_b64", "")
                url = arguments.get("url", "")
                threshold = arguments.get("threshold_percent")
                from ..tools.browser.visual_regression import get_engine

                engine = get_engine(self.working_directory or ".")

                if action == "list_baselines":
                    baselines = engine.list_baselines()
                    if not baselines:
                        return "No baselines stored yet. Capture a screenshot with set_baseline=True first."
                    lines = ["Stored baselines:"]
                    for b in baselines:
                        lines.append(
                            f"  • {b.get('name')} — {b.get('width')}×{b.get('height')} @ {b.get('captured_at', '')[:19]}"
                        )
                    return "\n".join(lines)

                elif action == "set_baseline":
                    if not vr_name:
                        return "Error: name is required"
                    if not b64:
                        return "Error: screenshot_b64 is required"
                    rec = engine.save_screenshot_b64(
                        vr_name, b64, url=url, set_as_baseline=True
                    )
                    return f"Baseline set for '{vr_name}' ({rec.width}×{rec.height})"

                elif action == "compare":
                    if not vr_name:
                        return "Error: name is required"
                    if not b64:
                        return "Error: screenshot_b64 is required"
                    try:
                        diff = engine.compare_b64(
                            vr_name, b64, threshold_percent=threshold
                        )
                        icon = "✅ PASS" if diff.passed else "❌ FAIL"
                        return (
                            f"{icon} — '{vr_name}': {diff.changed_percent:.3f}% pixels changed "
                            f"(threshold {diff.threshold_percent}%). "
                            f"Changed pixels: {diff.changed_pixels}."
                            + (
                                f"\nDiff image: {diff.diff_path}"
                                if diff.diff_path
                                else ""
                            )
                        )
                    except FileNotFoundError as e:
                        return f"Error: {e}"

                elif action == "get_history":
                    if not vr_name:
                        return "Error: name is required"
                    hist = engine.get_history(vr_name)
                    return json.dumps(hist, indent=2)

                elif action == "delete_baseline":
                    if not vr_name:
                        return "Error: name is required"
                    removed = engine.delete_baseline(vr_name)
                    return f"Baseline for '{vr_name}' {'deleted' if removed else 'not found'}"

                else:
                    return f"Unknown visual_regression action: {action}"

            elif name == "monitor":
                action = arguments.get("action", "status")
                from ..integrations.monitor import (
                    check_app_health,
                    fetch_app_logs,
                    get_status,
                    register_app,
                    start_monitor,
                    stop_monitor,
                    MonitoredApp,
                )

                if action == "status":
                    return json.dumps(get_status(), indent=2)
                elif action == "health_check":
                    url = arguments.get("url", "")
                    if not url:
                        return "Error: url is required for health_check"
                    result = await check_app_health(url)
                    return json.dumps(result, indent=2)
                elif action == "fetch_logs":
                    platform = arguments.get("platform", "docker")
                    config = arguments.get("config", {})
                    lines = int(arguments.get("lines", 50))
                    logs = await fetch_app_logs(platform, config, lines)
                    return logs or "(no logs returned)"
                elif action == "register":
                    app_cfg = MonitoredApp(
                        name=arguments.get("name", "app"),
                        health_url=arguments.get("health_url", ""),
                        platform=arguments.get("platform", "generic"),
                        platform_config=arguments.get("platform_config", {}),
                        check_interval_seconds=int(arguments.get("interval", 60)),
                        failure_threshold=int(arguments.get("failure_threshold", 3)),
                        session_id=self.session_id
                        if hasattr(self, "session_id")
                        else None,
                    )
                    register_app(app_cfg)
                    start_monitor()
                    return f"Registered {app_cfg.name} and started monitor"
                elif action == "start":
                    start_monitor()
                    return "Monitor started"
                elif action == "stop":
                    stop_monitor()
                    return "Monitor stopped"
                else:
                    return f"Unknown monitor action: {action}"

            elif name == "env_parity":
                action = arguments.get("action", "diff")
                project_root = arguments.get(
                    "project_root", self.working_directory or "."
                )
                from ..integrations.env_parity import (
                    generate_dockerfile,
                    generate_env_example,
                    generate_docker_compose,
                    diff_environments,
                )

                if action == "diff":
                    env_file = arguments.get("env_file", ".env")
                    import os as _os

                    prod_env = arguments.get("production_env")
                    result = diff_environments(
                        local_env_file=str(Path(project_root) / env_file),
                        production_env=prod_env,
                    )
                    return json.dumps(result, indent=2)
                elif action == "generate_dockerfile":
                    content, path = generate_dockerfile(
                        project_root,
                        project_type=arguments.get("project_type", "auto"),
                        frontend_dir=arguments.get("frontend_dir", "frontend"),
                        requirements_file=arguments.get(
                            "requirements_file", "requirements.txt"
                        ),
                        port=arguments.get("port"),
                        health_path=arguments.get("health_path", "/health"),
                        output_path=arguments.get("output_path"),
                    )
                    return f"Dockerfile generated at {path}:\n\n{content}"
                elif action == "generate_env_example":
                    content, path = generate_env_example(
                        project_root,
                        source_env_file=arguments.get("source_env_file", ".env"),
                        output_path=arguments.get("output_path"),
                        include_current_values=arguments.get(
                            "include_current_values", False
                        ),
                    )
                    return f".env.example generated at {path}:\n\n{content}"
                elif action == "generate_docker_compose":
                    content, path = generate_docker_compose(
                        project_root,
                        port=arguments.get("port"),
                        include_redis=arguments.get("include_redis", False),
                        include_postgres=arguments.get("include_postgres", False),
                        output_path=arguments.get("output_path"),
                    )
                    return f"docker-compose.yml generated at {path}:\n\n{content}"
                else:
                    return f"Unknown env_parity action: {action}"

            else:
                return f"Error: Tool '{name}' not implemented"

        except Exception as e:
            error_msg = f"Error executing {name}: {str(e)}"
            if self._artifact_logger:
                duration_ms = int((time.time() - start_time) * 1000)
                self._artifact_logger.log_tool_call(
                    call_id=call_id,
                    tool_name=name,
                    arguments=arguments,
                    result=error_msg,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
            return error_msg
        finally:
            self._current_activity_context = None

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            line = f"[Agent] {message}"
            try:
                print(line)
            except UnicodeEncodeError:
                # Windows consoles often use cp1252; npx/npm output may contain → etc.
                safe = line.encode("ascii", "backslashreplace").decode("ascii")
                print(safe)

    def _synthesize_tool_replies_for_open_assistant_tail(self) -> None:
        """
        If the last message is an assistant with tool_calls, ensure each id has a following tool
        message (OpenAI API requirement). Used after replan / abnormal exits from retry loops.
        """
        msgs = self.llm.conversation
        if not msgs:
            return
        last = msgs[-1]
        if last.role != "assistant" or not last.tool_calls:
            return
        for tc in last.tool_calls:
            if not tc.id:
                continue
            self.llm.add_tool_result(
                tc.id,
                tc.name,
                "Error: no tool output was recorded for this call (session replan or internal abort).",
            )

    def _init_artifact_logger(self, task_id: str, task_description: str) -> None:
        """Initialize the artifact logger for a task."""
        self._artifact_logger = create_artifact_logger(
            base_dir=self.artifact_dir,
            task_id=task_id,
            task_description=task_description,
        )
        self._artifact_logger.set_model(self.llm.config.model)

    def _check_command_safety(self, command: str) -> SafetyViolation | None:
        """Check if a command is safe to execute."""
        violation = self.safety_guard.check_command(command)
        if violation:
            self._log(f"SAFETY VIOLATION: {violation.message}")
            if self._artifact_logger:
                self._artifact_logger.log_tool_call(
                    call_id=str(uuid.uuid4()),
                    tool_name="terminal",
                    arguments={"command": command},
                    result=f"BLOCKED: {violation.message}",
                    success=False,
                    error=violation.message,
                )
        return violation

    def _check_file_edit_safety(
        self,
        file_path: str,
        content: str,
        is_delete: bool = False,
    ) -> SafetyViolation | None:
        """Check if a file edit is safe."""
        lines_changed = len(content.split("\n")) if content else 0
        violation = self.safety_guard.check_file_edit(
            file_path, lines_changed, is_delete
        )
        if violation:
            self._log(f"SAFETY VIOLATION: {violation.message}")
        return violation

    def _check_dependency_safety(
        self, file_path: str, *, change_type: str = "bump"
    ) -> SafetyViolation | None:
        """Check if a dependency change is allowed."""
        violation = self.safety_guard.check_dependency_change(
            file_path, change_type=change_type
        )
        if violation:
            self._log(f"SAFETY VIOLATION: {violation.message}")
        return violation

    def get_artifact_logger(self) -> ArtifactLogger | None:
        """Get the current artifact logger."""
        return self._artifact_logger

    def allow_dependency_changes(self, allow: bool = True) -> None:
        """Allow or disallow dependency changes."""
        self.safety_guard.allow_dependency_changes(allow)
        self._log(f"Dependency changes {'allowed' if allow else 'blocked'}")

    def _get_verification_runner(self):
        """Get or create the verification runner."""
        if self._verification_runner is None and self.working_directory:
            from ..verification.runner import create_verification_runner

            self._verification_runner = create_verification_runner(
                self.working_directory,
                verbose=self.verbose,
            )
        return self._verification_runner

    def _get_git_manager(self):
        """Get or create the git manager."""
        if self._git_manager is None and self.working_directory:
            from ..verification.git_manager import create_git_manager

            self._git_manager = create_git_manager(
                self.working_directory,
                verbose=self.verbose,
            )
        return self._git_manager

    def _get_repair_loop(self):
        """Get or create the repair loop."""
        if self._repair_loop is None and self.working_directory:
            from ..verification.repair import RepairLoop

            runner = self._get_verification_runner()
            git_mgr = self._get_git_manager()
            if runner:
                self._repair_loop = RepairLoop(
                    verification_runner=runner,
                    git_manager=git_mgr,
                    max_iterations=self.max_repair_iterations,
                    verbose=self.verbose,
                )
        return self._repair_loop

    async def run_verification(
        self,
        suite: VerificationSuite | None = None,
        task_id: str = "default",
    ):
        """
        Run verification checks on the current working directory.

        Args:
            suite: Optional verification suite. If not provided, uses Python defaults.
            task_id: Task identifier for tracking.

        Returns:
            VerificationResult with check outcomes.
        """
        runner = self._get_verification_runner()
        if not runner:
            self._log("No verification runner available (no working directory)")
            return None

        if suite is None:
            suite = create_auto_verification_suite(self.working_directory or ".")

        self._log("Running verification checks...")
        result = await runner.run_suite(suite, task_id)

        if result.passed:
            self._log(
                f"Verification passed: {result.checks_passed}/{result.total_checks} checks"
            )
        else:
            self._log(
                f"Verification failed: {len(result.blocking_failures)} blocking failures"
            )
            for failure_id in result.blocking_failures:
                for check_result in result.check_results:
                    if check_result.check_id == failure_id:
                        self._log(f"  - {failure_id}: {check_result.message}")

        return result

    async def run_repair_loop(
        self,
        suite: VerificationSuite | None = None,
        task_id: str = "default",
        repair_fn=None,
    ):
        """
        Run verification with automatic repair attempts.

        Args:
            suite: Optional verification suite.
            task_id: Task identifier.
            repair_fn: Optional custom repair function.

        Returns:
            RepairResult with repair outcomes.
        """
        repair_loop = self._get_repair_loop()
        if not repair_loop:
            self._log("No repair loop available (no working directory)")
            return None

        if suite is None:
            suite = create_auto_verification_suite(self.working_directory or ".")

        self._log("Running repair loop...")
        result = await repair_loop.run(suite, task_id, repair_fn)

        if result.status.value == "success":
            self._log(f"Repair succeeded after {result.total_attempts} attempt(s)")
        elif result.status.value == "escalated":
            self._log(f"Repair escalated after {result.total_attempts} attempt(s)")
            self._log(f"Remaining issues: {result.remaining_issues}")

        return result

    async def create_checkpoint(
        self, checkpoint_id: str, message: str = "Agent checkpoint"
    ):
        """Create a git checkpoint for potential rollback."""
        git_mgr = self._get_git_manager()
        if not git_mgr:
            return None
        return await git_mgr.create_checkpoint(checkpoint_id, message)

    async def rollback_to_checkpoint(self, checkpoint_id: str, hard: bool = False):
        """Rollback to a previous checkpoint."""
        git_mgr = self._get_git_manager()
        if not git_mgr:
            return None
        return await git_mgr.rollback_to_checkpoint(checkpoint_id, hard)

    async def get_diff(self):
        """Get the current git diff."""
        git_mgr = self._get_git_manager()
        if not git_mgr:
            return None
        return await git_mgr.get_diff()

    # Memory System Methods (Phase 4)

    def get_symbol_index(self) -> SymbolIndex:
        """Get or create the symbol index."""
        if self._symbol_index is None:
            workspace = self.working_directory or "."
            self._symbol_index = create_symbol_index(workspace)
        return self._symbol_index

    def get_vector_store(self) -> VectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            persist_path = None
            if self.working_directory:
                persist_path = f"{self.working_directory}/.plodder/vector_store.json"
            self._vector_store = create_vector_store(persist_path=persist_path)
        return self._vector_store

    def get_retrieval_manager(self) -> RetrievalManager:
        """Get or create the retrieval manager."""
        if self._retrieval_manager is None:
            workspace = self.working_directory or "."
            self._retrieval_manager = create_retrieval_manager(
                workspace_path=workspace,
                symbol_index=self.get_symbol_index(),
                vector_store=self.get_vector_store(),
            )
        return self._retrieval_manager

    def get_working_memory(self) -> WorkingMemory:
        """Get or create the working memory."""
        if self._working_memory is None:
            persist_path = None
            if self.working_directory:
                persist_path = f"{self.working_directory}/.plodder/working_memory.json"
            self._working_memory = create_working_memory(
                max_tokens=8000,
                persist_path=persist_path,
            )
        return self._working_memory

    def index_workspace(self, force: bool = False) -> dict:
        """
        Index the workspace for code search and retrieval.

        Args:
            force: Force re-indexing even if already indexed

        Returns:
            Statistics about the indexing
        """
        if self._memory_indexed and not force:
            return self.get_retrieval_manager().get_statistics()

        retrieval_mgr = self.get_retrieval_manager()
        stats = retrieval_mgr.index_workspace(force=force)
        self._memory_indexed = True

        self._log(f"Indexed workspace: {stats}")
        return stats

    def search_code(self, query: str, max_results: int = 10) -> list:
        """
        Search the codebase using the retrieval manager.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        if not self._memory_indexed:
            self.index_workspace()

        retrieval_mgr = self.get_retrieval_manager()
        result = retrieval_mgr.retrieve(query, max_results=max_results)
        return result.results

    def find_definition(self, name: str):
        """Find the definition of a symbol by name."""
        if not self._memory_indexed:
            self.index_workspace()

        return self.get_retrieval_manager().find_definition(name)

    def find_similar_code(self, text: str, max_results: int = 5) -> list:
        """Find code similar to the given text."""
        if not self._memory_indexed:
            self.index_workspace()

        return self.get_retrieval_manager().find_similar(text, max_results=max_results)

    def add_to_memory(
        self, content: str, item_type: str = "context", priority: str = "medium"
    ) -> str:
        """
        Add an item to working memory.

        Args:
            content: The content to remember
            item_type: Type of item (plan, constraint, decision, lesson, error, context, goal)
            priority: Priority level (critical, high, medium, low)

        Returns:
            The memory item ID
        """
        from ..memory.working_memory import MemoryItemType, MemoryPriority

        memory = self.get_working_memory()

        type_map = {
            "plan": memory.add_plan,
            "constraint": memory.add_constraint,
            "decision": memory.add_decision,
            "lesson": memory.add_lesson,
            "error": memory.add_error,
            "goal": memory.add_goal,
        }

        priority_map = {
            "critical": MemoryPriority.CRITICAL,
            "high": MemoryPriority.HIGH,
            "medium": MemoryPriority.MEDIUM,
            "low": MemoryPriority.LOW,
        }

        prio = priority_map.get(priority, MemoryPriority.MEDIUM)

        if item_type in type_map:
            return type_map[item_type](content, priority=prio)
        else:
            from ..memory.working_memory import MemoryItem

            item = MemoryItem(
                id="",
                item_type=MemoryItemType.CONTEXT,
                content=content,
                priority=prio,
            )
            return memory.add(item)

    def get_memory_context(self, max_tokens: int = 4000) -> str:
        """Get the current working memory context as a string."""
        return self.get_working_memory().get_context(max_tokens=max_tokens)

    def get_memory_statistics(self) -> dict:
        """Get statistics about the memory system."""
        return {
            "working_memory": self.get_working_memory().get_statistics(),
            "retrieval": self.get_retrieval_manager().get_statistics()
            if self._memory_indexed
            else {},
            "indexed": self._memory_indexed,
            "conversation_memory": self.get_conversation_memory().get_statistics()
            if self._conversation_memory
            else {},
        }

    # Conversation Memory Methods (Phase 18)

    def get_conversation_memory(self) -> ConversationMemory:
        """
        Get or create the conversation memory for cross-session learning.

        Returns:
            ConversationMemory instance
        """
        if self._conversation_memory is None:
            from pathlib import Path

            storage_path = Path.home() / ".plodder" / "conversation_memory.json"
            self._conversation_memory = create_conversation_memory(
                storage_path=str(storage_path),
                max_entries=1000,
            )
        return self._conversation_memory

    def enable_conversation_memory(self, enabled: bool = True) -> None:
        """Enable or disable conversation memory for this agent."""
        self._use_conversation_memory = enabled

    def get_context_from_memory(self, task_description: str) -> str:
        """
        Get relevant context from conversation memory for a task.

        Args:
            task_description: Description of the current task

        Returns:
            Context string with relevant past experiences
        """
        if not self._use_conversation_memory:
            return ""

        memory = self.get_conversation_memory()
        return memory.get_context_for_task(task_description, max_entries=5)

    def add_lesson_to_memory(
        self,
        lesson: str,
        context: str = "",
        importance: str = "medium",
        tags: list[str] | None = None,
    ) -> str:
        """
        Add a lesson learned to conversation memory.

        Args:
            lesson: The lesson content
            context: Context about when this lesson applies
            importance: Importance level (low, medium, high, critical)
            tags: Optional tags for categorization

        Returns:
            Entry ID
        """
        if not self._use_conversation_memory:
            return ""

        importance_map = {
            "low": Importance.LOW,
            "medium": Importance.MEDIUM,
            "high": Importance.HIGH,
            "critical": Importance.CRITICAL,
        }

        memory = self.get_conversation_memory()
        result = memory.add_lesson(
            lesson=lesson,
            context=context or "General",
            importance=importance_map.get(importance, Importance.MEDIUM),
            tags=tags or [],
            session_id=self.state.session_id,
        )
        return result or ""

    def _inject_knowledge_augmentations(self, task_description: str) -> None:
        """Golden curated examples + optional global corpus RAG (env-gated)."""
        # Golden / synthetic few-shot block
        try:
            from ..learning.golden_context import (
                default_golden_path,
                format_golden_context_for_prompt,
            )

            gpath = os.environ.get("GOLDEN_DATA_PATH", "").strip()
            golden = format_golden_context_for_prompt(
                gpath or default_golden_path(),
                max_records=int(os.environ.get("GOLDEN_MAX_RECORDS", "8")),
                max_chars=int(os.environ.get("GOLDEN_MAX_CHARS", "6000")),
            )
            if golden:
                self.llm.add_user_message(golden)
                self._log("Injected golden / curated examples context")
        except Exception as e:
            self._log(f"Golden context skipped: {e}")

        # Global RAG over pre-built corpus
        if os.environ.get("GLOBAL_RAG_ENABLED", "").lower() not in ("1", "true", "yes"):
            return
        try:
            from ..memory.global_corpus import format_global_corpus_block

            block = format_global_corpus_block(
                task_description,
                limit=int(os.environ.get("GLOBAL_RAG_TOP_K", "8")),
                max_chars=int(os.environ.get("GLOBAL_RAG_MAX_CHARS", "8000")),
            )
            if block:
                self.llm.add_user_message(block)
                self._log("Injected global corpus RAG context")
        except Exception as e:
            self._log(f"Global RAG skipped: {e}")

    def add_error_pattern_to_memory(
        self,
        error: str,
        cause: str,
        solution: str,
        tags: list[str] | None = None,
    ) -> str:
        """
        Add an error pattern and its solution to memory.

        Args:
            error: The error message or type
            cause: What caused the error
            solution: How the error was solved
            tags: Optional tags for categorization

        Returns:
            Entry ID
        """
        if not self._use_conversation_memory:
            return ""

        memory = self.get_conversation_memory()
        result = memory.add_error_pattern(
            error=error,
            cause=cause,
            solution=solution,
            tags=tags or [],
            session_id=self.state.session_id,
        )
        return result or ""

    def add_solution_pattern_to_memory(
        self,
        problem: str,
        solution: str,
        code_example: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Add a solution pattern to memory.

        Args:
            problem: Description of the problem
            solution: The solution that worked
            code_example: Optional code example
            tags: Optional tags for categorization

        Returns:
            Entry ID
        """
        if not self._use_conversation_memory:
            return ""

        memory = self.get_conversation_memory()
        result = memory.add_solution_pattern(
            problem=problem,
            solution=solution,
            code_example=code_example,
            tags=tags or [],
            session_id=self.state.session_id,
        )
        return result or ""

    def get_error_solutions(self, error_message: str) -> list[str]:
        """
        Get solutions for similar errors from memory.

        Args:
            error_message: The error message to find solutions for

        Returns:
            List of solution strings
        """
        if not self._use_conversation_memory:
            return []

        memory = self.get_conversation_memory()
        return memory.get_error_solutions(error_message)

    def save_task_summary(
        self,
        task: "TaskState",
        summary: str,
        lessons_learned: list[str] | None = None,
    ) -> str:
        """
        Save a task summary to conversation memory.

        Args:
            task: The completed task
            summary: Summary of what was done
            lessons_learned: Optional list of lessons learned

        Returns:
            Entry ID
        """
        if not self._use_conversation_memory:
            return ""

        memory = self.get_conversation_memory()

        task_summary = TaskSummary(
            task_id=task.task_id,
            session_id=self.state.session_id,
            description=task.goal.description,
            outcome=summary,
            success=task.status == TaskStatus.COMPLETED,
            duration_seconds=int((task.completed_at - task.started_at).total_seconds())
            if task.completed_at and task.started_at
            else 0,
            tools_used=list(task.commands_executed or []),
            files_modified=[fc.path for fc in task.files_changed]
            if task.files_changed
            else [],
            errors_encountered=[task.last_error] if task.last_error else [],
            lessons=lessons_learned or [],
        )

        result = memory.add_task_summary(task_summary)
        return result or ""

    def get_recent_lessons(self, limit: int = 10) -> list[str]:
        """
        Get recent lessons learned from memory.

        Args:
            limit: Maximum number of lessons to return

        Returns:
            List of lesson strings
        """
        if not self._use_conversation_memory:
            return []

        memory = self.get_conversation_memory()
        entries = memory.get_recent_lessons(limit=limit)
        return [e.content for e in entries]

    def search_memory(
        self,
        query: str,
        entry_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search conversation memory.

        Args:
            query: Search query
            entry_type: Optional filter by entry type
            limit: Maximum results to return

        Returns:
            List of matching entries as dicts
        """
        if not self._use_conversation_memory:
            return []

        memory = self.get_conversation_memory()

        type_filter = None
        if entry_type:
            type_map = {
                "task_summary": ConversationEntryType.TASK_SUMMARY,
                "lesson": ConversationEntryType.LESSON_LEARNED,
                "error": ConversationEntryType.ERROR_PATTERN,
                "solution": ConversationEntryType.SOLUTION_PATTERN,
                "preference": ConversationEntryType.USER_PREFERENCE,
                "feedback": ConversationEntryType.FEEDBACK,
            }
            type_filter = type_map.get(entry_type)

        entry_types = [type_filter] if type_filter else None
        entries = memory.search(query, entry_types=entry_types, limit=limit)
        return [e.to_dict() for e in entries]

    def get_reviewer_agent(self, strict_mode: bool = False) -> ReviewerAgent:
        """
        Get or create the reviewer agent.

        Args:
            strict_mode: If True, be stricter about diff discipline

        Returns:
            ReviewerAgent instance
        """
        if self._reviewer_agent is None:
            from ..agents import create_reviewer_agent

            self._reviewer_agent = create_reviewer_agent(
                strict_mode=strict_mode,
                auto_suggest_improvements=True,
            )
        return self._reviewer_agent

    async def review_changes(
        self,
        context: str | None = None,
        task_description: str | None = None,
    ) -> ReviewFeedback:
        """
        Review the current git diff using the reviewer agent.

        Args:
            context: Optional context about the codebase
            task_description: Optional description of what the change is trying to do

        Returns:
            ReviewFeedback with the review results
        """
        git_mgr = self._get_git_manager()
        if not git_mgr:
            from ..agents import ReviewFeedback

            return ReviewFeedback(
                approved=False,
                summary="Cannot review: Git manager not available",
            )

        diff_result = await git_mgr.get_diff()
        if not diff_result or not diff_result.success:
            return ReviewFeedback(
                approved=False,
                summary="Cannot review: Failed to get git diff",
            )

        diff = diff_result.data.get("diff", "")
        if not diff:
            return ReviewFeedback(
                approved=True,
                summary="No changes to review",
                overall_quality_score=10.0,
            )

        reviewer = self.get_reviewer_agent()
        return await reviewer.review_diff(diff, context, task_description)

    def quick_review_changes(self, diff: str) -> tuple[bool, list[str]]:
        """
        Perform a quick, synchronous review of a diff without LLM.

        Args:
            diff: The diff to review

        Returns:
            Tuple of (approved, list of issues)
        """
        reviewer = self.get_reviewer_agent()
        return reviewer.quick_review(diff)

    async def review_before_commit(
        self,
        task_description: str | None = None,
    ) -> tuple[bool, str]:
        """
        Review changes before committing.

        This is a convenience method that reviews the current changes
        and returns whether they should be committed.

        Args:
            task_description: Optional description of the task

        Returns:
            Tuple of (should_commit, review_report)
        """
        feedback = await self.review_changes(task_description=task_description)

        should_commit = feedback.approved and not feedback.has_blocking_issues
        report = feedback.format_report()

        if self._artifact_logger:
            self._artifact_logger.log_tool_call(
                call_id="review",
                tool_name="reviewer_agent",
                arguments={"task_description": task_description},
                result=report[:5000],
                duration_ms=0,
                success=should_commit,
            )

        return should_commit, report

    def get_planner_agent(
        self,
        strategy: Any = None,  # Avoid PlanningStrategy type at top level if possible
    ) -> Any:
        # Import inside to avoid circularity
        from ..agents import create_planner_agent, PlanningStrategy

        strategy = strategy or PlanningStrategy.ITERATIVE
        """
        Get or create the planner agent.
        
        Args:
            strategy: Default planning strategy to use
            
        Returns:
            PlannerAgent instance
        """
        if self._planner_agent is None:
            from ..agents import create_planner_agent

            self._planner_agent = create_planner_agent(
                default_strategy=strategy,
                max_steps=50,
                include_verification_steps=True,
            )
        return self._planner_agent

    async def create_plan(
        self,
        task: TaskState | None = None,
        context: str | None = None,
        strategy: Any = None,
    ) -> Any:
        from ..agents import PlanningResult

        """
        Create an execution plan for a task using the planner agent.
        
        Args:
            task: The task to plan for (uses current task if None)
            context: Optional context about the codebase
            strategy: Optional strategy override
            
        Returns:
            PlanningResult with the plan and analysis
        """
        task_to_plan = task or self.state.current_task
        if not task_to_plan:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No task provided and no current task set",
            )

        planner = self.get_planner_agent()
        result = await planner.create_plan(task_to_plan, context, strategy)

        if result.success and result.plan:
            self.state.current_plan = result.plan

            if self._artifact_logger:
                self._artifact_logger.log_tool_call(
                    call_id="plan",
                    tool_name="planner_agent",
                    arguments={
                        "task": task_to_plan.goal.description,
                        "strategy": strategy.value if strategy else "iterative",
                    },
                    result=result.reasoning[:5000],
                    duration_ms=0,
                    success=True,
                )

        return result

    async def analyze_task(
        self,
        task: TaskState | None = None,
        context: str | None = None,
    ) -> dict:
        """
        Analyze a task without creating a full plan.

        Args:
            task: The task to analyze (uses current task if None)
            context: Optional context about the codebase

        Returns:
            Dictionary with task analysis
        """
        task_to_analyze = task or self.state.current_task
        if not task_to_analyze:
            return {"error": "No task provided and no current task set"}

        planner = self.get_planner_agent()
        analysis = await planner.analyze_task(task_to_analyze, context)
        return analysis.to_dict()

    async def refine_plan(
        self,
        feedback: str,
    ) -> Any:
        from ..agents import PlanningResult

        """
        Refine the current plan based on feedback.
        
        Args:
            feedback: Feedback on what to improve
            
        Returns:
            PlanningResult with the refined plan
        """
        if not self.state.current_plan:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No current plan to refine",
            )

        if not self.state.current_task:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No current task set",
            )

        planner = self.get_planner_agent()
        result = await planner.refine_plan(
            self.state.current_plan,
            feedback,
            self.state.current_task,
        )

        if result.success and result.plan:
            self.state.current_plan = result.plan

        return result

    async def replan_from_failure(
        self,
        failed_step_id: str,
        error: str,
    ) -> Any:
        from ..agents import PlanningResult

        """
        Create a recovery plan after a step failure.
        
        Args:
            failed_step_id: ID of the step that failed
            error: Error message
            
        Returns:
            PlanningResult with a recovery plan
        """
        if not self.state.current_plan:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No current plan",
            )

        if not self.state.current_task:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning="No current task set",
            )

        failed_step = None
        for step in self.state.current_plan.steps:
            if step.step_id == failed_step_id:
                failed_step = step
                break

        if not failed_step:
            return PlanningResult(
                success=False,
                plan=None,
                analysis=None,
                validation=None,
                reasoning=f"Step {failed_step_id} not found in plan",
            )

        planner = self.get_planner_agent()
        result = await planner.replan_from_failure(
            self.state.current_plan,
            failed_step,
            error,
            self.state.current_task,
        )

        if result.success and result.plan:
            self.state.current_plan = result.plan

        return result

    async def auto_replan_from_repeated_failure(self, error: str) -> bool:
        """Replan the current task after repeated failures using the active plan step."""
        if not self.state.current_task or not self.state.current_plan:
            return False

        failed_step = None
        for step in self.state.current_plan.steps:
            if step.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                failed_step = step
                break

        if not failed_step:
            return False

        try:
            self.mark_plan_step_failed(failed_step.step_id, error)
        except Exception:
            pass

        result = await self.replan_from_failure(failed_step.step_id, error)
        if result.success and result.plan:
            self.add_to_memory(
                f"Auto-replanned after repeated failure on step {failed_step.step_id}",
                item_type="decision",
                priority="high",
            )
            return True
        return False

    def get_next_plan_step(self):
        """
        Get the next step to execute in the current plan.

        Returns:
            The next PlanStep to execute, or None if done/no plan
        """
        if not self.state.current_plan:
            return None

        planner = self.get_planner_agent()
        return planner.get_next_step(self.state.current_plan)

    def mark_plan_step_complete(
        self,
        step_id: str,
        result: str | None = None,
    ) -> None:
        """
        Mark a plan step as completed.

        Args:
            step_id: ID of the step to mark complete
            result: Optional result description
        """
        if not self.state.current_plan:
            return

        planner = self.get_planner_agent()
        self.state.current_plan = planner.mark_step_complete(
            self.state.current_plan,
            step_id,
            result,
        )

    def mark_plan_step_failed(
        self,
        step_id: str,
        error: str,
    ) -> None:
        """
        Mark a plan step as failed.

        Args:
            step_id: ID of the step to mark failed
            error: Error message
        """
        if not self.state.current_plan:
            return

        planner = self.get_planner_agent()
        self.state.current_plan = planner.mark_step_failed(
            self.state.current_plan,
            step_id,
            error,
        )

    def get_plan_progress(self) -> dict:
        """
        Get progress statistics for the current plan.

        Returns:
            Dictionary with progress statistics
        """
        if not self.state.current_plan:
            return {
                "total_steps": 0,
                "completed_steps": 0,
                "progress_percent": 0,
                "has_plan": False,
            }

        planner = self.get_planner_agent()
        progress = planner.get_plan_progress(self.state.current_plan)
        progress["has_plan"] = True
        return progress

    def create_minimal_plan(
        self,
        task: TaskState | None = None,
    ) -> None:
        """
        Create a minimal plan without LLM (for simple tasks).

        Args:
            task: The task to plan for (uses current task if None)
        """
        task_to_plan = task or self.state.current_task
        if not task_to_plan:
            return

        planner = self.get_planner_agent()
        self.state.current_plan = planner.create_minimal_plan(task_to_plan)

    def get_parallel_executor(self) -> ParallelExecutor:
        """
        Get or create the parallel executor.

        Returns:
            ParallelExecutor instance
        """
        if self._parallel_executor is None:

            async def execute_tool(tool_name: str, arguments: dict) -> str:
                return await self._execute_tool(tool_name, arguments)

            self._parallel_executor = create_parallel_executor(
                execute_fn=execute_tool,
                max_concurrent=self._max_parallel_tools,
                fail_fast=False,
            )
        return self._parallel_executor

    def get_batch_caller(self) -> BatchToolCaller:
        """
        Get or create the batch tool caller.

        Returns:
            BatchToolCaller instance for fluent API
        """
        if self._batch_caller is None:

            async def execute_tool(tool_name: str, arguments: dict) -> str:
                return await self._execute_tool(tool_name, arguments)

            self._batch_caller = create_batch_caller(
                execute_fn=execute_tool,
                max_concurrent=self._max_parallel_tools,
            )
        return self._batch_caller

    def enable_parallel_execution(self, enabled: bool = True) -> None:
        """
        Enable or disable parallel tool execution.

        Args:
            enabled: Whether to enable parallel execution
        """
        self._enable_parallel_execution = enabled

    def set_max_parallel_tools(self, max_tools: int) -> None:
        """
        Set the maximum number of parallel tool executions.

        Args:
            max_tools: Maximum number of concurrent tool calls
        """
        self._max_parallel_tools = max_tools
        self._parallel_executor = None
        self._batch_caller = None

    async def execute_tools_parallel(
        self,
        tool_calls: list[dict],
    ) -> ParallelExecutionResult:
        """
        Execute multiple tool calls in parallel when possible.

        Analyzes dependencies between calls and executes independent
        calls concurrently for improved performance.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'

        Returns:
            ParallelExecutionResult with all results and timing info
        """
        if not self._enable_parallel_execution or len(tool_calls) <= 1:
            results = []
            for tc in tool_calls:
                started_at = datetime.now(timezone.utc)
                try:
                    result = await self._execute_tool(tc["name"], tc["arguments"])
                    completed_at = datetime.now(timezone.utc)
                    duration_ms = int(
                        (completed_at - started_at).total_seconds() * 1000
                    )
                    success = "Error" not in str(result) and "BLOCKED" not in str(
                        result
                    )
                    results.append(
                        ToolCallResult(
                            call_id=tc.get("id", str(len(results))),
                            tool_name=tc["name"],
                            success=success,
                            result=result,
                            error=None if success else result,
                            started_at=started_at,
                            completed_at=completed_at,
                            duration_ms=duration_ms,
                        )
                    )
                except Exception as e:
                    completed_at = datetime.now(timezone.utc)
                    duration_ms = int(
                        (completed_at - started_at).total_seconds() * 1000
                    )
                    results.append(
                        ToolCallResult(
                            call_id=tc.get("id", str(len(results))),
                            tool_name=tc["name"],
                            success=False,
                            result=None,
                            error=str(e),
                            started_at=started_at,
                            completed_at=completed_at,
                            duration_ms=duration_ms,
                        )
                    )

            total_duration = sum(r.duration_ms for r in results)
            return ParallelExecutionResult(
                results=results,
                total_duration_ms=total_duration,
                parallel_speedup=1.0,
                execution_order=[[r.call_id for r in results]],
            )

        calls = [
            ToolCall.create(
                tool_name=tc["name"],
                arguments=tc["arguments"],
            )
            for tc in tool_calls
        ]

        executor = self.get_parallel_executor()
        return await executor.execute(calls)

    async def batch_read_files(self, paths: list[str]) -> dict[str, str]:
        """
        Read multiple files in parallel.

        Args:
            paths: List of file paths to read

        Returns:
            Dict mapping paths to file contents
        """
        tool_calls = [
            {"name": "editor", "arguments": {"action": "read_file", "path": path}}
            for path in paths
        ]

        result = await self.execute_tools_parallel(tool_calls)

        contents = {}
        for i, path in enumerate(paths):
            if i < len(result.results):
                r = result.results[i]
                contents[path] = r.result if r.success else f"Error: {r.error}"
            else:
                contents[path] = "Error: No result"

        return contents

    async def batch_search(
        self,
        patterns: list[str],
        path: str,
    ) -> dict[str, str]:
        """
        Search for multiple patterns in parallel.

        Args:
            patterns: List of search patterns
            path: Directory to search in

        Returns:
            Dict mapping patterns to search results
        """
        tool_calls = [
            {
                "name": "editor",
                "arguments": {"action": "search", "pattern": pattern, "path": path},
            }
            for pattern in patterns
        ]

        result = await self.execute_tools_parallel(tool_calls)

        results = {}
        for i, pattern in enumerate(patterns):
            if i < len(result.results):
                r = result.results[i]
                results[pattern] = r.result if r.success else f"Error: {r.error}"
            else:
                results[pattern] = "Error: No result"

        return results

    async def batch_terminal_commands(
        self,
        commands: list[str],
        working_directory: str | None = None,
    ) -> list[ToolCallResult]:
        """
        Execute multiple terminal commands.

        Note: Commands are analyzed for dependencies and may be
        executed sequentially if they depend on each other.

        Args:
            commands: List of shell commands
            working_directory: Optional working directory

        Returns:
            List of ToolCallResult for each command
        """
        tool_calls = [
            {
                "name": "terminal",
                "arguments": {
                    "command": cmd,
                    **(
                        {"working_directory": working_directory}
                        if working_directory
                        else {}
                    ),
                },
            }
            for cmd in commands
        ]

        result = await self.execute_tools_parallel(tool_calls)
        return result.results

    def get_parallel_execution_stats(self) -> dict:
        """
        Get statistics about parallel execution.

        Returns:
            Dict with parallel execution configuration and stats
        """
        return {
            "enabled": self._enable_parallel_execution,
            "max_parallel_tools": self._max_parallel_tools,
            "executor_initialized": self._parallel_executor is not None,
            "batch_caller_initialized": self._batch_caller is not None,
        }

    def _emit_playwright_browser_event(self, payload: dict[str, Any]) -> None:
        """Forward Playwright live rows to the dashboard (session wires on_browser_event → WebSocket)."""
        fn = (self.callbacks or {}).get("on_browser_event")
        if not fn:
            return
        try:
            fn(payload)
        except Exception:
            pass

    async def _trigger_callback(self, name: str, *args, **kwargs) -> None:
        """Trigger a callback if it exists, awaiting it if it's a coroutine."""
        if name in self.callbacks:
            callback = self.callbacks[name]
            try:
                import asyncio

                if asyncio.iscoroutine(callback) or inspect.iscoroutinefunction(
                    callback
                ):
                    await callback(*args, **kwargs)
                else:
                    # If it's a lambda or function that returns a coroutine, handle it
                    result = callback(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
                self._log(f"Error in callback '{name}': {e}")

    async def _emit_step_started_contract_event(
        self, task: TaskState, step_index: int, step: Any
    ) -> None:
        if not runtime_contracts_enabled() or not self.working_directory:
            return
        try:
            await emit_step_started(
                workspace=self.working_directory,
                session_id=self.state.session_id,
                task_id=task.task_id,
                step_index=step_index,
                step=step,
            )
        except Exception as e:
            self._log(f"Runtime step.started emission skipped: {e}")

    async def _emit_step_completed_contract_event(
        self, task: TaskState, step_index: int, step: Any
    ) -> None:
        if not runtime_contracts_enabled() or not self.working_directory:
            return
        try:
            await emit_step_completed(
                workspace=self.working_directory,
                session_id=self.state.session_id,
                task_id=task.task_id,
                step_index=step_index,
                step=step,
            )
        except Exception as e:
            self._log(f"Runtime step.completed emission skipped: {e}")

    @staticmethod
    def _sanitize_tool_args_for_log(arguments: dict[str, Any]) -> dict[str, Any]:
        """Shrink tool arguments for JSONL (avoid huge file bodies in session_events)."""
        out: dict[str, Any] = {}
        for k, v in arguments.items():
            if k in ("content", "body", "data", "patch", "script"):
                if isinstance(v, str) and len(v) > 800:
                    out[k] = f"<{len(v)} chars>"
                else:
                    out[k] = v
            elif isinstance(v, str):
                out[k] = v[:2000] if len(v) > 2000 else v
            elif isinstance(v, (int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, list):
                out[k] = v[:40] if len(v) > 40 else v
            elif isinstance(v, dict):
                out[k] = "<dict>"
            else:
                out[k] = str(v)[:500]
        return out

    @staticmethod
    def _is_hard_rate_limit_error(message: str) -> bool:
        """Detect non-recoverable provider quota exhaustion errors."""
        m = (message or "").lower()
        hard_markers = (
            "llm hard quota reached",
            "resource_exhausted",
            "quota exceeded",
            "exceeded your current quota",
            "insufficient_quota",
            "generaterequestsperdayperprojectpermodel-freetier",
            "billing details",
        )
        return any(marker in m for marker in hard_markers)

    def _llm_usage_payload_for_session_event(self) -> dict[str, Any]:
        """Cumulative LLM tokens + rough USD; per-event token/cost delta since last think/observe."""
        llm = getattr(self, "llm", None)
        if not llm or not hasattr(llm, "get_usage_stats"):
            return {}
        stats = llm.get_usage_stats()
        cfg = getattr(llm, "config", None)
        model = getattr(cfg, "model", None) or "unknown"
        pt = int(stats.get("prompt_tokens", 0))
        ct = int(stats.get("completion_tokens", 0))
        tt = int(stats.get("total_tokens", pt + ct))

        prev = getattr(self, "_last_event_llm_tokens", None)
        if isinstance(prev, tuple) and len(prev) == 2:
            prev_pt, prev_ct = int(prev[0]), int(prev[1])
        else:
            prev_pt, prev_ct = 0, 0
        dpt = max(0, pt - prev_pt)
        dct = max(0, ct - prev_ct)
        self._last_event_llm_tokens = (pt, ct)

        cum_cost, cum_method = estimate_llm_cost_usd(model, pt, ct)
        del_cost, del_method = estimate_llm_cost_usd(model, dpt, dct)

        payload: dict[str, Any] = {
            "llm_model": model,
            "llm_prompt_tokens": pt,
            "llm_completion_tokens": ct,
            "llm_total_tokens": tt,
            "llm_prompt_tokens_delta": dpt,
            "llm_completion_tokens_delta": dct,
            "llm_cost_estimate_method_cumulative": cum_method,
            "llm_cost_estimate_method_delta": del_method,
            "llm_estimated_cost_is_approximate": True,
        }
        if cum_cost is not None:
            payload["llm_estimated_cost_usd_cumulative"] = round(cum_cost, 8)
        if del_cost is not None and (dpt > 0 or dct > 0):
            payload["llm_estimated_cost_usd_delta"] = round(del_cost, 10)
        return payload

    def _governance_payload_for_session_event(
        self,
        event: AgentStreamEvent,
        llm_usage_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Emit observe-only governance telemetry signals as additive event metadata."""
        if not self._governance_telemetry_enabled:
            return {}

        signals: list[dict[str, Any]] = []

        if self._governance_emit_budget_signals and llm_usage_payload:
            total_tokens = int(llm_usage_payload.get("llm_total_tokens") or 0)
            status = "tracking"
            detail = "Token tracking active"
            if self._governance_token_budget > 0:
                ratio = total_tokens / float(self._governance_token_budget)
                if ratio >= 1.0:
                    status = "limit_exceeded"
                    detail = "Token budget exceeded"
                elif ratio >= 0.9:
                    status = "near_limit"
                    detail = "Token budget near limit"
                else:
                    status = "within_limit"
                    detail = "Token budget within limit"
            signals.append(
                build_governance_signal(
                    "budget",
                    status=status,
                    counters={
                        "llm_total_tokens": total_tokens,
                        "token_budget": self._governance_token_budget,
                    },
                    detail=detail,
                )
            )

        if self._governance_emit_retry_signals:
            consecutive_failures = int(getattr(self, "_consecutive_failures", 0) or 0)
            same_error_streak = int(getattr(self, "_same_error_streak_n", 0) or 0)
            retry_status = "stable"
            if (
                consecutive_failures > self.max_immediate_retries
                or same_error_streak >= 3
            ):
                retry_status = "elevated"
            elif consecutive_failures > 0 or same_error_streak > 0:
                retry_status = "active"
            signals.append(
                build_governance_signal(
                    "retry",
                    status=retry_status,
                    counters={
                        "consecutive_failures": consecutive_failures,
                        "same_error_streak": same_error_streak,
                        "max_immediate_retries": int(self.max_immediate_retries),
                    },
                    detail="Observe-only retry pressure signal",
                )
            )

        if self._governance_emit_loop_signals:
            current_iteration = int(getattr(self.state, "iteration", 0) or 0)
            max_iterations = int(self.max_iterations)
            no_progress_streak = int(
                getattr(self, "_no_progress_inspection_streak", 0) or 0
            )
            loop_status = "stable"
            if max_iterations > 0 and current_iteration >= int(max_iterations * 0.9):
                loop_status = "near_ceiling"
            if no_progress_streak >= 2:
                loop_status = "elevated"
            signals.append(
                build_governance_signal(
                    "loop",
                    status=loop_status,
                    counters={
                        "iteration": current_iteration,
                        "max_iterations": max_iterations,
                        "no_progress_streak": no_progress_streak,
                        "event_kind": event.kind.value,
                    },
                    detail="Observe-only loop proximity signal",
                )
            )

        normalized = normalize_governance_signals(signals)
        if not normalized:
            return {}
        return {
            "governance_schema": "governance.telemetry.v1",
            "governance_observe_only": True,
            "governance_signals": normalized,
        }

    def _discover_repo_root_for_ops_telemetry(self) -> Path | None:
        env_root = (
            os.environ.get("PLODDER_REPO_ROOT")
            or os.environ.get("MINI_DEVIN_REPO_ROOT")
            or ""
        ).strip()
        if env_root:
            candidate = Path(env_root).expanduser().resolve()
            if candidate.exists():
                return candidate
        here = Path(__file__).resolve()
        for parent in (here.parent, *here.parents):
            if (parent / "pyproject.toml").is_file() and (
                parent / "mini_devin"
            ).is_dir():
                return parent
        return None

    def _resolve_governance_ops_collector(self) -> FileOpsTelemetryCollector | None:
        if self._governance_ops_collector is not None:
            return self._governance_ops_collector
        try:
            root = self._discover_repo_root_for_ops_telemetry()
            if root is None:
                return None
            telemetry_root = root / ".plodder" / "ops" / "telemetry"
            self._governance_ops_collector = FileOpsTelemetryCollector(
                events_file=telemetry_root / "events.jsonl",
                state_file=telemetry_root / "state.json",
                config=telemetry_config_from_env(),
            )
        except Exception:
            self._governance_ops_collector = None
        return self._governance_ops_collector

    def _bridge_governance_signals_to_ops_telemetry(
        self, signals: list[dict[str, Any]]
    ) -> None:
        if not signals:
            return
        collector = self._resolve_governance_ops_collector()
        if collector is None:
            return
        for row in signals:
            if not isinstance(row, dict):
                continue
            try:
                collector.record_governance_signal(
                    str(row.get("signal_type") or "unknown"),
                    str(row.get("status") or "unknown"),
                    counters=row.get("counters")
                    if isinstance(row.get("counters"), dict)
                    else {},
                    tags={
                        "source": "agent.session_event",
                        "session_id": self.state.session_id,
                        "observe_only": bool(row.get("observe_only", True)),
                    },
                )
            except Exception:
                continue

    def _append_session_event(self, event: AgentStreamEvent) -> None:
        wd = self.working_directory
        if not wd:
            return
        flat: dict[str, Any] | None = None
        llm_payload: dict[str, Any] = {}
        if event.kind in (AgentEventKind.STATUS, AgentEventKind.OBSERVATION):
            u = self._llm_usage_payload_for_session_event()
            llm_payload = u if u else {}
            gov_payload = self._governance_payload_for_session_event(event, llm_payload)
            combined: dict[str, Any] = {}
            if llm_payload:
                combined.update(llm_payload)
            if gov_payload:
                combined.update(gov_payload)
                self._bridge_governance_signals_to_ops_telemetry(
                    gov_payload.get("governance_signals")
                    if isinstance(gov_payload.get("governance_signals"), list)
                    else []
                )
            flat = combined if combined else None
        append_standard_event(
            wd,
            event,
            flat_extras=flat,
            session_id=self.state.session_id,
        )

    def _browser_observation_meta(self, payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}

        def _read(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        elements = _read(payload, "interactive_elements", []) or []
        return {
            "action": _read(payload, "action"),
            "url": _read(payload, "url"),
            "title": _read(payload, "title"),
            "detail": _read(payload, "detail"),
            "has_screenshot": bool(_read(payload, "screenshot_base64")),
            "interactive_element_count": len(elements),
            "viewport_width": _read(payload, "viewport_width"),
            "viewport_height": _read(payload, "viewport_height"),
            "action_time_ms": _read(payload, "action_time_ms"),
        }

    def _append_browser_observation(
        self,
        tool: str,
        output: str,
        *,
        error: str | None = None,
        browser_meta: dict[str, Any] | None = None,
        plan_step: Any = None,
    ) -> None:
        preview = (output or "").strip()
        if len(preview) > 8000:
            preview = preview[:8000] + "\n…(truncated)…"
        state_after = self._activity_state.to_meta_snapshot()
        stream_meta: dict[str, Any] = {"plan_step": plan_step}
        ctx = self._current_activity_context
        if ctx and isinstance(ctx.get("activity"), dict):
            stream_meta["activity"] = {**ctx["activity"], "state_after": state_after}
        elif ctx:
            stream_meta.update(ctx)
            if "activity" not in stream_meta:
                stream_meta["activity"] = {"state_after": state_after}
        else:
            stream_meta["activity"] = {"state_after": state_after}
        if browser_meta:
            stream_meta["browser"] = browser_meta
        self._append_session_event(
            AgentStreamEvent(
                kind=AgentEventKind.OBSERVATION,
                tool_name=tool,
                exit_code=0 if error is None else 1,
                output=preview or None,
                error=error,
                legacy_type="observe",
                meta=stream_meta,
            )
        )

    def _workspace_path_set(self) -> set[str]:
        root = Path(self.working_directory or ".").resolve()
        out: set[str] = set()
        if not root.is_dir():
            return out
        skip_dirnames = {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            ".plodder",
        }
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirnames]
            if any(p in skip_dirnames for p in Path(dirpath).parts):
                continue
            for name in filenames:
                if count >= 4000:
                    return out
                p = Path(dirpath) / name
                try:
                    rel = p.relative_to(root).as_posix()
                except ValueError:
                    continue
                out.add(rel)
                count += 1
        return out

    @staticmethod
    def _diff_path_sets(before: set[str], after: set[str]) -> dict[str, list[str]]:
        return {
            "added_sorted": sorted(after - before)[:200],
            "removed_sorted": sorted(before - after)[:200],
        }

    def _should_use_process_execution_terminal(self) -> bool:
        from mini_devin.sandbox.process_execution_sandbox import (
            use_host_process_terminal_for_tooling,
        )

        return use_host_process_terminal_for_tooling()

    def _get_pex_terminal(self) -> Any:
        if self._pex_terminal is None:
            from mini_devin.sandbox.process_execution_sandbox import (
                ProcessExecutionSandbox,
            )

            self._pex_terminal = ProcessExecutionSandbox(self.working_directory or ".")
        return self._pex_terminal

    def _workspace_verify_command_hint(self, written_path: str) -> str:
        root = Path(self.working_directory or ".").resolve()
        hints: list[str] = []
        if written_path.endswith(".py"):
            if (
                (root / "pyproject.toml").is_file()
                or (root / "ruff.toml").is_file()
                or (root / ".ruff.toml").is_file()
            ):
                ex = _sys_for_prompt.executable
                hints.append(
                    f"**Verify (lint)**: `{ex} -m ruff check {written_path}` or `{ex} -m ruff check .`"
                )
                hints.append(
                    f"**Auto-fix (optional)**: next `editor` on this file with **`apply_ruff_fix`: true**, "
                    f'or `{ex} -m ruff check --fix "{written_path}"`'
                )
            if (
                (root / "pytest.ini").is_file()
                or (root / "pyproject.toml").is_file()
                or (root / "tests").is_dir()
            ):
                hints.append(
                    f"**Verify (tests)**: `{_sys_for_prompt.executable} -m pytest -q` (narrow paths if slow)"
                )
        elif (
            written_path.endswith((".ts", ".tsx", ".js", ".jsx"))
            and (root / "package.json").is_file()
        ):
            hints.append(
                "**Verify**: `npm test` or `npm run lint` if defined in `package.json`."
            )
        if hints:
            return "\n".join(f"- {h}" for h in hints)
        return ""

    def _attach_filesystem_observe(
        self,
        output: str,
        fs_before: set[str],
        *,
        tool: str,
        exit_code: int | None,
        plan_step: Any = None,
        written_paths: list[str] | None = None,
    ) -> str:
        fs_after = self._workspace_path_set()
        delta = self._diff_path_sets(fs_before, fs_after)
        recovery_hint: str | None = None
        if tool == "terminal":
            recovery_hint = terminal_recovery_hint(
                exit_code,
                output or "",
                command=getattr(self, "_last_terminal_command", None),
            )
        display_out = (output or "") + (f"\n\n{recovery_hint}" if recovery_hint else "")
        preview = display_out.strip()
        if len(preview) > 8000:
            preview = preview[:8000] + "\n…(truncated)…"
        if tool == "terminal":
            self._activity_state.record_terminal_outcome(exit_code)
        if tool == "editor" and written_paths:
            self._activity_state.record_editor_path(written_paths[0])
        state_after = self._activity_state.to_meta_snapshot()
        stream_meta: dict[str, Any] = {
            "plan_step": plan_step,
            "filesystem_delta": delta,
        }
        ctx = self._current_activity_context
        if ctx and isinstance(ctx.get("activity"), dict):
            stream_meta["activity"] = {**ctx["activity"], "state_after": state_after}
        elif ctx:
            stream_meta.update(ctx)
            if "activity" not in stream_meta:
                stream_meta["activity"] = {"state_after": state_after}
        else:
            stream_meta["activity"] = {"state_after": state_after}
        if recovery_hint:
            stream_meta["recovery_hint"] = recovery_hint
        self._append_session_event(
            AgentStreamEvent(
                kind=AgentEventKind.OBSERVATION,
                tool_name=tool,
                exit_code=exit_code,
                output=preview or None,
                legacy_type="observe",
                meta=stream_meta,
            )
        )
        if self.working_directory and tool in ("terminal", "editor"):
            try:
                Planner.append_checkpoint(
                    self.working_directory,
                    tool,
                    exit_code,
                    str(plan_step) if plan_step is not None else None,
                )
            except Exception as _e:
                self._log(f"PLAN.md checkpoint append skipped: {_e}")
        parts = [
            display_out,
            "",
            "## Observe (filesystem)",
            json.dumps(delta, ensure_ascii=False),
        ]
        for wp in written_paths or []:
            hint = self._workspace_verify_command_hint(wp)
            if hint:
                parts.extend(["", "## Verification suggested", hint])
        if self._workspace_sidecar and (
            delta.get("added_sorted") or delta.get("removed_sorted")
        ):
            self._workspace_sidecar.refresh(blocking=False)
        return "\n".join(parts)

    async def _auto_ruff_check_file(self, rel_path: str) -> str:
        if os.name == "nt" or not rel_path.endswith(".py"):
            return ""
        root = Path(self.working_directory or ".").resolve()
        if not (
            (root / "pyproject.toml").is_file()
            or (root / "ruff.toml").is_file()
            or (root / ".ruff.toml").is_file()
        ):
            return ""
        if importlib.util.find_spec("ruff") is None:
            self._log(
                "Auto-verify (ruff): `ruff` is not installed for this interpreter; "
                "skipping lint (install ruff in the image or add to pyproject optional deps)."
            )
            return ""
        exe = _sys_for_prompt.executable
        cmd = f'"{exe}" -m ruff check "{rel_path}"'
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=90.0)
            text = (out or b"").decode("utf-8", errors="replace").strip()
            self._append_session_event(
                AgentStreamEvent(
                    kind=AgentEventKind.OBSERVATION,
                    tool_name="ruff",
                    exit_code=proc.returncode,
                    legacy_type="auto_verify",
                    meta={"path": rel_path, "auto_verify": "ruff_check"},
                )
            )
            tag = "passed" if proc.returncode == 0 else "failed"
            fix_hint = ""
            if proc.returncode != 0:
                fix_hint = (
                    "\n\n## Optional: clean up with ruff --fix\n"
                    f"- Set **`apply_ruff_fix`: true** on your next `editor` call with `write_file`, `str_replace`, or "
                    f"`apply_patch` for `{rel_path}` so the orchestrator runs `ruff check --fix` after that edit.\n"
                    f'- Or use `terminal`: `{exe} -m ruff check --fix "{rel_path}"` then `{exe} -m ruff check "{rel_path}"`.\n'
                )
            return (
                f"\n\n## Auto-verify (ruff) — {tag}\n```\n{text[:4000]}\n```{fix_hint}"
            )
        except FileNotFoundError as e:
            self._log(f"Auto-verify (ruff): interpreter or ruff not runnable: {e}")
            return ""
        except Exception as e:
            self._log(f"Auto-verify (ruff): non-fatal error: {e}")
            return f"\n\n## Auto-verify (ruff) — error\n{e}"

    async def _run_ruff_fix_file(self, rel_path: str) -> str:
        """Run `ruff check --fix` on one file after a successful Python edit (non-Windows hosts)."""
        if os.name == "nt" or not rel_path.endswith(".py"):
            return ""
        root = Path(self.working_directory or ".").resolve()
        if not (
            (root / "pyproject.toml").is_file()
            or (root / "ruff.toml").is_file()
            or (root / ".ruff.toml").is_file()
        ):
            return ""
        if importlib.util.find_spec("ruff") is None:
            self._log(
                "Ruff auto-fix skipped: `ruff` package not installed for this interpreter."
            )
            return ""
        exe = _sys_for_prompt.executable
        cmd = f'"{exe}" -m ruff check --fix "{rel_path}"'
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            text = (out or b"").decode("utf-8", errors="replace").strip()
            self._append_session_event(
                AgentStreamEvent(
                    kind=AgentEventKind.OBSERVATION,
                    tool_name="ruff_fix",
                    exit_code=proc.returncode,
                    legacy_type="auto_verify",
                    meta={"path": rel_path, "auto_verify": "ruff_fix"},
                )
            )
            tag = "completed" if proc.returncode == 0 else "finished_with_errors"
            return (
                f"\n\n## Ruff auto-fix (`check --fix`) — {tag}\n```\n{text[:4000]}\n```"
            )
        except FileNotFoundError as e:
            self._log(f"Ruff auto-fix: interpreter or ruff not runnable: {e}")
            return ""
        except Exception as e:
            self._log(f"Ruff auto-fix: non-fatal error: {e}")
            return f"\n\n## Ruff auto-fix — error\n{e}"

    async def _update_phase(self, new_phase: AgentPhase) -> None:
        """Update the agent phase."""
        old_phase = self.state.phase
        self.state.phase = new_phase
        self._log(f"Phase transition: {old_phase.value} -> {new_phase.value}")
        await self._trigger_callback("on_phase_change", new_phase.value)

    def _playbook_repo_root(self) -> Path:
        """Prefer workspace ``skills/`` when present; else repo root from env / discovery."""
        wd = self.working_directory
        if wd:
            p = Path(wd).resolve()
            if (p / "skills").is_dir():
                return p
        return discover_repo_root()

    @staticmethod
    def _task_blob_for_skill_match(task: TaskState) -> str:
        parts: list[str] = [task.goal.description or ""]
        if task.goal.acceptance_criteria:
            parts.extend(str(c) for c in task.goal.acceptance_criteria)
        return "\n".join(parts)

    @staticmethod
    def _infer_auto_playbook_tags(text: str) -> list[str]:
        """Map task wording to ``skills/<tag>.md`` stems (order preserved, de-duplicated)."""
        raw = (text or "").lower()
        seen: set[str] = set()
        out: list[str] = []

        def add(tag: str) -> None:
            if tag not in seen:
                seen.add(tag)
                out.append(tag)

        if re.search(r"\b(review|verify|validate|audit)\b", raw, re.I):
            add("code_review")
        if re.search(r"\bcheck\b", raw, re.I) and not re.search(
            r"\bcheckout\b", raw, re.I
        ):
            add("code_review")
        if "code review" in raw or "pull request" in raw or "pull-request" in raw:
            add("code_review")
        if re.search(r"\b(lint|linting|rubric)\b", raw, re.I):
            add("code_review")
        if re.search(r"\b(refactor|restructure)\b", raw, re.I) or re.search(
            r"\b(clean\s*up|cleanup|migrate)\b", raw, re.I
        ):
            add("refactor")
        if re.search(
            r"\b(expedia|booking\.com|agoda|hotel|lodging|reservation|travel booking|hotel booking)\b",
            raw,
            re.I,
        ):
            add("travel_booking")
        return out

    def _apply_auto_injected_playbooks(self, task: TaskState) -> None:
        """
        Reset system prompt to base, then append matching ``skills/*.md`` blocks.

        Emits a single STATUS row listing loaded tags (via :meth:`_append_session_event`).
        """
        base = getattr(self, "_system_prompt_base", None)
        if not isinstance(base, str) or not base:
            return
        self.llm.set_system_prompt(base)

        tags = self._infer_auto_playbook_tags(self._task_blob_for_skill_match(task))
        if not tags:
            return

        root = self._playbook_repo_root()
        loaded: list[str] = []
        chunks: list[str] = []
        for tag in tags:
            body = load_playbook_markdown(root, tag)
            if body:
                loaded.append(tag)
                cap = 12_000
                if len(body) > cap:
                    body = body[:cap] + "\n\n…(truncated)…"
                chunks.append(f"### Skill: `{tag}`\n\n{body}".strip())

        if not loaded:
            return

        suffix = (
            "\n\n## Auto-loaded skill playbooks (this task)\n\n"
            + "\n\n---\n\n".join(chunks)
        )
        self.llm.set_system_prompt(base + suffix)

        summary = "Active skill loaded: " + ", ".join(loaded)
        if self.working_directory:
            self._append_session_event(
                AgentStreamEvent(
                    kind=AgentEventKind.STATUS,
                    role="system",
                    text=summary,
                    legacy_type="skill_autoload",
                    meta={"active_skills": loaded, "repo_root": str(root)},
                )
            )
        else:
            self._log(f"[skills] {summary} (no workspace — event not written to JSONL)")

    def _simple_file_task_satisfied(
        self,
        task: TaskState,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
    ) -> bool:
        """Cheaply complete simple file/content tasks after a successful write/read."""
        if tool_name != "editor":
            return False
        action = tool_args.get("action")
        if action not in {"write_file", "create_file", "read_file"}:
            return False

        description = task.goal.description if task.goal else ""
        file_match = re.search(
            r"(?:file named|file)\s+[`\"']?([A-Za-z0-9._-]+)[`\"']?",
            description,
            re.IGNORECASE,
        )
        exact_match = re.search(
            r"(?:contain exactly|contains exactly|exact content|exactly this text|with exactly(?: this text)?)\s*:?\s*[`\"']?([^`\"'\r\n]+)",
            description,
            re.IGNORECASE,
        )
        if not file_match or not exact_match:
            return False

        expected_path = file_match.group(1).strip()
        expected_content = exact_match.group(1).strip()
        actual_path = str(tool_args.get("path", "")).replace("\\", "/")
        if actual_path != expected_path.replace("\\", "/"):
            return False

        if action in {"write_file", "create_file"}:
            return str(tool_args.get("content", "")).strip() == expected_content

        lines = [line.rstrip("\r") for line in (tool_result or "").splitlines()]
        content_lines = [line for line in lines if not line.startswith("File:")]
        actual_content = "\n".join(content_lines).strip()
        return actual_content == expected_content

    def _progress_guard_message(
        self,
        task: TaskState,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        tool_success: bool,
    ) -> str | None:
        """Nudge the model when it keeps inspecting a clean workspace without edits."""
        if not tool_success:
            self._no_progress_inspection_streak = 0
            return None

        action = str(tool_args.get("action", "") or "")
        command = str(tool_args.get("command", "") or "").strip().lower()
        path = str(tool_args.get("path", "") or "").replace("\\", "/").strip("/")

        is_inspection = False
        if tool_name == "editor":
            is_inspection = action == "list_directory" or (
                action == "read_file" and path.upper() == "PLAN.md"
            )
        elif tool_name == "terminal":
            listing_commands = {
                "dir",
                "ls",
                "ls -la",
                "get-childitem",
                "get-childitem -force",
                "git status",
            }
            is_inspection = command in listing_commands or command.endswith(
                "; get-childitem -force"
            )

        if not is_inspection or task.files_changed:
            self._no_progress_inspection_streak = 0
            self._progress_guard_fired_count = 0
            return None

        self._no_progress_inspection_streak += 1
        threshold = int(os.environ.get("PLODDER_PROGRESS_GUARD_AFTER_INSPECTIONS", "2"))
        if self._no_progress_inspection_streak < threshold:
            return None

        self._no_progress_inspection_streak = 0
        # Escalating hard-stop: each time the soft nudge fails to change behaviour we count it.
        # Weak models often ignore the nudge and loop forever, burning tokens. After a few fires
        # without any file change, stop the run (BLOCKED) instead of inspecting indefinitely.
        self._progress_guard_fired_count += 1
        hard_stop_after = int(os.environ.get("PLODDER_PROGRESS_GUARD_HARD_STOP", "3"))
        if hard_stop_after > 0 and self._progress_guard_fired_count >= hard_stop_after:
            self._progress_guard_force_stop = True
            description = task.goal.description if task.goal else "the task"
            return (
                "Progress guard (hard stop): you have repeatedly inspected an unchanged workspace "
                f"({self._progress_guard_fired_count} guard triggers) without making any edits or "
                "real progress. Ending this run to avoid wasting tokens. If the workspace is empty "
                "or missing the target repository, that is the blocker \u2014 report it. "
                f"Task: {description}"
            )
        description = task.goal.description if task.goal else "the task"
        if self._is_no_edit_verification_task(task):
            return (
                "Progress guard: you have inspected the same clean workspace several times. "
                "Do not edit files for this task. Run the requested verification command now, "
                "or if it already passed with exit code 0, reply TASK COMPLETE with the observed result. "
                f"Task: {description}"
            )
        return (
            "Progress guard: you have inspected the same clean workspace several times without changing files. "
            "Stop listing directories or checking git status. Take the next concrete implementation action now. "
            "For this task, create or edit the requested file using the editor write_file/str_replace/apply_patch tool, "
            f"then verify it. Task: {description}"
        )

    def _is_no_edit_verification_task(self, task: TaskState) -> bool:
        """True when the user explicitly asked for inspection/testing without edits."""
        description = (task.goal.description if task.goal else "").lower()
        no_edit_markers = (
            "do not edit",
            "don't edit",
            "no edit",
            "no-edit",
            "do not modify",
            "don't modify",
            "without editing",
            "without modifying",
        )
        verification_markers = (
            "run ",
            "test",
            "pytest",
            "git status",
            "inspect",
            "smoke test",
            "check",
        )
        return any(marker in description for marker in no_edit_markers) and any(
            marker in description for marker in verification_markers
        )

    def _no_edit_verification_satisfied(
        self,
        task: TaskState,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
    ) -> bool:
        """Auto-complete explicit no-edit verification tasks after a successful test/check command."""
        if not self._is_no_edit_verification_task(task) or tool_name != "terminal":
            return False
        command = str(tool_args.get("command", "") or "").lower()
        result = (tool_result or "").lower()
        if "exit code: 0" not in result:
            return False
        verification_commands = (
            "pytest",
            "unittest",
            "npm test",
            "npm run test",
            "pnpm test",
            "yarn test",
            "ruff",
            "mypy",
            "tsc",
        )
        return any(marker in command for marker in verification_commands)

    def _terminal_task_complete_satisfied(
        self,
        tool_name: str,
        tool_result: str,
    ) -> bool:
        """Treat an explicit TASK COMPLETE terminal echo as completion."""
        if tool_name != "terminal":
            return False
        result = tool_result or ""
        return "TASK COMPLETE" in result.upper() and "Exit code: 0" in result

    def _clone_status_verification_satisfied(
        self,
        task: TaskState,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
    ) -> bool:
        """Auto-complete clone tasks after git status confirms the cloned repo is clean."""
        if tool_name != "terminal":
            return False
        description = (task.goal.description if task.goal else "").lower()
        if "clone" not in description or "git status" not in description:
            return False
        command = str(tool_args.get("command", "") or "").strip().lower()
        working_directory = str(tool_args.get("working_directory", "") or "").strip()
        if command != "git status" or not working_directory:
            return False
        result = (tool_result or "").lower()
        return "exit code: 0" in result and (
            "working tree clean" in result
            or "nothing to commit" in result
            or "up to date with" in result
        )

    def _clone_target_from_successful_command(
        self,
        task: TaskState,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
    ) -> str | None:
        """Return cloned folder path when a clone task has just cloned successfully."""
        if tool_name != "terminal":
            return None
        description = (task.goal.description if task.goal else "").lower()
        if "clone" not in description or "git status" not in description:
            return None
        command = str(tool_args.get("command", "") or "").strip()
        command_lower = command.lower()
        result = tool_result or ""
        if not command_lower.startswith("git clone ") or "Exit code: 0" not in result:
            return None

        parts = command.split()
        if (
            len(parts) >= 4
            and not parts[-1].startswith("http")
            and not parts[-1].startswith("git@")
        ):
            return parts[-1].strip("'\"")

        match = re.search(r"Cloning into ['\"]([^'\"]+)['\"]", result)
        if match:
            return match.group(1)

        repo_url = parts[-1] if parts else ""
        repo_name = repo_url.rstrip("/").rsplit("/", 1)[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        return repo_name or None

    def _clone_target_from_existing_directory_failure(
        self,
        task: TaskState,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
    ) -> str | None:
        """Return repo folder when a repeated clone failed only because it already exists."""
        if tool_name != "terminal":
            return None
        description = (task.goal.description if task.goal else "").lower()
        command = str(tool_args.get("command", "") or "").strip().lower()
        result = tool_result or ""
        if "clone" not in description or not command.startswith("git clone "):
            return None
        match = re.search(
            r"destination path ['\"]([^'\"]+)['\"] already exists",
            result,
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1)

    async def run(self, task: TaskState) -> TaskState:
        """
        Run the agent on a task.

        Args:
            task: The task to execute

        Returns:
            The updated task state
        """
        self.state.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)

        self._ensure_workspace_sidecar()
        wd = self.working_directory
        wl = load_worklog(wd) if wd else None
        resumed = bool(wl and wl.last_task_id == task.task_id and wl.current_plan)
        if resumed:
            self._plan_sent = True
            self._plan_steps = list(wl.current_plan)
            self._current_step_idx = min(
                max(0, wl.current_step_idx),
                max(0, len(self._plan_steps) - 1),
            )
            self._append_session_event(
                AgentStreamEvent(
                    kind=AgentEventKind.STATUS,
                    role="system",
                    text=(
                        f"Resumed `.plodder/worklog.json` for task `{task.task_id}` — "
                        f"step {self._current_step_idx + 1}/{len(self._plan_steps)}."
                    ),
                    legacy_type="worklog_resume",
                    meta={"worklog": wl.to_json_dict()},
                )
            )
        else:
            self._plan_sent = False
            self._plan_steps = []
            self._current_step_idx = 0

        self._tools_used_count = 0
        self._no_tool_streak = 0
        self._session_self_correct_lessons = []
        self._last_event_llm_tokens = None
        self._same_error_streak_tool = None
        self._same_error_streak_fp = None
        self._same_error_streak_n = 0
        self._no_progress_inspection_streak = 0
        self._progress_guard_fired_count = 0
        self._progress_guard_force_stop = False
        self._post_mortem_diagnostic_triggers.clear()
        self._post_mortem_failure_streaks.clear()
        self._post_mortem_recovery_paths.clear()
        self._pending_diagnostic_fp = None
        self._activity_state = AgentActivityState()
        self._current_activity_context = None

        # Initialize artifact logger
        self._init_artifact_logger(task.task_id, task.goal.description)

        # Reset safety guard for new task
        self.safety_guard.reset_all()

        self._log(f"Starting task: {task.goal.description}")
        await self._update_phase(AgentPhase.INTAKE)
        self._apply_auto_injected_playbooks(task)

        # Index + README capture ("self-learning" on repo open)
        try:
            from .workspace_bootstrap import run_workspace_bootstrap

            _boot = run_workspace_bootstrap(self)
            if _boot.get("readme_chars"):
                self._log(
                    f"Workspace bootstrap: README {_boot.get('readme_file', '')} "
                    f"({_boot.get('readme_chars')} chars)"
                )
        except Exception as e:
            self._log(f"Workspace bootstrap skipped: {e}")

        if self.working_directory:
            try:
                Planner.sync_plan_file(self.working_directory, task.goal.description)
                self._append_session_event(
                    AgentStreamEvent(
                        kind=AgentEventKind.STATUS,
                        role="system",
                        legacy_type="task_start",
                        meta={
                            "task_id": task.task_id,
                            "goal": (task.goal.description or "")[:4000],
                        },
                    )
                )
                await self._trigger_callback(
                    "on_message",
                    "Planner: `PLAN.md` updated at workspace root.",
                    is_token=False,
                )
                self._seed_task_working_memory(
                    task,
                    resumed_plan=list(wl.current_plan)
                    if resumed and wl and wl.current_plan
                    else None,
                )
            except Exception as e:
                self._log(f"Planner sync skipped: {e}")

        # Prepend project memory context if this session has a linked project
        _proj_ctx = getattr(self, "_project_context_injection", None)
        if _proj_ctx:
            # Also do a semantic refresh for this specific task
            try:
                _pid = getattr(self, "_project_id", None)
                if _pid:
                    from ..integrations.project_memory import get_project_memory  # type: ignore

                    _pm = get_project_memory()
                    _proj_ctx = _pm.get_context_for_task(_pid, task.goal.description)
            except Exception:
                pass

        _bench_ctx = getattr(self, "_benchmark_context_injection", None)
        _retrieval_ctx = self._build_task_retrieval_context(task)
        _task_desc_for_prompt = self._compact_task_description_for_prompt(
            task.goal.description or ""
        )
        _criteria_for_prompt = self._compact_acceptance_criteria(
            task.goal.acceptance_criteria
        )
        _criteria_omitted = max(
            0, len(task.goal.acceptance_criteria or []) - len(_criteria_for_prompt)
        )
        _criteria_lines = _criteria_for_prompt or ["Complete the task successfully"]
        if _criteria_omitted > 0:
            _criteria_lines.append(
                f"... ({_criteria_omitted} additional criteria omitted for prompt efficiency)"
            )

        # Add task description to conversation
        _context_prefix_parts: list[str] = []
        if _proj_ctx:
            _context_prefix_parts.append(_proj_ctx)
        if _bench_ctx:
            _context_prefix_parts.append(
                "Benchmark lessons for better reliability:\n" + _bench_ctx
            )
        if _retrieval_ctx:
            _context_prefix_parts.append("Task retrieval context:\n" + _retrieval_ctx)
        _proj_prefix = (
            "\n\n" + "\n\n---\n\n".join(_context_prefix_parts) + "\n\n---\n"
            if _context_prefix_parts
            else ""
        )
        _rtc = _runtime_context_block(self.working_directory)
        task_message = f"""{_proj_prefix}{_rtc}

---

Task: {_task_desc_for_prompt}

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in _criteria_lines)}

Working Directory: {self.working_directory or 'current directory'}

Structured planning: read `PLAN.md` at the workspace root; every `terminal` / `editor` call must include **`plan_step`** (e.g. `"STEP-2"`).

IMPORTANT: You MUST use tools to complete this task. Do NOT just write text descriptions. 
Autonomous coding policy: inspect the repository narrowly before editing, make the smallest style-consistent diff, explain reasoning before risky edits, run focused verification after modifications, and retry once you understand any failing test output.
For greenfield website/app/ecommerce work: inspect briefly, then implement. Do not run more than two inspection-only
tool calls before creating or editing the first project file. Break broad requests into the PLAN.md milestones and
complete the smallest usable vertical slice first.
Call a tool (editor or terminal) immediately as your first action."""
        if self._workspace_sidecar:
            try:
                task_message = (
                    task_message
                    + "\n\n## Live workspace index (background watcher)\n\n```text\n"
                    + self._workspace_sidecar.get_snapshot_text(
                        max_lines=1200, max_chars=14_000
                    )
                    + "\n```"
                )
            except Exception as e:
                self._log(f"Workspace tree append skipped: {e}")

        self.llm.add_user_message(task_message)

        if self.working_directory:
            try:
                evs = load_session_events(self.working_directory, max_lines=150)
                if evs:
                    slim: list[dict[str, Any]] = []
                    for e in evs[-45:]:
                        slim.append(
                            {
                                "ts": e.get("ts"),
                                "type": e.get("type"),
                                "tool": e.get("tool"),
                                "plan_step": e.get("plan_step"),
                                "exit_code": e.get("exit_code"),
                                "llm_total_tokens": e.get("llm_total_tokens"),
                                "llm_estimated_cost_usd_cumulative": e.get(
                                    "llm_estimated_cost_usd_cumulative"
                                ),
                            }
                        )
                    self.llm.add_user_message(
                        "Workspace session event tail (resume / continuity):\n"
                        + json.dumps(slim, ensure_ascii=False)
                    )
            except Exception as e:
                self._log(f"Session event replay skipped: {e}")

        # Retrieve relevant context from conversation memory (Phase 18)
        memory_context = ""
        if self._use_conversation_memory:
            try:
                memory_context = self.get_context_from_memory(task.goal.description)
                if memory_context:
                    self._log("Retrieved relevant context from conversation memory")
                    self.llm.add_user_message(
                        f"Here is relevant context from past experiences that may help:\n\n{memory_context}"
                    )
            except Exception as e:
                self._log(f"Warning: Failed to retrieve memory context: {e}")

        self._inject_knowledge_augmentations(task.goal.description)

        _bb_for_run = os.environ.get(
            "PLODDER_BACKBONE_FOR_RUN", ""
        ).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if _bb_for_run:
            await self._async_setup_backbone()
            self._use_backbone_dispatch = True

        try:
            # Main agent loop
            iteration = 0
            while iteration < self.max_iterations:
                iteration += 1
                self.state.iteration = iteration
                self._log(f"Iteration {iteration}/{self.max_iterations}")

                # Progress-guard hard-stop: a stuck inspection loop was detected; end the run
                # instead of burning more tokens re-sending unchanged context.
                if getattr(self, "_progress_guard_force_stop", False):
                    self._log(
                        "Progress guard hard-stop: ending run (repeated inspection without progress)."
                    )
                    self._update_phase(AgentPhase.BLOCKED)
                    break

                # Notify frontend of iteration update
                await self._trigger_callback(
                    "on_iteration", iteration, self.max_iterations
                )

                # Reset per-iteration safety counters
                self.safety_guard.reset_iteration()

                try:
                    # Define token callback
                    async def handle_token(token: str):
                        # The callback handles token messages in the frontend
                        await self._trigger_callback("on_message", token, is_token=True)

                    # Get LLM response with tools and streaming
                    _msgs, _eph = await self._prepare_messages_for_llm_turn()
                    response = await self.llm.complete(
                        tools=self._get_tool_schemas(),
                        tool_choice="auto",
                        stream=True,
                        on_token=handle_token,
                        messages_for_api=_msgs,
                        ephemeral_user_messages=_eph,
                    )
                    # Detect step progression in the message content
                    if response.content:
                        # --- Plan detection: fire on_plan_created when agent produces a numbered plan ---
                        if not getattr(self, "_plan_sent", False):
                            # Look for numbered lists that indicate a plan
                            import re as _re

                            plan_lines = _re.findall(
                                r"^\s*(?:\d+\.|-|\*|•)\s+(.+)",
                                response.content,
                                _re.MULTILINE,
                            )
                            # Only treat as a plan if we have 2+ distinct bullet/number points
                            if len(plan_lines) >= 2:
                                steps = [
                                    l.strip() for l in plan_lines[:20] if l.strip()
                                ]
                                if steps:
                                    self._plan_sent = True
                                    self._plan_steps = steps
                                    self._current_step_idx = 0
                                    await self._trigger_callback(
                                        "on_plan_created", steps
                                    )
                                    await self._trigger_callback(
                                        "on_step_started", 0, steps[0]
                                    )
                                    await self._emit_step_started_contract_event(
                                        task, 0, steps[0]
                                    )
                                    self._persist_worklog(task)

                        # --- Step progression detection when a plan is already active ---
                        if getattr(self, "_plan_sent", False):
                            new_step_reached = -1
                            # Look for patterns like "Moving to milestone 2" or "Step 2:""
                            step_patterns = [
                                r"move to step (\d+)",
                                r"moving to step (\d+)",
                                r"starting step (\d+)",
                                r"now on step (\d+)",
                                r"proceed to step (\d+)",
                                r"step (\d+):",
                                r"milestone (\d+):",
                                r"^\s*(\d+)\.",  # Numbered line
                            ]

                            content_lower = response.content.lower()
                            for pattern in step_patterns:
                                match = _re.search(
                                    pattern, content_lower, _re.MULTILINE
                                )
                                if match:
                                    try:
                                        new_step_reached = (
                                            int(match.group(1)) - 1
                                        )  # 0-indexed
                                        break
                                    except (ValueError, IndexError):
                                        continue

                            if 0 <= new_step_reached < len(self._plan_steps):
                                if new_step_reached > self._current_step_idx:
                                    # Complete previous step
                                    await self._trigger_callback(
                                        "on_step_completed",
                                        self._current_step_idx,
                                        self._plan_steps[self._current_step_idx],
                                    )
                                    await self._emit_step_completed_contract_event(
                                        task,
                                        self._current_step_idx,
                                        self._plan_steps[self._current_step_idx],
                                    )
                                    # Start new step
                                    self._current_step_idx = new_step_reached
                                    await self._trigger_callback(
                                        "on_step_started",
                                        self._current_step_idx,
                                        self._plan_steps[self._current_step_idx],
                                    )
                                    await self._emit_step_started_contract_event(
                                        task,
                                        self._current_step_idx,
                                        self._plan_steps[self._current_step_idx],
                                    )
                                    self._persist_worklog(task)

                    # Handle tool calls
                    if response.tool_calls:
                        await self._update_phase(AgentPhase.EXECUTE)

                        # Add assistant message with tool calls
                        self.llm.add_assistant_message(
                            content=response.content,
                            tool_calls=response.tool_calls,
                        )
                        if (response.content or "").strip() and self.working_directory:
                            self._append_session_event(
                                AgentStreamEvent(
                                    kind=AgentEventKind.STATUS,
                                    role="agent",
                                    text=(response.content or "")[:12000],
                                    legacy_type="think",
                                    meta={"task_id": task.task_id},
                                )
                            )
                        # OpenAI requires every tool_call_id from this assistant message to get a tool
                        # message next, in order — before any new "user" message. Inline self-correction
                        # injects user messages between tools, so disable it when the batch has >1 call.
                        _single_tool_batch = len(response.tool_calls) == 1
                        # Track all tool_call_ids answered for this assistant turn (replenish on mid-batch break)
                        _batch_answered_tool_ids: set[str] = set()
                        _should_run_critic = False
                        
                        import uuid
                        _turn_checkpoint_id = f"pre_edit_{uuid.uuid4().hex[:8]}"
                        git_mgr = self._get_git_manager()
                        if git_mgr:
                            await git_mgr.create_checkpoint(_turn_checkpoint_id, "Auto-checkpoint before tool executions")

                        # Execute each tool
                        for tc in response.tool_calls:
                            self._last_failed_tool = None
                            if "on_tool_start" in self.callbacks:
                                await self._trigger_callback(
                                    "on_tool_start", tc.name, tc.arguments
                                )

                            tool_success = False
                            retry_count = 0
                            final_result = ""
                            # One OpenAI tool message per tool_call_id (retry path may add early; avoid duplicate at end)
                            tool_ids_answered: set[str] = set()
                            # Track total tool uses this run (used to gate TASK COMPLETE)
                            self._tools_used_count = (
                                getattr(self, "_tools_used_count", 0) + 1
                            )
                            self._no_tool_streak = 0  # reset streak on real tool call

                            # Self-correction retry loop
                            while retry_count <= self.max_immediate_retries:
                                self._log(
                                    f"Executing tool {tc.name} ({retry_count}/{self.max_immediate_retries}): {json.dumps(tc.arguments)[:100]}..."
                                )
                                import time

                                start_time = time.time()

                                # For tool execution classification
                                exit_code = None

                                try:
                                    result = await self._execute_tool(
                                        tc.name,
                                        tc.arguments,
                                        thought=response.content,
                                    )
                                except Exception as _tool_ex:
                                    result = f"Error: tool execution raised an exception: {_tool_ex}"
                                    self._log(f"Tool execution error: {_tool_ex}")
                                duration_ms = (time.time() - start_time) * 1000
                                self._log(f"Tool result: {result[:200]}...")
                                final_result = result

                                # Parse exit code from terminal output if available
                                if tc.name == "terminal" and "Exit code:" in result:
                                    try:
                                        # Parse the first integer after "Exit code:" even when
                                        # additional sections (e.g. Observe snapshot) follow.
                                        m = re.search(r"Exit code:\\s*(-?\\d+)", result)
                                        if m:
                                            exit_code = int(m.group(1))
                                    except (ValueError, TypeError):
                                        pass

                                # Classify the result
                                error_type = self._correction_engine.classify_error(
                                    tc.name, result, exit_code
                                )

                                existing_clone_target = (
                                    self._clone_target_from_existing_directory_failure(
                                        task, tc.name, tc.arguments, result
                                    )
                                )
                                if existing_clone_target:
                                    status_args = {
                                        "command": "git status",
                                        "working_directory": existing_clone_target,
                                        "plan_step": "STEP-4",
                                    }
                                    status_result = await self._execute_tool(
                                        "terminal", status_args
                                    )
                                    task.commands_executed.append("git status")
                                    if self._clone_status_verification_satisfied(
                                        task, "terminal", status_args, status_result
                                    ):
                                        result = status_result
                                        final_result = status_result
                                        tc.arguments = status_args
                                        tool_success = True
                                        break

                                if (
                                    self._terminal_task_complete_satisfied(
                                        tc.name, result
                                    )
                                    or self._clone_status_verification_satisfied(
                                        task, tc.name, tc.arguments, result
                                    )
                                    or self._no_edit_verification_satisfied(
                                        task, tc.name, tc.arguments, result
                                    )
                                ):
                                    tool_success = True
                                    break

                                if error_type == ErrorType.SUCCESS:
                                    self._same_error_streak_tool = None
                                    self._same_error_streak_fp = None
                                    self._same_error_streak_n = 0
                                    tool_success = True
                                    if self._pending_diagnostic_fp is not None:
                                        ok_args = (
                                            dict(tc.arguments)
                                            if isinstance(tc.arguments, dict)
                                            else {}
                                        )
                                        self._post_mortem_recovery_paths.append(
                                            infer_recovery_path_summary(
                                                after_error_fingerprint=self._pending_diagnostic_fp,
                                                next_tool_name=tc.name,
                                                next_args=ok_args,
                                            )
                                        )
                                        self._pending_diagnostic_fp = None
                                    # If we had retried, record the successful correction
                                    if retry_count > 0:
                                        self._log(
                                            f"Self-correction successful after {retry_count} retries"
                                        )
                                        # Reset failures and notify UI about success
                                        self._consecutive_failures = 0
                                        await self._trigger_callback(
                                            "on_message",
                                            f"✅ Successfully corrected error in {tc.name}",
                                            is_token=False,
                                        )
                                        lf = self._last_failed_tool
                                        if lf and self._use_conversation_memory:
                                            try:
                                                tname, bad_args, err_txt = lf
                                                good_args = (
                                                    dict(tc.arguments)
                                                    if isinstance(tc.arguments, dict)
                                                    else {"raw": tc.arguments}
                                                )
                                                self._correction_engine.record_correction(
                                                    {
                                                        "name": tname,
                                                        "arguments": bad_args,
                                                    },
                                                    err_txt,
                                                    {
                                                        "name": tname,
                                                        "arguments": good_args,
                                                    },
                                                )
                                                try:
                                                    self.add_to_memory(
                                                        f"Recovered from {tname} failure by using {json.dumps(good_args, default=str)[:280]}",
                                                        item_type="lesson",
                                                        priority="high",
                                                    )
                                                except Exception:
                                                    pass
                                                self.get_conversation_memory().add_error_pattern(
                                                    error=err_txt[:400],
                                                    cause=f"{tname} tool (self-correct)",
                                                    solution=json.dumps(
                                                        good_args, default=str
                                                    )[:500],
                                                    tags=["self_correction", tname],
                                                    session_id=getattr(
                                                        self.state, "session_id", None
                                                    ),
                                                    task_id=task.task_id,
                                                )
                                                self._session_self_correct_lessons.append(
                                                    f"Self-corrected {tname}: {json.dumps(good_args, default=str)[:280]}"
                                                )
                                            except Exception as mem_err:
                                                self._log(
                                                    f"Warning: could not persist self-correction to memory: {mem_err}"
                                                )
                                        self._last_failed_tool = None
                                    break

                                # Tool failed
                                self._log(
                                    f"Tool {tc.name} failed with {error_type.value}"
                                )
                                _args_dict = (
                                    dict(tc.arguments)
                                    if isinstance(tc.arguments, dict)
                                    else {}
                                )
                                _fp = error_fingerprint(tc.name, str(result), exit_code)
                                if (
                                    tc.name == self._same_error_streak_tool
                                    and _fp == self._same_error_streak_fp
                                ):
                                    self._same_error_streak_n += 1
                                else:
                                    self._same_error_streak_tool = tc.name
                                    self._same_error_streak_fp = _fp
                                    self._same_error_streak_n = 1

                                try:
                                    self.add_to_memory(
                                        f"{tc.name} failed: {(str(result) or '')[:800]}",
                                        item_type="error",
                                        priority="high",
                                    )
                                except Exception:
                                    pass

                                if not _single_tool_batch:
                                    self._log(
                                        "Skipping inline self-correction: multiple tool_calls in one turn "
                                        "(API requires all tool results before new user messages)."
                                    )
                                    break

                                if not self._correction_engine.should_retry(
                                    error_type, retry_count
                                ):
                                    self._log(
                                        "Max retries reached or error not retryable immediately"
                                    )
                                    break

                                try:
                                    self.add_to_memory(
                                        f"Retrying {tc.name} after {error_type.value}: {hint[:300]}",
                                        item_type="decision",
                                        priority="medium",
                                    )
                                except Exception:
                                    pass

                                _forced_diag = self._same_error_streak_n >= 3
                                if _forced_diag:
                                    wd0 = self.working_directory or "."
                                    diag = gather_workspace_diagnostics_sync(wd0)
                                    sc = format_system_correction_block(
                                        workspace_display=wd0,
                                        is_windows=(os.name == "nt"),
                                    )
                                    _lfc = str(_args_dict.get("command", "") or "")
                                    inc0 = incremental_recovery_hint(
                                        tc.name,
                                        _args_dict,
                                        error_type,
                                        str(result),
                                        last_failed_command=_lfc,
                                    )
                                    self._post_mortem_failure_streaks.append(
                                        FailureStreakRecord(
                                            tool_name=tc.name,
                                            error_fingerprint=_fp,
                                            streak_length=self._same_error_streak_n,
                                        )
                                    )
                                    self._post_mortem_diagnostic_triggers.append(
                                        DiagnosticTriggerRecord(
                                            error_fingerprint=_fp,
                                            tool_name=tc.name,
                                            exit_code=exit_code,
                                            output_preview=str(result)[:400],
                                        )
                                    )
                                    self._pending_diagnostic_fp = _fp
                                    self.llm.add_user_message(
                                        f"{sc}\n\n### Live workspace snapshot (after 3 identical tool errors)\n"
                                        f"```\n{diag}\n```\n\n{inc0}"
                                    )
                                    self._append_session_event(
                                        AgentStreamEvent(
                                            kind=AgentEventKind.STATUS,
                                            role="system",
                                            text="Injected system correction + workspace snapshot (3× identical tool error).",
                                            legacy_type="forced_diagnostic",
                                            meta={
                                                "tool": tc.name,
                                                "error_fingerprint": _fp,
                                            },
                                        )
                                    )
                                    self._same_error_streak_n = 0
                                    self._same_error_streak_tool = None
                                    self._same_error_streak_fp = None

                                # Need to retry with hint
                                try:
                                    bad_args = (
                                        dict(tc.arguments)
                                        if isinstance(tc.arguments, dict)
                                        else {"raw": tc.arguments}
                                    )
                                except Exception:
                                    bad_args = {}
                                self._last_failed_tool = (
                                    tc.name,
                                    bad_args,
                                    str(result)[:1200],
                                )
                                retry_count += 1
                                hint = self._correction_engine.get_retry_hint(
                                    error_type, tc.name, tc.arguments, result
                                )
                                inc_line = incremental_recovery_hint(
                                    tc.name,
                                    _args_dict,
                                    error_type,
                                    str(result),
                                    last_failed_command=str(
                                        _args_dict.get("command", "") or ""
                                    ),
                                )

                                await self._trigger_callback(
                                    "on_message",
                                    f"🔄 **Self-Correcting**: Tool '{tc.name}' failed. Retrying... "
                                    f"(Attempt {retry_count}/{self.max_immediate_retries})\n*Hint:* {hint}",
                                    is_token=False,
                                )

                                # We feed the failed result and hint to LLM to get a corrected tool call
                                self.llm.add_tool_result(tc.id, tc.name, result)
                                tool_ids_answered.add(tc.id)
                                _batch_answered_tool_ids.add(tc.id)
                                if _forced_diag:
                                    self.llm.add_user_message(
                                        f"Your tool call failed (see System Correction + snapshot above).\n"
                                        f"Reminder: {hint}\n{inc_line}\n"
                                        "Provide a **different** corrected tool call — do not repeat the same command."
                                    )
                                else:
                                    self.llm.add_user_message(
                                        f"Your tool call failed: {hint}\n{inc_line}\n"
                                        "Please review the error and provide a corrected tool call."
                                    )

                                # Ask LLM again (observation / diagnostics stack — default Gemini Flash)
                                (
                                    _msgs,
                                    _eph,
                                ) = await self._prepare_messages_for_llm_turn()
                                retry_response = (
                                    await self._get_observation_llm().complete(
                                        tools=self._get_tool_schemas(),
                                        tool_choice="required",  # Force tool use
                                        stream=False,
                                        messages_for_api=_msgs,
                                        ephemeral_user_messages=_eph,
                                    )
                                )

                                if retry_response.tool_calls:
                                    # Update the current tool call with LLM's new attempt
                                    tc = retry_response.tool_calls[0]
                                    self.llm.add_assistant_message(
                                        content=retry_response.content,
                                        tool_calls=[tc],
                                    )
                                else:
                                    # LLM didn't provide a tool call, break retry loop
                                    break

                            if "on_tool_result" in self.callbacks:
                                await self._trigger_callback(
                                    "on_tool_result",
                                    tc.name,
                                    tc.arguments,
                                    final_result,
                                    duration_ms,
                                )

                            # Add final tool result once per tool_call_id (retry already appended failure for same id)
                            if tc.id not in tool_ids_answered:
                                self.llm.add_tool_result(tc.id, tc.name, final_result)
                                tool_ids_answered.add(tc.id)
                                _batch_answered_tool_ids.add(tc.id)

                            if tool_success and tc.name == "editor" and isinstance(tc.arguments, dict):
                                _act = tc.arguments.get("action", "")
                                if _act in ("write_file", "str_replace", "apply_patch"):
                                    _should_run_critic = True

                            if _single_tool_batch:
                                progress_guard = self._progress_guard_message(
                                    task,
                                    tc.name,
                                    tc.arguments,
                                    tool_success=tool_success,
                                )
                                if progress_guard:
                                    self.llm.add_user_message(progress_guard)
                                    self._append_session_event(
                                        AgentStreamEvent(
                                            kind=AgentEventKind.STATUS,
                                            role="system",
                                            text=progress_guard,
                                            legacy_type="progress_guard",
                                            meta={
                                                "task_id": task.task_id,
                                                "tool": tc.name,
                                            },
                                        )
                                    )

                            # Track in task state
                            if tc.name == "terminal":
                                task.commands_executed.append(
                                    tc.arguments.get("command", "")
                                )

                            clone_target = None
                            if tool_success:
                                clone_target = (
                                    self._clone_target_from_successful_command(
                                        task, tc.name, tc.arguments, final_result
                                    )
                                )
                            if clone_target:
                                status_args = {
                                    "command": "git status",
                                    "working_directory": clone_target,
                                    "plan_step": "STEP-4",
                                }
                                if "on_tool_start" in self.callbacks:
                                    await self._trigger_callback(
                                        "on_tool_start", "terminal", status_args
                                    )
                                import time as _time

                                _status_start = _time.time()
                                status_result = await self._execute_tool(
                                    "terminal", status_args
                                )
                                status_duration_ms = (
                                    _time.time() - _status_start
                                ) * 1000
                                if "on_tool_result" in self.callbacks:
                                    await self._trigger_callback(
                                        "on_tool_result",
                                        "terminal",
                                        status_args,
                                        status_result,
                                        status_duration_ms,
                                    )
                                task.commands_executed.append("git status")
                                if self._clone_status_verification_satisfied(
                                    task, "terminal", status_args, status_result
                                ):
                                    await self._update_phase(AgentPhase.COMPLETE)
                                    task.status = TaskStatus.COMPLETED
                                    task.completed_at = datetime.now(timezone.utc)
                                    self._log(
                                        "Task completed after automatic cloned repository git status."
                                    )
                                    break

                            if tool_success and self._terminal_task_complete_satisfied(
                                tc.name, final_result
                            ):
                                await self._update_phase(AgentPhase.COMPLETE)
                                task.status = TaskStatus.COMPLETED
                                task.completed_at = datetime.now(timezone.utc)
                                self._log(
                                    "Task completed after explicit terminal TASK COMPLETE signal."
                                )
                                break

                            if (
                                tool_success
                                and self._clone_status_verification_satisfied(
                                    task, tc.name, tc.arguments, final_result
                                )
                            ):
                                await self._update_phase(AgentPhase.COMPLETE)
                                task.status = TaskStatus.COMPLETED
                                task.completed_at = datetime.now(timezone.utc)
                                self._log(
                                    "Task completed after cloned repository git status passed."
                                )
                                break

                            if tool_success and self._no_edit_verification_satisfied(
                                task, tc.name, tc.arguments, final_result
                            ):
                                await self._update_phase(AgentPhase.COMPLETE)
                                task.status = TaskStatus.COMPLETED
                                task.completed_at = datetime.now(timezone.utc)
                                self._log(
                                    "Task completed after no-edit verification passed."
                                )
                                break

                            if tool_success and self._simple_file_task_satisfied(
                                task, tc.name, tc.arguments, final_result
                            ):
                                await self._update_phase(AgentPhase.COMPLETE)
                                task.status = TaskStatus.COMPLETED
                                task.completed_at = datetime.now(timezone.utc)
                                self._log(
                                    "Task completed after targeted file verification."
                                )
                                break

                            # Handle Escalation to Planner
                            if not tool_success:
                                self._consecutive_failures += 1
                                if self._correction_engine.should_replan(
                                    self._consecutive_failures
                                ):
                                    await self._trigger_callback(
                                        "on_message",
                                        "🔁 **Replanning needed**: Consecutive failures exceeded limit.",
                                        is_token=False,
                                    )
                                    self._consecutive_failures = 0

                                    if await self.auto_replan_from_repeated_failure(
                                        final_result
                                    ):
                                        # Force breakdown of iteration loop to let new plan take over
                                        task.error_count += 1
                                        # Replan must not leave orphan tool_calls on the last assistant message
                                        for _ptc in response.tool_calls:
                                            if _ptc.id not in _batch_answered_tool_ids:
                                                self.llm.add_tool_result(
                                                    _ptc.id,
                                                    _ptc.name,
                                                    "Error: replan aborted before this tool_call_id received a result.",
                                                )
                                                _batch_answered_tool_ids.add(_ptc.id)
                                        self._synthesize_tool_replies_for_open_assistant_tail()
                                        break

                        # --- Multi-Agent Critic System Injection ---
                        if _should_run_critic and task.status != TaskStatus.COMPLETED:
                            self._log("Triggering ReviewerAgent for Critic check...")
                            try:
                                review_feedback = await self.review_changes(task_description=task.goal.description)
                                if not review_feedback.approved or review_feedback.has_blocking_issues:
                                    # State Rollback
                                    if git_mgr:
                                        self._log(f"Critic rejected changes. Rolling back to {_turn_checkpoint_id}...")
                                        await git_mgr.rollback_to_checkpoint(_turn_checkpoint_id, hard=True)
                                        if "on_message" in self.callbacks:
                                            await self._trigger_callback(
                                                "on_message",
                                                "⏪ **State Rollback Triggered**: Critic rejected changes. Reverted to previous clean state.",
                                                is_token=False,
                                            )
                                    
                                    critic_msg = (
                                        "Critic / Reviewer Feedback on your recent edit:\n\n"
                                        + review_feedback.format_report()
                                        + "\n\n⚠️ **STATE ROLLBACK ACTIVATED**: Your recent code changes have been automatically reverted because they contained HIGH/CRITICAL issues. "
                                        "The file is now back to its previous pristine state. Please rewrite the code from scratch incorporating the feedback above."
                                    )
                                    self._log(f"Reviewer produced feedback (Approved: {review_feedback.approved})")
                                    self.llm.add_user_message(critic_msg)
                                    self._append_session_event(
                                        AgentStreamEvent(
                                            kind=AgentEventKind.STATUS,
                                            role="system",
                                            text="Critic feedback appended and rollback executed.",
                                            legacy_type="critic_feedback",
                                            meta={"task_id": task.task_id},
                                        )
                                    )
                                    if "on_message" in self.callbacks:
                                        await self._trigger_callback(
                                            "on_message",
                                            "🔍 **Critic Review Failed**: Issues found with recent changes. See details in context.",
                                            is_token=False,
                                        )
                                else:
                                    self._log("Reviewer approved changes.")
                            except Exception as e:
                                self._log(f"Critic Review failed: {e}")
                        # ---------------------------------------------

                        if task.status == TaskStatus.COMPLETED:
                            break
                    else:
                        # No tool calls - check if task is complete
                        if response.content:
                            self._log(f"Assistant: {response.content[:200]}...")
                            self.llm.add_assistant_message(content=response.content)

                            # Detect numbered plan in LLM response (e.g. "1. Step one\n2. Step two")
                            content = response.content or ""
                            plan_lines = re.findall(
                                r"^\s*(\d+)\.\s+(.+)", content, re.MULTILINE
                            )
                            if len(plan_lines) >= 2 and not getattr(
                                self, "_plan_sent", False
                            ):
                                steps = [text.strip() for _, text in plan_lines]
                                self._plan_sent = True
                                self._plan_steps = steps
                                self._current_step_idx = 0
                                await self._trigger_callback("on_plan_created", steps)
                                await self._update_phase(AgentPhase.PLAN)

                                # Start first step
                                if steps:
                                    await self._trigger_callback(
                                        "on_step_started", 0, steps[0]
                                    )
                                self._persist_worklog(task)

                            # Only allow TASK COMPLETE if at least one tool was used this run
                            _tools_used = getattr(self, "_tools_used_count", 0)
                            content_lower = response.content.lower()
                            is_completion = any(
                                phrase in content_lower
                                for phrase in [
                                    "task complete",
                                    "task is complete",
                                    "successfully completed",
                                    "finished the task",
                                    "completed the task",
                                    "all done",
                                ]
                            )

                            if is_completion and _tools_used > 0:
                                await self._update_phase(AgentPhase.COMPLETE)
                                task.status = TaskStatus.COMPLETED
                                task.completed_at = datetime.now(timezone.utc)
                                self._log("Task completed!")
                                break
                            is_verified_completion = (
                                _tools_used > 0
                                and "complete" in content_lower
                                and any(
                                    phrase in content_lower
                                    for phrase in (
                                        "verified",
                                        "confirmed",
                                        "contains exactly",
                                        "as required",
                                        "correct content",
                                    )
                                )
                            )
                            if is_verified_completion:
                                await self._update_phase(AgentPhase.COMPLETE)
                                task.status = TaskStatus.COMPLETED
                                task.completed_at = datetime.now(timezone.utc)
                                self._log("Task completed after verified summary.")
                                break
                            elif is_completion and _tools_used == 0:
                                # Agent said complete without using any tools — force it to act
                                self._log(
                                    "Completion signal with no tool use — forcing tool call."
                                )
                                self.llm.add_user_message(
                                    "You have not used any tools yet. You MUST use the available tools (terminal, editor, etc.) "
                                    "to actually perform the task. Do NOT just describe what you would do — call a tool now."
                                )
                            elif response.finish_reason == "stop":
                                # Nudge with forced tool use on the next iteration
                                self._no_tool_streak = (
                                    getattr(self, "_no_tool_streak", 0) + 1
                                )
                                force_after = int(
                                    os.environ.get(
                                        "PLODDER_FORCE_TOOL_AFTER_TEXT_TURNS", "2"
                                    )
                                )
                                if self._no_tool_streak >= force_after:
                                    # Avoid spending a second LLM call immediately after the first text-only turn.
                                    self._log(
                                        "Multiple text-only turns — injecting forced tool call."
                                    )
                                    (
                                        _msgs,
                                        _eph,
                                    ) = await self._prepare_messages_for_llm_turn()
                                    forced_response = (
                                        await self._get_observation_llm().complete(
                                            tools=self._get_tool_schemas(),
                                            tool_choice="required",
                                            stream=False,
                                            messages_for_api=_msgs,
                                            ephemeral_user_messages=_eph,
                                        )
                                    )
                                    self._no_tool_streak = 0
                                    if forced_response.tool_calls:
                                        # Re-inject as if it was a normal tool call turn
                                        self.llm.add_assistant_message(
                                            content=forced_response.content,
                                            tool_calls=forced_response.tool_calls,
                                        )
                                        for tc in forced_response.tool_calls:
                                            try:
                                                result = await self._execute_tool(
                                                    tc.name,
                                                    tc.arguments,
                                                    thought=forced_response.content,
                                                )
                                            except Exception as _fe:
                                                result = f"Error: forced tool execution failed: {_fe}"
                                            self._tools_used_count = (
                                                getattr(self, "_tools_used_count", 0)
                                                + 1
                                            )
                                            self.llm.add_tool_result(
                                                tc.id, tc.name, result
                                            )
                                else:
                                    self.llm.add_user_message(
                                        "Continue with the next necessary tool call. If the task is already verified, "
                                        "reply with TASK COMPLETE and do not verify the same fact again."
                                    )

                except Exception as e:
                    error_msg = f"Error in iteration: {str(e)}"
                    self._log(error_msg)
                    # Repair OpenAI message ordering if we stopped mid tool_calls (e.g. LLM error mid-retry)
                    try:
                        self._synthesize_tool_replies_for_open_assistant_tail()
                    except Exception:
                        pass

                    # Report error to UI if callback exists
                    await self._trigger_callback(
                        "on_message", f"⚠️ **Error**: {str(e)}", is_token=False
                    )

                    if self._is_hard_rate_limit_error(str(e)):
                        await self._update_phase(AgentPhase.BLOCKED)
                        task.status = TaskStatus.FAILED
                        task.last_error = (
                            "Provider quota exhausted (hard rate limit). "
                            "Execution paused to avoid retry spam. "
                            "Switch model/provider or wait for quota reset before retrying."
                        )
                        break

                    task.last_error = str(e)
                    task.error_count += 1

                    if task.error_count >= 3:
                        await self._update_phase(AgentPhase.BLOCKED)
                        task.status = TaskStatus.FAILED
                        break

            # Max iterations reached
            if iteration >= self.max_iterations and task.status != TaskStatus.COMPLETED:
                self._log("Max iterations reached")
                task.status = TaskStatus.FAILED
                task.last_error = (
                    f"Max iterations ({self.max_iterations}) reached. "
                    "Increase max iterations in New Session, split the task into smaller steps, "
                    "or ensure API keys and workspace paths are correct."
                )

            # Update token usage (primary + optional observation/diagnostics client)
            usage = self.llm.get_usage_stats()
            task.total_tokens_used = usage["total_tokens"]
            if (
                self._observation_llm is not None
                and self._observation_llm is not self.llm
            ):
                ou = self._observation_llm.get_usage_stats()
                task.total_tokens_used += ou["total_tokens"]

            # Save final artifacts
            if self._artifact_logger:
                self._artifact_logger.update_tokens(task.total_tokens_used)

                # Get git diff if available
                git_mgr = self._get_git_manager()
                if git_mgr:
                    diff_result = await git_mgr.get_diff()
                    if diff_result and not diff_result.is_empty:
                        self._artifact_logger.set_diff(diff_result.diff_text)

                # Get final summary from last assistant message
                summary = ""
                for msg in reversed(self.llm.conversation):
                    if msg.role == "assistant" and msg.content:
                        summary = msg.content
                        break

                # Complete the artifact logger
                status = (
                    "completed" if task.status == TaskStatus.COMPLETED else "failed"
                )
                if self.state.phase == AgentPhase.BLOCKED:
                    status = "blocked"
                self._artifact_logger.complete(status=status, summary=summary)
                try:
                    _pm = PostMortemPayload(
                        goal_or_task_id=task.task_id,
                        diagnostic_triggers=list(self._post_mortem_diagnostic_triggers),
                        recovery_paths=list(self._post_mortem_recovery_paths),
                        failure_streaks=list(self._post_mortem_failure_streaks),
                    )
                    Path(self._artifact_logger.get_run_dir()).joinpath(
                        "post_mortem.md"
                    ).write_text(
                        format_post_mortem(_pm),
                        encoding="utf-8",
                    )
                except Exception as _pm_err:
                    self._log(f"Warning: could not write post_mortem.md: {_pm_err}")

                self._log(f"Artifacts saved to: {self._artifact_logger.get_run_dir()}")

            # Save task summary to conversation memory (Phase 18)
            if self._use_conversation_memory:
                try:
                    self.save_task_summary(
                        task=task,
                        summary=summary,
                        lessons_learned=list(self._session_self_correct_lessons),
                    )
                    self._log("Task summary saved to conversation memory")
                except Exception as e:
                    self._log(f"Warning: Failed to save task summary to memory: {e}")

            # ── Checkpoint → Verify → Rollback-on-failure ──────────────────────────
            # Skip when the agent made no repo edits and ran no shell commands (e.g. browse-only tasks),
            # so pytest/npm verification does not fail unrelated workspaces.
            _skip_post_verify = not task.files_changed and not task.commands_executed
            if task.status == TaskStatus.COMPLETED and self.working_directory:
                git_mgr = self._get_git_manager()
                if git_mgr and _skip_post_verify:
                    self._log(
                        "Skipping post-task verification (no file edits or terminal commands)"
                    )
                    await self._trigger_callback(
                        "on_message",
                        "ℹ️ **Git**: Skipping automated verification — no workspace file edits or shell commands this run.",
                        is_token=False,
                    )
                elif git_mgr:
                    checkpoint_id = f"post-task-{task.task_id}"
                    try:
                        # Create a checkpoint with the current changes
                        await git_mgr.create_checkpoint(
                            checkpoint_id, f"agent: {task.goal.description[:60]}"
                        )
                        self._log(f"Checkpoint created: {checkpoint_id}")
                        await self._trigger_callback(
                            "on_message",
                            "📌 **Git**: Checkpoint created — running verification…",
                            is_token=False,
                        )

                        # Run verification (tests, lints)
                        verification = await self.run_verification(task_id=task.task_id)

                        if (
                            verification
                            and not verification.passed
                            and verification.blocking_failures
                        ):
                            # Tests failed — rollback to checkpoint
                            await self._trigger_callback(
                                "on_message",
                                f"⚠️ **Verification failed** ({len(verification.blocking_failures)} issue(s)). "
                                "Rolling back to last checkpoint…",
                                is_token=False,
                            )
                            self._log(
                                f"Verification failed. Rolling back to {checkpoint_id}"
                            )
                            await git_mgr.rollback_to_checkpoint(
                                checkpoint_id, hard=True
                            )
                            task.status = TaskStatus.FAILED
                            task.last_error = f"Verification failed: {', '.join(verification.blocking_failures)}"
                            await self._trigger_callback(
                                "on_message",
                                "🔄 **Rolled back** to pre-task state. Please review errors and retry.",
                                is_token=False,
                            )
                        elif verification and verification.passed:
                            self._log("Verification passed — keeping changes")
                            await self._trigger_callback(
                                "on_message",
                                f"✅ **Verification passed** ({verification.checks_passed}/{verification.total_checks} checks)",
                                is_token=False,
                            )
                    except Exception as e:
                        self._log(f"Checkpoint/verify step skipped: {e}")

            # Auto git commit after task completion (only if still COMPLETED after verification)
            if (
                self.auto_git_commit
                and task.status == TaskStatus.COMPLETED
                and self.working_directory
            ):
                await self._auto_commit_changes(task)

            # Optional teacher review + JSONL training log (OpenAI-first; see TEACHER_* env vars)
            summary_for_teacher = ""
            for _msg in reversed(self.llm.conversation):
                if _msg.role == "assistant" and _msg.content:
                    summary_for_teacher = _msg.content
                    break
            try:
                await maybe_log_teacher_review(
                    task=task,
                    agent_model=self.llm.config.model,
                    conversation=list(self.llm.conversation),
                    summary=summary_for_teacher,
                    verbose=self.verbose,
                )
            except Exception as _e:
                self._log(f"Teacher review skipped: {_e}")

            return task
        finally:
            if _bb_for_run:
                await self._async_teardown_backbone()

    async def _auto_commit_changes(self, task: TaskState) -> None:
        """Automatically commit and optionally push changes after task completion."""
        from pathlib import Path

        cwd = self.working_directory
        git_dir = Path(cwd) / ".git"
        try:
            # Init repo if not a git repo yet
            if not git_dir.exists():
                import asyncio

                proc = await asyncio.create_subprocess_shell(
                    "git init && git add -A",
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            else:
                proc = await asyncio.create_subprocess_shell(
                    "git add -A",
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

            # Build commit message from task description (use -F file so quotes/newlines cannot break git)
            import tempfile

            short_desc = task.goal.description[:72] if task.goal.description else "task"
            commit_msg = f"feat: {short_desc}"
            msg_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    suffix=".gitmsg",
                    delete=False,
                    newline="\n",
                ) as tf:
                    tf.write(commit_msg)
                    msg_path = tf.name
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "commit",
                    "-F",
                    msg_path,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
            finally:
                if msg_path:
                    try:
                        os.unlink(msg_path)
                    except OSError:
                        pass
            if proc.returncode == 0:
                self._log(f"Auto-committed changes: {commit_msg}")
                await self._trigger_callback(
                    "on_message",
                    f"✅ **Git**: Committed — `{commit_msg}`",
                    is_token=False,
                )
            else:
                out = (stdout or b"").decode() + (stderr or b"").decode()
                if "nothing to commit" in out:
                    self._log("Auto-commit: nothing to commit")
                else:
                    self._log(f"Auto-commit warning: {out[:200]}")

            # Push if enabled (skip pushing default branch unless opted in — PR-first GitHub flow)
            if self.git_push:
                allow_default_push = os.environ.get(
                    "PLODDER_GIT_PUSH_DEFAULT_BRANCH", ""
                ).strip().lower() in ("1", "true", "yes", "on")
                cur = await _git_current_branch(cwd)
                origin_def = await _git_origin_default_branch(cwd)
                if not allow_default_push and cur and origin_def and cur == origin_def:
                    hint = (
                        "Auto-push skipped: you are on the remote default branch. "
                        "Use the `github` tool (create_branch → commit → create_pr), "
                        "or set env `PLODDER_GIT_PUSH_DEFAULT_BRANCH=1` to allow push here."
                    )
                    self._log(hint)
                    await self._trigger_callback(
                        "on_message",
                        f"⚠️ **Git**: {hint}",
                        is_token=False,
                    )
                else:
                    proc = await asyncio.create_subprocess_shell(
                        "git push",
                        cwd=cwd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await proc.communicate()
                    if proc.returncode == 0:
                        self._log("Auto-pushed to remote")
                        await self._trigger_callback(
                            "on_message", "✅ **Git**: Pushed to remote", is_token=False
                        )
                    else:
                        err = (stderr or b"").decode()
                        self._log(f"Auto-push failed: {err[:200]}")
        except Exception as e:
            self._log(f"Auto-commit failed: {e}")

    async def _async_setup_backbone(self) -> None:
        """Create EventStream + runtime + AgentController and start the consumer (idempotent)."""
        if self._backbone_controller is not None:
            return
        from ..backbone import AgentStateStore, EventStream
        from ..backbone.controller import AgentController, AgentHostRuntime
        from ..backbone.local_runtime import LocalRuntime

        store = AgentStateStore()
        stream = EventStream(store=store)
        use_host = os.environ.get(
            "PLODDER_BACKBONE_HOST_RUNTIME", "1"
        ).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        runtime: Any = (
            AgentHostRuntime(self) if use_host else LocalRuntime(self.registry)
        )
        controller = AgentController(stream, store, runtime)
        self._backbone_stream = stream
        self._backbone_runtime = runtime
        self._backbone_controller = controller
        await controller.start()

    async def _async_teardown_backbone(self) -> None:
        """Stop backbone consumer and clear references (safe if already torn down)."""
        self._use_backbone_dispatch = False
        ctrl = self._backbone_controller
        if ctrl is None:
            self._backbone_stream = None
            self._backbone_runtime = None
            return
        try:
            await ctrl.stop()
        finally:
            self._backbone_controller = None
            self._backbone_stream = None
            self._backbone_runtime = None

    async def run_simple(self, task_description: str) -> str:
        """
        Simplified interface to run a task from a description.

        Args:
            task_description: Natural language description of the task

        Returns:
            Summary of what was done
        """
        await self._async_setup_backbone()
        self._use_backbone_dispatch = True
        try:
            task = TaskState(
                task_id=str(uuid.uuid4()),
                goal=TaskGoal(
                    description=task_description,
                    acceptance_criteria=[],
                ),
            )

            result = await self.run(task)

            # Get the last assistant message as summary
            summary = f"Task {'completed' if result.status == TaskStatus.COMPLETED else 'failed'}"
            for msg in reversed(self.llm.conversation):
                if msg.role == "assistant" and msg.content:
                    summary = msg.content
                    break

            if result.status == TaskStatus.COMPLETED:
                await self._trigger_callback("on_task_complete", summary)
            elif result.status == TaskStatus.FAILED:
                await self._trigger_callback("on_task_failed", result.last_error)

            return summary
        finally:
            await self._async_teardown_backbone()


async def create_agent(
    model: str = "gpt-4o",
    api_key: str | None = None,
    working_directory: str | None = None,
    verbose: bool = True,
    callbacks: dict[str, Any] | None = None,
) -> Agent:
    """Create an agent with default configuration."""
    import os

    llm = create_llm_client(
        model=model,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )

    return Agent(
        llm_client=llm,
        working_directory=working_directory,
        verbose=verbose,
        callbacks=callbacks,
    )
