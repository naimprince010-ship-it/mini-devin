"""
Unified Session Driver — multi-turn tool use (filesystem + sandbox) for Plodder.

The model emits strict JSON with ``tool_calls``; the driver executes them against a
``SessionWorkspace`` and optional ``ExecutionSandbox``, then feeds results back until
``status: done`` or max rounds.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from plodder.core.universal_prompt_engine import PolyglotSystemPrompt, UniversalPromptEngine
from plodder.sandbox.execution_sandbox import SandboxResult
from plodder.sandbox.toolchain_detect import pick_default_entry
from plodder.agent.orchestrator import PlodderWorklog, SessionRecoveryTracker, export_worklog_json
from plodder.memory.learned_patterns import load_learned_patterns_for_prompt
from plodder.memory.session_memory import EpisodeMemory
from plodder.memory.workspace_code_index import WorkspaceCodeIndex
from plodder.orchestration.reasoning_loop import (
    REASONING_LOOP_SEED_SUFFIX,
    apply_sliding_window_to_messages,
    build_agent_thought_text,
    extract_shell_inner,
    goal_suggests_frontend_stack,
    monologue_validation_error,
    parse_driver_turn,
    terminal_failure_followup_hints,
    tool_observation_truncation_limit,
    visual_review_done_gate,
)
from plodder.sandbox.stateful_shell_tracker import StatefulShellTracker
from plodder.sandbox.stream_truncate import truncate_stream
from plodder.workspace.atomic_editor import atomic_edit
from plodder.workspace.session_workspace import SessionWorkspace

# API / HTTP session id for ``live_preview`` when ``UnifiedSessionDriver`` is constructed without ``session_id``.
_plodder_api_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "plodder_api_session_id", default=None
)


@contextlib.contextmanager
def plodder_session_id_binding(session_id: str):
    """
    Bind the current asyncio task to an API ``session_id`` so ``UnifiedSessionDriver(..., session_id=None)``
    still resolves ``live_preview`` to the correct ``/api/sessions/{id}/live-preview/`` registration.

    Used by :class:`mini_devin.sessions.manager.SessionManager` during ``run_task`` (avoids mutating
    ``os.environ`` when multiple sessions may run concurrently).
    """
    token = _plodder_api_session_id.set(session_id)
    try:
        yield
    finally:
        _plodder_api_session_id.reset(token)


ChatMessage = dict[str, Any]
LLMFn = Callable[[list[ChatMessage]], Awaitable[str]]


def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(n, hi))


class SupportsSessionSandbox(Protocol):
    def run_detected(
        self,
        files: dict[str, str],
        *,
        entry: str,
        language: str | None = None,
        language_key: str | None = None,
        timeout_sec: int | None = None,
        network: bool = False,
    ) -> SandboxResult: ...

    def run_shell_in_workspace(
        self,
        files: dict[str, str],
        argv: list[str],
        *,
        language_hint: str | None = None,
        timeout_sec: int | None = None,
        network: bool = False,
    ) -> SandboxResult: ...


@dataclass
class UnifiedSessionResult:
    success: bool
    goal: str
    rationale: str
    rounds: int
    workspace_root: str
    transcript: list[dict[str, Any]] = field(default_factory=list)
    plan_markdown: str = ""
    #: OpenHands-style markdown: failure streaks, diagnostic fingerprints, recovery paths
    post_mortem_markdown: str = ""
    #: Serialized :class:`PlodderWorklog` (action–observation stream); also written to ``worklog.json``
    worklog: dict[str, Any] = field(default_factory=dict)


_BASE_ALLOWED_TOOLS = frozenset(
    {
        "fs_list",
        "fs_read",
        "fs_write",
        "fs_delete",
        "sandbox_run",
        "sandbox_shell",
        "github",
        "gitleaks",
        "search_codebase",
        "atomic_edit",
        "lsp_check",
        "playwright_observe",
        "browser_click",
        "browser_type",
        "browser_scroll",
        "browser_close",
        "live_preview",
    }
)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 20] + "\n…(truncated)…\n"


def _shell_scaffold_operation_cancelled(stdout: str, stderr: str, command: str) -> bool:
    """
    npm/create-vite often exits 0 while printing ``Operation cancelled`` when stdin is not a TTY
    and the target directory already exists (default non-interactive overwrite = no).
    Treat that as failure so follow-up hints and ``ok: false`` propagate.
    """
    blob = ((stdout or "") + "\n" + (stderr or "")).lower()
    if "operation cancelled" not in blob:
        return False
    cmd = (command or "").lower()
    if any(p in cmd for p in ("create-vite", "create vite", "npm create vite", "@vite/create-vite")):
        return True
    return "npx" in cmd and "vite" in cmd


class UnifiedSessionDriver:
    """
    Multi-file autonomous loop: list/read/write/delete + sandbox run/shell.

    Lazy-imports ``parse_pseudo_plan_json`` inside ``run`` to avoid import cycles with ``self_heal``.
    """

    def __init__(
        self,
        *,
        llm: LLMFn,
        workspace: SessionWorkspace,
        sandbox: SupportsSessionSandbox | None,
        engine: UniversalPromptEngine | None = None,
        max_rounds: int = 24,
        max_tool_calls_per_turn: int = 8,
        sandbox_timeout_sec: int | None = 120,
        inject_logic_plan: bool = True,
        enable_episode_memory: bool = True,
        enable_workspace_code_index: bool = True,
        enable_stateful_shell: bool = True,
        enforce_think_before_act: bool = True,
        inject_long_context_anchor: bool = True,
        session_id: str | None = None,
    ) -> None:
        self._llm = llm
        self._ws = workspace
        self._sandbox = sandbox
        if session_id is not None:
            raw_sid = session_id
        else:
            ctx = _plodder_api_session_id.get()
            raw_sid = (ctx or "").strip() or None
            if not raw_sid:
                raw_sid = os.environ.get("PLODDER_SESSION_ID")
        self._session_id = (str(raw_sid).strip() if raw_sid else "") or None
        self._engine = engine or UniversalPromptEngine(PolyglotSystemPrompt())
        self._max_rounds = max_rounds
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._sandbox_timeout_sec = sandbox_timeout_sec
        self._inject_logic_plan = inject_logic_plan
        self._enable_episode_memory = enable_episode_memory
        self._enable_workspace_code_index = enable_workspace_code_index
        self._enable_stateful_shell = enable_stateful_shell
        self._enforce_think_before_act = enforce_think_before_act
        self._inject_long_context_anchor = inject_long_context_anchor
        self._runtime_code_index: WorkspaceCodeIndex | None = None
        self._runtime_episode_memory: EpisodeMemory | None = None
        self._runtime_shell_tracker: StatefulShellTracker | None = None
        self._browser_lock = threading.Lock()
        self._browser_pw: Any = None  # PlodderPlaywrightSession | None (lazy import)

    def allowed_tools(self) -> frozenset[str]:
        """Tool names this driver accepts (subclasses may extend)."""
        return _BASE_ALLOWED_TOOLS

    def _build_seed_user(self, goal: str, plan_md: str) -> str:
        """First user turn after system: workspace context + goal + plan + instructions."""
        seed_user = (
            f"## Workspace root\n`{self._ws.root}`\n\n"
            f"## Goal\n{goal.strip()}\n\n"
        )
        if plan_md:
            seed_user += f"## Pseudo-logic plan (follow; do not contradict)\n{plan_md}\n\n"
        seed_user += (
            "Implement using tools. Prefer **atomic_edit** for surgical file edits. "
            "When ready to verify, use `sandbox_run` with the correct `entry` path; use **lsp_check** "
            "and the browser tools (**playwright_observe**, then **browser_click** / **browser_type** "
            "using `element_id` from `interactive_elements`) before claiming UI/code success."
        )
        seed_user += REASONING_LOOP_SEED_SUFFIX
        if goal_suggests_frontend_stack(goal):
            seed_user += (
                "\n\n## Headless ``sandbox_shell`` (no TTY)\n"
                "npm / **create-vite** cannot answer interactive prompts. If the target folder may already exist, "
                "either **`rm -rf <dir>`** then scaffold, or use **`npm create vite@latest <dir> -- --template … "
                "--overwrite`** (and **`--no-interactive`** when the CLI supports it). "
                "If you see **Operation cancelled**, do **not** repeat the same bare `npx create-vite …` without "
                "``--overwrite`` or a clean path.\n"
            )
            snap = self._host_environment_snapshot_block()
            if snap.strip():
                seed_user += (
                    "\n\n## Host workspace snapshot (environmental grounding)\n"
                    "Plodder captured `pwd`-style layout at session start — **reconcile** with "
                    "`sandbox_shell` + `fs_list` before assuming paths exist.\n```text\n"
                    + _truncate(snap, 9000)
                    + "\n```\n"
                )
            seed_user += (
                "\n\n## Live Preview (optional; same API as the mini-devin agent)\n"
                "When a **dev server is listening on this machine’s** ``127.0.0.1`` (typical with **host "
                "process** ``sandbox_shell``, e.g. Railway without Docker), call **`live_preview`** with "
                "``action: probe`` (optional ``ports`` list), then **`live_preview`** with ``action: set_active_port`` "
                "and ``port`` set to a **listening** port from the probe so the **Browser** tab can load the "
                "proxied iframe. Use **``playwright_observe``** with the same host URL (e.g. ``http://127.0.0.1:5173``). "
                "**Ephemeral Docker** ``sandbox_shell`` runs cannot publish ports to the host — probe will usually "
                "return empty; then use a tunnel URL in ``playwright_observe`` or switch to host sandbox.\n"
            )
        return seed_user

    def _host_environment_snapshot_block(self) -> str:
        try:
            from mini_devin.reliability.self_correction import gather_workspace_diagnostics_sync

            return gather_workspace_diagnostics_sync(str(self._ws.root), max_chars=8000)
        except Exception:
            return ""

    def _long_context_anchor_block(self, worklog: PlodderWorklog, round_idx: int) -> str:
        """
        Re-inject PLAN.md and worklog tail so long-horizon tasks (Gemini-scale context) stay grounded.
        Full PLAN.md every 5 rounds; worklog tail every round.
        """
        chunks: list[str] = []
        plan_path = Path(self._ws.root) / "PLAN.md"
        if round_idx == 0 or round_idx % 5 == 0:
            if plan_path.is_file():
                try:
                    pt = plan_path.read_text(encoding="utf-8", errors="replace")
                    if len(pt) > 40_000:
                        pt = pt[:40_000] + "\n\n...(PLAN.md truncated)\n"
                    chunks.append("## PLAN.md (workspace root)\n```markdown\n" + pt + "\n```")
                except OSError:
                    pass
        summary = (worklog.summary_of_progress or "").strip()
        if summary:
            tail = summary[-8000:] if len(summary) > 8000 else summary
            chunks.append("## Worklog summary_of_progress (tail)\n```text\n" + tail + "\n```")
        tail_events = worklog.events[-10:]
        if tail_events:
            chunks.append(
                "## Recent worklog events (last 10)\n```json\n"
                + json.dumps(tail_events, indent=2, default=str, ensure_ascii=False)[:8000]
                + "\n```"
            )
        return "\n\n".join(chunks) if chunks else ""

    async def _browser_session_shutdown_async(self) -> None:
        def _close() -> None:
            with self._browser_lock:
                pw = self._browser_pw
                if pw is not None:
                    try:
                        pw.close()
                    except Exception:
                        pass
                    self._browser_pw = None

        await asyncio.to_thread(_close)

    async def _browser_exec(self, fn: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
        def wrapper() -> dict[str, Any]:
            with self._browser_lock:
                from plodder.tools.browser_grounding import PlodderPlaywrightSession

                if self._browser_pw is None:
                    self._browser_pw = PlodderPlaywrightSession(headless=True)
                return fn(self._browser_pw)

        return await asyncio.to_thread(wrapper)

    @staticmethod
    def _observe_bundle_kwargs(args: dict[str, Any]) -> dict[str, Any]:
        wait_ms = _clamp_int(args.get("wait_ms"), 200, 8000, 900)
        capture_console = bool(args.get("capture_console", True))
        include_accessibility = bool(args.get("include_accessibility", True))
        max_elements = _clamp_int(args.get("max_interactive_elements"), 10, 200, 120)
        full_page = bool(args.get("full_page_screenshot", False))
        vw_arg = args.get("viewport_width")
        vh_arg = args.get("viewport_height")
        try:
            viewport_width = int(vw_arg) if vw_arg is not None and str(vw_arg).strip() != "" else None
        except (TypeError, ValueError):
            viewport_width = None
        try:
            viewport_height = int(vh_arg) if vh_arg is not None and str(vh_arg).strip() != "" else None
        except (TypeError, ValueError):
            viewport_height = None
        return {
            "wait_ms": wait_ms,
            "capture_console": capture_console,
            "include_accessibility": include_accessibility,
            "max_elements": max_elements,
            "full_page": full_page,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
        }

    @staticmethod
    def _nav_url_from_args(args: dict[str, Any]) -> str | None:
        raw = args.get("url")
        if raw is None:
            return None
        s = str(raw).strip()
        return s if s else None

    async def _complete(self, messages: list[ChatMessage]) -> str:
        return await self._llm(messages)

    def _system_content(self) -> str:
        parts = [
            self._engine.persona.base_instruction(),
            self._engine.session_unified_driver_contract(),
            self._engine.session_unified_tools_catalog(),
        ]
        base = "\n\n".join(parts)
        try:
            learned = load_learned_patterns_for_prompt(self._ws.root)
        except Exception:
            learned = ""
        if learned:
            base = base + "\n\n" + learned
        return base

    async def run(self, goal: str) -> UnifiedSessionResult:
        from plodder.orchestration.self_heal import parse_pseudo_plan_json

        transcript: list[dict[str, Any]] = []
        plan_md = ""
        recovery_tracker = SessionRecoveryTracker()
        worklog = PlodderWorklog(goal=goal, workspace_root=str(self._ws.root))
        prior_round_diagnostic = False

        self._runtime_code_index = None
        self._runtime_episode_memory = None
        if self._enable_episode_memory:
            self._runtime_episode_memory = EpisodeMemory(self._ws.root)
            self._runtime_episode_memory.append(
                "meta",
                {"event": "session_start", "goal": goal[:4000]},
                round_idx=None,
            )
        if self._enable_workspace_code_index:
            idx = WorkspaceCodeIndex(self._ws.root)
            try:
                stats = await asyncio.to_thread(idx.index_workspace)
                transcript.append({"phase": "code_index", "stats": stats})
                self._runtime_code_index = idx
            except Exception as e:
                transcript.append({"phase": "code_index", "error": str(e)})

        self._runtime_shell_tracker = (
            StatefulShellTracker(self._ws.root) if self._enable_stateful_shell else None
        )

        if self._inject_logic_plan:
            plan_messages: list[ChatMessage] = [
                *self._engine.system_messages(),
                {"role": "user", "content": self._engine.planner_json_user_prompt(goal)},
            ]
            plan_raw = await self._complete(plan_messages)
            plan = parse_pseudo_plan_json(plan_raw, fallback_goal=goal)
            plan_md = plan.to_markdown_brief()
            transcript.append({"phase": "plan", "raw": plan_raw[:12000]})

        seed_user = self._build_seed_user(goal, plan_md)

        messages: list[ChatMessage] = [
            {"role": "system", "content": self._system_content()},
            {"role": "user", "content": seed_user},
        ]

        final_rationale = ""
        success = False
        rounds_used = 0
        terminated_with_done = False
        sliding_running_summary = ""

        async def _maybe_sliding_compact() -> None:
            nonlocal sliding_running_summary
            try:
                sliding_running_summary = await apply_sliding_window_to_messages(
                    messages,
                    self._complete,
                    sliding_running_summary,
                )
            except Exception:
                pass

        try:
            for round_idx in range(self._max_rounds):
                if round_idx > 0 and self._runtime_episode_memory:
                    cont = self._runtime_episode_memory.get_condensed_context()
                    if cont.strip():
                        messages.append(
                            {
                                "role": "user",
                                "content": "## Episode continuity (from session_memory.jsonl)\n" + cont,
                            }
                        )
                if self._inject_long_context_anchor:
                    anchor = self._long_context_anchor_block(worklog, round_idx)
                    if anchor.strip():
                        messages.append(
                            {
                                "role": "user",
                                "content": "## Long-horizon continuity (PLAN.md + worklog)\n" + anchor,
                            }
                        )
                raw = await self._complete(messages)
                tr_assistant: dict[str, Any] = {"round": round_idx, "assistant_raw": raw[:24000]}
                try:
                    turn = parse_driver_turn(raw)
                    for k in (
                        "observe",
                        "think",
                        "act_summary",
                        "sub_goal",
                        "risk_assessment",
                        "expected_outcome",
                    ):
                        if k in turn:
                            tr_assistant[k] = turn[k]
                except (json.JSONDecodeError, ValueError) as e:
                    transcript.append(tr_assistant)
                    messages.append({"role": "assistant", "content": raw[:12000]})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "## Parse error\n"
                                f"Your last message was not valid JSON: {e}\n"
                                "Reply again with **only** one JSON object per the contract."
                            ),
                        }
                    )
                    rounds_used += 1
                    await _maybe_sliding_compact()
                    continue

                transcript.append(tr_assistant)
                final_rationale = turn["rationale"]
                thought_blob = build_agent_thought_text(turn)
                if self._runtime_episode_memory and thought_blob.strip():
                    self._runtime_episode_memory.append(
                        "thought",
                        {"text": thought_blob[:8000]},
                        round_idx=round_idx,
                    )
                if turn["status"] == "done":
                    gate = visual_review_done_gate(goal, worklog)
                    if gate:
                        messages.append({"role": "assistant", "content": raw[:12000]})
                        messages.append({"role": "user", "content": gate})
                        rounds_used += 1
                        await _maybe_sliding_compact()
                        continue
                    low = final_rationale.lower()
                    success = not any(
                        w in low
                        for w in (
                            "failed",
                            "failure",
                            "could not",
                            "cannot",
                            "unable",
                            "not successful",
                            "still broken",
                        )
                    )
                    rounds_used += 1
                    terminated_with_done = True
                    break

                if self._enforce_think_before_act:
                    mono_err = monologue_validation_error(turn)
                    if mono_err:
                        messages.append({"role": "assistant", "content": raw[:12000]})
                        messages.append({"role": "user", "content": mono_err})
                        rounds_used += 1
                        await _maybe_sliding_compact()
                        continue

                thought = build_agent_thought_text(turn)

                calls = turn["tool_calls"][: self._max_tool_calls_per_turn]
                if not calls:
                    messages.append({"role": "assistant", "content": raw[:12000]})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You set `status` to `continue` but provided no `tool_calls`. "
                                "Emit tools or set `status` to `done`."
                            ),
                        }
                    )
                    rounds_used += 1
                    await _maybe_sliding_compact()
                    continue

                results: list[dict[str, Any]] = []
                diag_hints: list[dict[str, str]] = []
                for c in calls:
                    name = str(c.get("name", ""))
                    args = c.get("args") or {}
                    if name not in self.allowed_tools():
                        results.append({"tool": name, "ok": False, "error": f"unknown tool {name!r}"})
                        res = results[-1]
                    else:
                        res = await self._exec_tool_async(name, args)
                        results.append(res)
                    if self._runtime_episode_memory:
                        self._runtime_episode_memory.append(
                            "action",
                            {"tool": name, "args": dict(args) if isinstance(args, dict) else {}},
                            round_idx=round_idx,
                        )
                        self._runtime_episode_memory.append(
                            "observation",
                            {
                                "tool": name,
                                "ok": bool(res.get("ok", True)),
                                "summary": json.dumps(res, default=str, ensure_ascii=False)[:6000],
                            },
                            round_idx=round_idx,
                        )
                    bundle = recovery_tracker.record_tool_observation(name, res)
                    worklog.record_action_observation(
                        round_idx=round_idx,
                        thought=thought,
                        tool_name=name,
                        args=dict(args) if isinstance(args, dict) else {},
                        result=res,
                        prior_round_had_diagnostic_injection=prior_round_diagnostic,
                        diagnostic_bundle=bundle,
                    )
                    if bundle and (bundle.system_block or bundle.incremental_hint):
                        diag_hints.append(
                            {
                                "tool": name,
                                "system_block": bundle.system_block,
                                "incremental_hint": bundle.incremental_hint,
                                "error_fingerprint": bundle.error_fingerprint,
                            }
                        )

                obs = "## Tool results\n```json\n" + json.dumps(results, indent=2) + "\n```"
                obs += terminal_failure_followup_hints(results)
                if diag_hints:
                    obs += "\n\n## Orchestrator (repeated-failure observation)\n" + json.dumps(
                        diag_hints, indent=2
                    )
                messages.append({"role": "assistant", "content": raw[:12000]})
                obs_limit = tool_observation_truncation_limit()
                messages.append({"role": "user", "content": _truncate(obs, obs_limit)})
                tr_row: dict[str, Any] = {"round": round_idx, "tool_results": results}
                if diag_hints:
                    tr_row["diagnostic_hints"] = diag_hints
                transcript.append(tr_row)
                prior_round_diagnostic = bool(diag_hints)
                rounds_used += 1
                await _maybe_sliding_compact()
        finally:
            await self._browser_session_shutdown_async()

        if not terminated_with_done:
            success = False
            if not final_rationale.strip():
                final_rationale = "Stopped: max rounds without status done."

        if not final_rationale:
            final_rationale = "(no terminal rationale)"

        worklog_payload = worklog.to_dict()
        worklog_payload["final_rationale"] = final_rationale
        worklog_payload["success"] = success
        worklog_payload["rounds_used"] = rounds_used
        worklog_payload["recovery_paths"] = [asdict(r) for r in recovery_tracker.recovery_paths]
        try:
            export_worklog_json(worklog_payload, self._ws.root)
        except OSError:
            worklog_payload["export_error"] = "failed to write worklog.json (disk full or permissions)"

        if self._runtime_episode_memory:
            self._runtime_episode_memory.append(
                "meta",
                {"event": "session_end", "success": success, "rounds": rounds_used},
                round_idx=None,
            )
        self._runtime_code_index = None
        self._runtime_episode_memory = None
        self._runtime_shell_tracker = None

        return UnifiedSessionResult(
            success=success,
            goal=goal,
            rationale=final_rationale,
            rounds=rounds_used,
            workspace_root=str(self._ws.root),
            transcript=transcript,
            plan_markdown=plan_md,
            post_mortem_markdown=recovery_tracker.format_report(goal=goal),
            worklog=worklog_payload,
        )

    async def _exec_tool_async(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "fs_list":
                path = str(args.get("path", ".") or ".")
                items = self._ws.list_dir(path)
                return {"tool": name, "ok": True, "path": path, "entries": items}
            if name == "fs_read":
                path = str(args["path"])
                text = self._ws.read_file(path)
                return {"tool": name, "ok": True, "path": path, "content": _truncate(text, 10000)}
            if name == "fs_write":
                path = str(args["path"])
                content = str(args.get("content", ""))
                self._ws.write_file(path, content)
                return {"tool": name, "ok": True, "path": path, "bytes": len(content.encode("utf-8"))}
            if name == "fs_delete":
                path = str(args["path"])
                self._ws.delete_path(path)
                return {"tool": name, "ok": True, "path": path}
            if name == "sandbox_run":
                return await self._tool_sandbox_run_async(args)
            if name == "sandbox_shell":
                return await self._tool_sandbox_shell_async(args)
            if name == "live_preview":
                return await self._tool_live_preview_async(args)
            if name == "github":
                from plodder.orchestration.github_tool_dispatch import run_github_tool_for_workspace

                return await run_github_tool_for_workspace(str(self._ws.root), args)
            if name == "gitleaks":
                return await self._tool_gitleaks_async(args)
            if name == "search_codebase":
                q = str(args.get("query", "")).strip()
                if not q:
                    return {"tool": name, "ok": False, "error": "query is required"}
                top_k = int(args.get("top_k", 8))
                idx = self._runtime_code_index
                if idx is None:
                    return {
                        "tool": name,
                        "ok": False,
                        "error": "workspace code index not available (indexing failed or disabled)",
                    }
                hits = idx.search(q, top_k=top_k)
                return {"tool": name, "ok": True, "query": q, "hits": hits}
            if name == "atomic_edit":
                rel = str(args.get("path", "")).strip()
                mode = str(args.get("mode", "str_replace")).strip().lower()
                if mode not in ("str_replace", "write_full"):
                    return {"tool": name, "ok": False, "error": "mode must be str_replace or write_full"}
                try:
                    out = atomic_edit(
                        self._ws,
                        rel,
                        mode=mode,  # type: ignore[arg-type]
                        old_string=args.get("old_string"),
                        new_string=args.get("new_string"),
                        content=args.get("content"),
                    )
                    return {"tool": name, "ok": True, **out}
                except Exception as ex:
                    return {"tool": name, "ok": False, "error": str(ex)}
            if name == "lsp_check":
                rel = str(args.get("path", "")).strip()
                if not rel:
                    return {"tool": name, "ok": False, "error": "path is required"}
                from mini_devin.lsp.diagnostics import collect_diagnostics

                content = args.get("content")
                c = str(content) if content is not None else None
                items, src = collect_diagnostics(str(self._ws.root), rel, c)
                err_n = sum(1 for d in items if d.get("severity") == "error")
                return {
                    "tool": name,
                    "ok": err_n == 0,
                    "path": rel,
                    "source": src,
                    "diagnostics": items[:80],
                    "error_count": err_n,
                }
            if name == "playwright_observe":
                nav_url = self._nav_url_from_args(args)
                kw = self._observe_bundle_kwargs(args)

                def _observe(s: Any) -> dict[str, Any]:
                    return s.observe_bundle(url=nav_url, **kw)

                data = await self._browser_exec(_observe)
                return {"tool": name, **data}
            if name == "browser_click":
                kw = self._observe_bundle_kwargs(args)
                post_wait = _clamp_int(args.get("post_wait_ms"), 200, 8000, 600)
                verify = bool(args.get("verify", True))
                vimg = bool(args.get("verify_include_screenshot", False))
                eid = str(args.get("element_id", ""))

                def _click(s: Any) -> dict[str, Any]:
                    return s.click_element(
                        eid,
                        post_wait_ms=post_wait,
                        verify=verify,
                        verify_include_screenshot=vimg,
                        capture_console=kw["capture_console"],
                        include_accessibility=kw["include_accessibility"],
                        max_elements=kw["max_elements"],
                        full_page=kw["full_page"],
                    )

                data = await self._browser_exec(_click)
                return {"tool": name, **data}
            if name == "browser_type":
                kw = self._observe_bundle_kwargs(args)
                post_wait = _clamp_int(args.get("post_wait_ms"), 200, 8000, 600)
                verify = bool(args.get("verify", True))
                vimg = bool(args.get("verify_include_screenshot", False))
                submit = bool(args.get("submit", False))
                eid = str(args.get("element_id", ""))
                text = str(args.get("text", ""))

                def _type(s: Any) -> dict[str, Any]:
                    return s.type_element(
                        eid,
                        text,
                        submit=submit,
                        post_wait_ms=post_wait,
                        verify=verify,
                        verify_include_screenshot=vimg,
                        capture_console=kw["capture_console"],
                        include_accessibility=kw["include_accessibility"],
                        max_elements=kw["max_elements"],
                        full_page=kw["full_page"],
                    )

                data = await self._browser_exec(_type)
                return {"tool": name, **data}
            if name == "browser_scroll":
                direction = str(args.get("direction", "down")).strip().lower()
                if direction not in ("up", "down", "top", "bottom"):
                    return {
                        "tool": name,
                        "ok": False,
                        "error": "direction must be one of: up, down, top, bottom",
                    }
                pixels = _clamp_int(args.get("pixels"), 50, 4000, 600)

                def _scroll(s: Any) -> dict[str, Any]:
                    return s.scroll_viewport(direction, pixels=pixels)  # type: ignore[arg-type]

                data = await self._browser_exec(_scroll)
                return {"tool": name, **data}
            if name == "browser_close":
                await self._browser_session_shutdown_async()
                return {"tool": name, "ok": True}
        except Exception as e:  # noqa: BLE001 — surface to model
            return {"tool": name, "ok": False, "error": str(e), "args": args}
        return {"tool": name, "ok": False, "error": "unreachable"}

    async def _tool_sandbox_run_async(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._sandbox is None:
            return {"tool": "sandbox_run", "ok": False, "error": "sandbox not configured"}
        entry_raw = str(args.get("entry") or "").strip()
        if not entry_raw:
            return {"tool": "sandbox_run", "ok": False, "error": "entry is required"}
        lang = args.get("language")
        language = None if lang in (None, "", "auto") else str(lang)
        lk_raw = args.get("language_key")
        language_key = str(lk_raw).strip() if lk_raw not in (None, "") else None
        network = bool(args.get("network", False))
        files = self._ws.snapshot_text_files()
        if not files:
            return {"tool": "sandbox_run", "ok": False, "error": "workspace snapshot empty"}
        entry = entry_raw.replace("\\", "/").lstrip("/")
        resolved: str | None = None
        if entry in files:
            resolved = entry
        else:
            for k in files:
                kn = k.replace("\\", "/")
                if kn == entry or kn.endswith("/" + entry):
                    resolved = kn
                    break
        if resolved is None:
            resolved = pick_default_entry(files)
        if resolved is None:
            return {"tool": "sandbox_run", "ok": False, "error": f"entry {entry_raw!r} not found in snapshot"}

        result = await asyncio.to_thread(
            self._sandbox.run_detected,
            files,
            entry=resolved,
            language=language,
            language_key=language_key,
            timeout_sec=self._sandbox_timeout_sec,
            network=network,
        )
        return self._sandbox_result_dict("sandbox_run", result)

    async def _tool_live_preview_async(self, args: dict[str, Any]) -> dict[str, Any]:
        """
        Register a listening TCP port on **this host's** ``127.0.0.1`` for the session Live Preview iframe.

        Matches :func:`mini_devin.orchestrator.agent.Agent._execute_tool` ``live_preview`` behaviour;
        requires ``session_id`` (constructor or ``PLODDER_SESSION_ID``).
        """
        from mini_devin.api.live_preview_state import (
            allowed_ports,
            probe_local_ports_sync,
            set_session_preview_port,
        )

        sid = self._session_id
        if not sid:
            return {
                "tool": "live_preview",
                "ok": False,
                "error": (
                    "session_id not set — pass ``session_id`` to ``UnifiedSessionDriver`` / ``AgentLoopConfig``, "
                    "set env ``PLODDER_SESSION_ID``, or run inside ``plodder_session_id_binding(api_session_id)`` "
                    "(``SessionManager.run_task`` does this automatically)."
                ),
            }

        action = str(args.get("action", "probe") or "probe").lower()
        if action == "probe":
            ports = args.get("ports")
            if not isinstance(ports, list) or not ports:
                raw = (os.environ.get("LIVE_PREVIEW_PROBE_PORTS") or "5173,3000,8080,4200,8000").strip()
                ports = []
                for x in raw.split(","):
                    x = x.strip()
                    if x.isdigit():
                        ports.append(int(x))
            try:
                candidates = [int(p) for p in ports]
            except (TypeError, ValueError):
                candidates = [5173, 3000, 8080]
            listening = probe_local_ports_sync(candidates)
            return {
                "tool": "live_preview",
                "ok": True,
                "action": "probe",
                "listening_ports": listening,
                "allowed_ports": sorted(allowed_ports()),
                "next_step": (
                    "If a port listed is your dev server, call live_preview with action set_active_port "
                    "and port=<that int> so the Browser tab iframe can load /api/sessions/<id>/live-preview/."
                ),
                "note": (
                    "Ports must accept TCP on 127.0.0.1 on **this** machine. Ephemeral Docker sandbox_shell "
                    "commands do not expose listener ports here — use host process sandbox, port publish, or a tunnel URL."
                ),
            }

        if action == "set_active_port":
            try:
                p = int(args.get("port", 0))
            except (TypeError, ValueError):
                return {"tool": "live_preview", "ok": False, "error": "invalid port", "action": "set_active_port"}
            ok = await set_session_preview_port(sid, p)
            if not ok:
                return {
                    "tool": "live_preview",
                    "ok": False,
                    "action": "set_active_port",
                    "error": f"Port {p} not allowed or not accepting TCP on 127.0.0.1",
                    "allowed_ports": sorted(allowed_ports()),
                }
            return {
                "tool": "live_preview",
                "ok": True,
                "action": "set_active_port",
                "active_port": p,
                "browser_iframe": f"/api/sessions/{sid}/live-preview/",
            }

        return {"tool": "live_preview", "ok": False, "error": f"unknown live_preview action {action!r}"}

    async def _tool_gitleaks_async(self, args: dict[str, Any]) -> dict[str, Any]:
        import shutil

        root = str(self._ws.root)
        exe = shutil.which("gitleaks")
        if not exe:
            return {
                "tool": "gitleaks",
                "ok": False,
                "error": "gitleaks not on PATH — install from https://github.com/gitleaks/gitleaks/releases",
            }
        extra: list[str] = []
        if isinstance(args.get("extra_args"), list):
            extra = [str(x) for x in args["extra_args"] if str(x).strip()]
        cmd = [exe, "detect", "--source", root, "--redact", *extra]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await proc.communicate()
        stdout = (out_b or b"").decode("utf-8", errors="replace")
        stderr = (err_b or b"").decode("utf-8", errors="replace")
        return {
            "tool": "gitleaks",
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": _truncate(stdout, 8000),
            "stderr": _truncate(stderr, 4000),
        }

    async def _tool_sandbox_shell_async(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._sandbox is None:
            return {"tool": "sandbox_shell", "ok": False, "error": "sandbox not configured"}
        argv = args.get("argv")
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            return {"tool": "sandbox_shell", "ok": False, "error": "argv must be list[str]"}
        hint = args.get("language_hint")
        language_hint = str(hint) if hint is not None else None
        network = bool(args.get("network", False))
        files = self._ws.snapshot_text_files()
        argv_list = list(argv)
        inner = extract_shell_inner(argv_list)
        if self._runtime_shell_tracker is not None:
            argv_list = self._runtime_shell_tracker.wrap_argv(argv_list)
        result = await asyncio.to_thread(
            self._sandbox.run_shell_in_workspace,
            files,
            argv_list,
            language_hint=language_hint,
            timeout_sec=self._sandbox_timeout_sec,
            network=network,
        )
        if self._runtime_shell_tracker is not None and inner is not None:
            self._runtime_shell_tracker.ingest_user_command(inner)
        return self._sandbox_result_dict("sandbox_shell", result)

    @staticmethod
    def _sandbox_result_dict(tool: str, result: SandboxResult) -> dict[str, Any]:
        raw_out = result.stdout or ""
        raw_err = result.stderr or ""
        out, out_was_trunc = truncate_stream(raw_out)
        err, err_was_trunc = truncate_stream(raw_err)
        base_ok = result.exit_code == 0 and not result.timed_out
        if base_ok and _shell_scaffold_operation_cancelled(raw_out, raw_err, result.command):
            base_ok = False
        return {
            "tool": tool,
            "ok": base_ok,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "command": result.command,
            "stdout": out,
            "stderr": err,
            "stdout_truncated": out_was_trunc,
            "stderr_truncated": err_was_trunc,
        }
