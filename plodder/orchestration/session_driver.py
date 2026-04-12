"""
Unified Session Driver — multi-turn tool use (filesystem + sandbox) for Plodder.

The model emits strict JSON with ``tool_calls``; the driver executes them against a
``SessionWorkspace`` and optional ``ExecutionSandbox``, then feeds results back until
``status: done`` or max rounds.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from plodder.core.universal_prompt_engine import PolyglotSystemPrompt, UniversalPromptEngine
from plodder.sandbox.execution_sandbox import SandboxResult
from plodder.sandbox.toolchain_detect import pick_default_entry
from plodder.workspace.session_workspace import SessionWorkspace


ChatMessage = dict[str, Any]
LLMFn = Callable[[list[ChatMessage]], Awaitable[str]]


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


def _strip_json_fence(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_driver_turn(raw: str) -> dict[str, Any]:
    """Parse one assistant JSON object; raise ValueError on invalid shape."""
    data = json.loads(_strip_json_fence(raw))
    if not isinstance(data, dict):
        raise ValueError("root must be object")
    status = data.get("status", "continue")
    if status not in ("continue", "done"):
        raise ValueError("status must be continue|done")
    calls = data.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be list")
    for c in calls:
        if not isinstance(c, dict) or "name" not in c:
            raise ValueError("each tool_call needs name")
        if "args" not in c or not isinstance(c["args"], dict):
            c["args"] = {}
    return {
        "status": status,
        "rationale": str(data.get("rationale", "")),
        "tool_calls": calls,
    }


@dataclass
class UnifiedSessionResult:
    success: bool
    goal: str
    rationale: str
    rounds: int
    workspace_root: str
    transcript: list[dict[str, Any]] = field(default_factory=list)
    plan_markdown: str = ""


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
    }
)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 20] + "\n…(truncated)…\n"


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
    ) -> None:
        self._llm = llm
        self._ws = workspace
        self._sandbox = sandbox
        self._engine = engine or UniversalPromptEngine(PolyglotSystemPrompt())
        self._max_rounds = max_rounds
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._sandbox_timeout_sec = sandbox_timeout_sec
        self._inject_logic_plan = inject_logic_plan

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
            "Implement using tools. Prefer small edits. "
            "When ready to verify, use `sandbox_run` with the correct `entry` path."
        )
        return seed_user

    async def _complete(self, messages: list[ChatMessage]) -> str:
        return await self._llm(messages)

    def _system_content(self) -> str:
        parts = [
            self._engine.persona.base_instruction(),
            self._engine.session_unified_driver_contract(),
            self._engine.session_unified_tools_catalog(),
        ]
        return "\n\n".join(parts)

    async def run(self, goal: str) -> UnifiedSessionResult:
        from plodder.orchestration.self_heal import parse_pseudo_plan_json

        transcript: list[dict[str, Any]] = []
        plan_md = ""

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

        for round_idx in range(self._max_rounds):
            raw = await self._complete(messages)
            transcript.append({"round": round_idx, "assistant_raw": raw[:24000]})
            try:
                turn = parse_driver_turn(raw)
            except (json.JSONDecodeError, ValueError) as e:
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
                continue

            final_rationale = turn["rationale"]
            if turn["status"] == "done":
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
                continue

            results: list[dict[str, Any]] = []
            for c in calls:
                name = str(c.get("name", ""))
                args = c.get("args") or {}
                if name not in self.allowed_tools():
                    results.append({"tool": name, "ok": False, "error": f"unknown tool {name!r}"})
                    continue
                results.append(await self._exec_tool_async(name, args))

            obs = "## Tool results\n```json\n" + json.dumps(results, indent=2) + "\n```"
            messages.append({"role": "assistant", "content": raw[:12000]})
            messages.append({"role": "user", "content": _truncate(obs, 14000)})
            transcript.append({"round": round_idx, "tool_results": results})
            rounds_used += 1

        if not terminated_with_done:
            success = False
            if not final_rationale.strip():
                final_rationale = "Stopped: max rounds without status done."

        if not final_rationale:
            final_rationale = "(no terminal rationale)"

        return UnifiedSessionResult(
            success=success,
            goal=goal,
            rationale=final_rationale,
            rounds=rounds_used,
            workspace_root=str(self._ws.root),
            transcript=transcript,
            plan_markdown=plan_md,
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
            if name == "github":
                from plodder.orchestration.github_tool_dispatch import run_github_tool_for_workspace

                return await run_github_tool_for_workspace(str(self._ws.root), args)
            if name == "gitleaks":
                return await self._tool_gitleaks_async(args)
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
        result = await asyncio.to_thread(
            self._sandbox.run_shell_in_workspace,
            files,
            list(argv),
            language_hint=language_hint,
            timeout_sec=self._sandbox_timeout_sec,
            network=network,
        )
        return self._sandbox_result_dict("sandbox_shell", result)

    @staticmethod
    def _sandbox_result_dict(tool: str, result: SandboxResult) -> dict[str, Any]:
        return {
            "tool": tool,
            "ok": result.exit_code == 0 and not result.timed_out,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "command": result.command,
            "stdout": _truncate(result.stdout or "", 6000),
            "stderr": _truncate(result.stderr or "", 6000),
        }
