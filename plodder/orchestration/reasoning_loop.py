"""
OpenHands-style **Observation → Reasoning → Action** contract for the unified JSON driver.

Each turn is a state step: prior tool outputs are **Observations**; structured fields are
**Reasoning**; ``tool_calls`` are **Actions**. The driver enforces a think-before-act monologue
for high-impact tools (terminal + file mutation).
"""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_json_fence(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_driver_turn(raw: str) -> dict[str, Any]:
    """Parse one assistant JSON object for the unified driver; raise ValueError on invalid shape."""
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
    out: dict[str, Any] = {
        "status": status,
        "rationale": str(data.get("rationale", "")),
        "tool_calls": calls,
    }
    for k in ("observe", "think", "act_summary", "sub_goal", "risk_assessment", "expected_outcome"):
        if k in data and data[k] is not None:
            out[k] = str(data[k])[:4000]
    return out


# Tools that must be preceded by an explicit monologue in the same JSON object.
TOOLS_REQUIRING_MONOLOGUE = frozenset(
    {
        "sandbox_shell",
        "sandbox_run",
        "atomic_edit",
        "fs_write",
        "fs_delete",
    }
)

# Minimum non-whitespace length per monologue field (keeps one-word stubs from passing).
_MONOLOGUE_MIN_LEN = 12

_FRONTEND_GOAL_PAT = re.compile(
    r"(?i)\b(react|vite|next\.?js|nextjs|tailwind|dashboard|frontend|spa|vue|svelte|"
    r"npm\s+install|package\.json|playwright|component|tsx|jsx)\b"
)

_DEV_SERVER_PAT = re.compile(
    r"(?i)\b(npm\s+run\s+dev|yarn\s+dev|pnpm\s+dev|vite\b|webpack-dev|next\s+dev|"
    r"astro\s+dev|parcel\b)\b"
)


def goal_suggests_frontend_stack(goal: str) -> bool:
    """Heuristic: UI / Node dashboard work → incremental build + LSP + browser diagnostics."""
    g = (goal or "").strip()
    return bool(_FRONTEND_GOAL_PAT.search(g))


def shell_argv_suggests_dev_server(argv: list[str]) -> bool:
    """True if a sandbox_shell argv looks like starting a long-lived dev server."""
    if not argv:
        return False
    inner = ""
    if len(argv) >= 3 and argv[0] in ("sh", "bash") and argv[1] in ("-c", "-lc"):
        inner = str(argv[2])
    else:
        inner = " ".join(str(x) for x in argv)
    return bool(_DEV_SERVER_PAT.search(inner))


def terminal_failure_followup_hints(results: list[dict[str, Any]]) -> str:
    """OpenHands-style nudge after a failed terminal observation (first failure, not only streaks)."""
    for res in results:
        if res.get("ok", True):
            continue
        tool = str(res.get("tool", ""))
        if tool not in ("sandbox_shell", "sandbox_run"):
            continue
        lines = [
            "",
            "## OpenHands retry-with-diagnostic",
            "Before repeating the same terminal command: run **lsp_check** on paths mentioned in stderr "
            "or on files you recently edited; for web/runtime failures use **playwright_observe** with "
            "`capture_console: true` on the dev URL to read **browser console** output and uncaught page errors.",
        ]
        argv = res.get("argv")
        cmd = str(res.get("command", "") or "")
        devish = False
        if isinstance(argv, list):
            devish = shell_argv_suggests_dev_server([str(x) for x in argv])
        if not devish and cmd:
            devish = shell_argv_suggests_dev_server(["sh", "-c", cmd])
        if devish:
            lines.append(
                "This failure pattern matches a **dev server** class command (`npm run dev`, `vite`, …): "
                "prefer **playwright_observe** with `capture_console: true` before the next edit cycle."
            )
        return "\n".join(lines)
    return ""


def monologue_validation_error(turn: dict[str, Any]) -> str | None:
    """
    If the turn schedules any ``TOOLS_REQUIRING_MONOLOGUE`` call, require OpenHands-style
    **sub_goal**, **risk_assessment**, and **expected_outcome** on the same JSON object.

    Returns a remediation message for the model, or ``None`` if OK / no such tools.
    """
    if str(turn.get("status", "")) == "done":
        return None
    calls = turn.get("tool_calls") or []
    if not isinstance(calls, list):
        return None
    needs = any(isinstance(c, dict) and str(c.get("name", "")) in TOOLS_REQUIRING_MONOLOGUE for c in calls)
    if not needs:
        return None
    missing: list[str] = []
    for key in ("sub_goal", "risk_assessment", "expected_outcome"):
        raw = str(turn.get(key, "") or "").strip()
        if len(raw) < _MONOLOGUE_MIN_LEN:
            missing.append(key)
    if not missing:
        return None
    return (
        "## Think-before-act (required)\n"
        "Your JSON scheduled terminal or file-mutation tools without a complete **monologue**.\n"
        "On the **same** JSON object as ``tool_calls``, include these string fields "
        f"(each ≥ {_MONOLOGUE_MIN_LEN} characters):\n"
        "- ``sub_goal``: concrete outcome for this turn.\n"
        "- ``risk_assessment``: what could go wrong (paths, deps, breaking changes).\n"
        "- ``expected_outcome``: what you expect to observe after tools run "
        "(stdout patterns, files created, diagnostics cleared).\n"
        f"Missing or too short: {', '.join(missing)}.\n"
        "Reply again with **only** one valid JSON object."
    )


def build_agent_thought_text(turn: dict[str, Any]) -> str:
    """Flatten rationale + optional O–R–A fields for worklog / episode memory."""
    parts: list[str] = []
    r = str(turn.get("rationale", "")).strip()
    if r:
        parts.append(r)
    order = (
        "observe",
        "think",
        "sub_goal",
        "risk_assessment",
        "expected_outcome",
        "act_summary",
    )
    for k in order:
        v = str(turn.get(k, "") or "").strip()
        if v:
            parts.append(f"{k}: {v}")
    return "\n\n".join(parts) if parts else ""


REASONING_LOOP_SEED_SUFFIX = """
## State machine (OpenHands-style)

Each assistant JSON is one step: **Observation** (what you learned from prior tool results),
**Reasoning** (plan + risks), **Action** (``tool_calls``). Never assume a path exists—confirm
with ``fs_list`` / ``fs_read`` or ``sandbox_shell`` mapping before mutating files.

### Think-before-act (monologue)

Whenever you call **any** of:
``sandbox_shell``, ``sandbox_run``, ``atomic_edit``, ``fs_write``, ``fs_delete``,
the **same** JSON object **must** include:

- ``sub_goal``: what this turn accomplishes (one concrete sub-goal).
- ``risk_assessment``: what might fail and how you will detect it.
- ``expected_outcome``: what you expect to see in the next observation.

Optional trace keys (same object): ``observe``, ``think``, ``act_summary`` (short strings).

### Environmental grounding (feet & eyes)

Before large UI or React/Vite work: map the tree (``fs_list`` / ``fs_read``) and use
``sandbox_shell`` with ``pwd`` and a **bounded** recursive listing (e.g.
``find . -maxdepth 5 -type f | head -n 200`` or ``ls -R`` piped through ``head``) so paths
match the real workspace.

### Retry-with-diagnostic

If a terminal command fails: **do not** blindly repeat it. Run ``lsp_check`` on edited paths
cited in stderr; for UI/runtime issues run ``playwright_observe`` with ``capture_console: true``
to capture **browser console** lines. Then fix and retry.

### Atomic incremental development (hands)

For Node/React dashboards, avoid writing the entire app in one turn:

1. **Step A** — ``package.json`` + lockfile strategy, then ``npm install`` / ``pnpm install`` (``network: true``).
2. **Step B** — config only (e.g. ``tailwind.config.*``, ``vite.config.*``).
3. **Step C** — **one** component or route at a time; after each substantive edit, ``lsp_check``
   on the touched file before moving on.

### Long-term alignment

Keep actions consistent with **PLAN.md** at the workspace root (if present) and with the
running **worklog** summary you see each turn—preserve the overall goal across many rounds.

Before ``status: "done"``, run **lsp_check** on files you edited and **playwright_observe**
(with console when relevant) before claiming UI/code success.
"""


def extract_shell_inner(argv: list[str]) -> str | None:
    """Return the inner script for ``sh -c`` / ``bash -lc``, else ``None``."""
    if len(argv) >= 3 and argv[0] in ("sh", "bash") and argv[1] in ("-c", "-lc"):
        return str(argv[2])
    return None


__all__ = [
    "REASONING_LOOP_SEED_SUFFIX",
    "TOOLS_REQUIRING_MONOLOGUE",
    "build_agent_thought_text",
    "extract_shell_inner",
    "goal_suggests_frontend_stack",
    "monologue_validation_error",
    "parse_driver_turn",
    "shell_argv_suggests_dev_server",
    "terminal_failure_followup_hints",
]
