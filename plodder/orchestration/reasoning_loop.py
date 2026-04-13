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
    r"npm\s+install|package\.json|playwright|component|tsx|jsx|shadcn|radix|lucide|premium\s+ui)\b"
)

_DEV_SERVER_PAT = re.compile(
    r"(?i)\b(npm\s+run\s+dev|yarn\s+dev|pnpm\s+dev|vite\b|webpack-dev|next\s+dev|"
    r"astro\s+dev|parcel\b)\b"
)

# Paths that count as "UI surface" mutations for mandatory visual review before ``done``.
_UI_MUTATION_PATH_RE = re.compile(
    r"(?i)(\.(tsx|jsx|vue|svelte|css)(\.\w+)?$|tailwind\.config\.[cm]?js|/"
    r"(components?|ui|pages|app|routes|layouts|widgets|styles)/)"
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


def path_suggests_ui_surface(path: str) -> bool:
    """True if a workspace-relative path is likely front-end UI."""
    p = (path or "").strip().replace("\\", "/")
    if not p:
        return False
    return bool(_UI_MUTATION_PATH_RE.search(p))


def worklog_has_ui_mutation(events: list[dict[str, Any]]) -> bool:
    """Scan worklog events for atomic_edit/fs_write on UI-like paths."""
    for e in events:
        if e.get("event_type") != "action_observation":
            continue
        action = e.get("action") or {}
        tool = str(action.get("tool") or "")
        if tool not in ("atomic_edit", "fs_write"):
            continue
        args = action.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        rel = str(args.get("path", "") or "")
        if path_suggests_ui_surface(rel):
            return True
    return False


def _playwright_ok_widths_after_last_ui(events: list[dict[str, Any]]) -> tuple[bool, list[int]]:
    """
    After the last UI mutation, collect viewport widths from successful ``playwright_observe``.

    Returns (has_any_ok_observe, list of widths).
    """
    last_ui_idx = -1
    for i, e in enumerate(events):
        if e.get("event_type") != "action_observation":
            continue
        action = e.get("action") or {}
        tool = str(action.get("tool") or "")
        if tool not in ("atomic_edit", "fs_write"):
            continue
        args = action.get("arguments") or {}
        if isinstance(args, dict) and path_suggests_ui_surface(str(args.get("path", "") or "")):
            last_ui_idx = i

    if last_ui_idx < 0:
        return False, []

    widths: list[int] = []
    for e in events[last_ui_idx + 1 :]:
        if e.get("event_type") != "action_observation":
            continue
        action = e.get("action") or {}
        if str(action.get("tool") or "") != "playwright_observe":
            continue
        obs = e.get("observation") or {}
        raw = obs.get("raw") if isinstance(obs, dict) else None
        if not isinstance(raw, dict) or not raw.get("ok", True):
            continue
        args = action.get("arguments") or {}
        w = args.get("viewport_width") if isinstance(args, dict) else None
        if w is None:
            w = raw.get("viewport_width")
        try:
            widths.append(int(w) if w is not None else 1280)
        except (TypeError, ValueError):
            widths.append(1280)

    return bool(widths), widths


def visual_review_done_gate(goal: str, worklog: object) -> str | None:
    """
    When the goal is front-end-ish and the worklog shows UI file mutations, block ``status: done``
    until the agent ran **playwright_observe** twice after the last UI edit: mobile (~375) and
    desktop (~1440) viewports, so screenshots can be critiqued for layout and contrast.
    """
    if not goal_suggests_frontend_stack(goal):
        return None
    events = list(getattr(worklog, "events", None) or [])
    if not worklog_has_ui_mutation(events):
        return None

    has_observe, widths = _playwright_ok_widths_after_last_ui(events)
    if not has_observe:
        return (
            "## Visual review required (blocking ``done``)\n"
            "You edited **UI/front-end files** but did not run **playwright_observe** after the last edit.\n"
            "1. Ensure the dev server URL is reachable from the host (e.g. `http://127.0.0.1:5173`).\n"
            "2. Call **playwright_observe** with **`viewport_width`: 375** (and optional `viewport_height`: 812**) "
            "and **`capture_console`: true**.\n"
            "3. Call **playwright_observe** again with **`viewport_width`: 1440** (e.g. height 900).\n"
            "4. In your **next** assistant JSON (before ``done``), briefly critique both screenshots: "
            "**alignment**, **text/background contrast**, and **obvious responsive breaks**.\n"
            "Reply with **only** one valid JSON object (`status: continue` + those tool calls), not `done` yet."
        )

    mobile = any(w <= 480 for w in widths)
    desktop = any(w >= 1200 for w in widths)
    if not (mobile and desktop):
        return (
            "## Visual review incomplete (blocking ``done``)\n"
            "After your last UI edit you must capture **two** successful **playwright_observe** runs with "
            "different viewports:\n"
            "- **Mobile audit**: `viewport_width` **375** (height e.g. 812), `capture_console`: true.\n"
            "- **Desktop audit**: `viewport_width` **1440** (height e.g. 900).\n"
            "Then analyze alignment, contrast, and layout breaks before setting ``status`` to ``done``.\n"
            "Reply with **only** one valid JSON object continuing the loop."
        )
    return None


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

### Designer mindset (UX)

Act as a **product designer**, not only a coder:

- **Consistency**: Centralize tokens—**Tailwind** ``tailwind.config.*`` (or CSS variables /
  design tokens)—for **color**, **spacing**, and **typography**. Avoid one-off hex/radius litter.
- **Component-driven build order**: (1) **layout shells** / page frames → (2) **atomic** pieces
  (buttons, cards, inputs) → (3) **composed pages** / routes. Never skip straight to a monolithic page.
- **Micro-interactions**: Every interactive control gets **hover**, **focus-visible**, and
  **motion** via short **transitions** (opacity/transform/shadow)—keyboard users included.
- **Premium UI stack**: Prefer **shadcn/ui** (Radix primitives), **Radix UI** primitives, and
  **Lucide** (or similar) icons over hand-rolled complex CSS when the stack allows (e.g. React + Vite).
- **State-driven UI**: For data views and forms, always implement **loading**, **error**, and
  **empty** states (skeletons/spinners, inline errors, friendly empty copy)—not only the happy path.

### Mandatory visual audit (QA) before ``done``

After you **create or materially change** any UI route/component (``.tsx``/``.jsx``/``.vue``/``.css``
under ``components/``, ``pages/``, ``app/``, etc.):

1. Run **playwright_observe** on the **exact dev URL + route** (e.g. ``http://127.0.0.1:5173/`` or ``/dashboard``).
2. **Critique** the returned screenshot(s) in your rationale before finishing:
   **alignment** (centering, grids, spacing), **contrast** (body text vs background), **responsive**
   obvious breaks.
3. **Two viewports** (separate tool calls), after the last UI edit:
   - **375×812** (mobile) with ``capture_console: true``.
   - **1440×900** (desktop).
   Pass ``viewport_width`` / ``viewport_height`` on ``playwright_observe`` args.
4. If something fails visually, **fix and re-observe** before ``status: "done"``.

The driver may **reject** ``done`` until this audit is satisfied for front-end goals.

### Long-term alignment

Keep actions consistent with **PLAN.md** at the workspace root (if present) and with the
running **worklog** summary you see each turn—preserve the overall goal across many rounds.

Before ``status: "done"``, run **lsp_check** on files you edited, complete the **visual audit**
above when you touched UI, and run **playwright_observe** before claiming UI/code success.
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
    "path_suggests_ui_surface",
    "shell_argv_suggests_dev_server",
    "terminal_failure_followup_hints",
    "visual_review_done_gate",
    "worklog_has_ui_mutation",
]
