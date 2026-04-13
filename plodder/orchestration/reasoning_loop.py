"""
OpenHands-style **Observation → Reasoning → Action** contract for the unified JSON driver.

Each turn is a state step: prior tool outputs are **Observations**; structured fields are
**Reasoning**; ``tool_calls`` are **Actions**. The driver enforces a think-before-act monologue
for high-impact tools (terminal + file mutation).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Awaitable, Callable

ChatMessage = dict[str, Any]
LLMFn = Callable[[list[ChatMessage]], Awaitable[str]]

# OpenHands-style sliding window: full detail for the last N tool-observation rounds.
SLIDING_WINDOW_TOOL_ROUNDS = int(os.environ.get("PLODDER_SLIDING_WINDOW_ROUNDS", "20") or "20")
SLIDING_WINDOW_TOOL_ROUNDS = max(3, min(SLIDING_WINDOW_TOOL_ROUNDS, 50))

RUNNING_SUMMARY_SYSTEM_PROMPT = """You maintain a **Running Summary** for a coding agent (Plodder / OpenHands-style).

Rules:
1. Output **plain markdown only** (no JSON, no tool calls). Be dense and factual.
2. **Never replace or paraphrase the user's Goal** — that text is kept separately; this summary covers *older* turns only.
3. **Preserve verbatim detail** for:
   - **LSP / type errors** (file paths, line numbers, error codes, messages)
   - **Browser console logs** and **page errors** from Playwright / UI observe tools
   - **Terminal failures** (exit codes, last lines of stderr)
4. Keep: files created/edited, key decisions, open bugs, and what remains to be done.
5. Merge the **Existing running summary** with the **New segment** into one updated summary (deduplicate stale facts).
"""


def sliding_window_memory_disabled() -> bool:
    """When true, the session driver keeps the full message list (legacy behavior)."""
    v = (os.environ.get("PLODDER_SLIDING_WINDOW_DISABLE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def tool_observation_truncation_limit() -> int:
    """
    Max chars for a single ``## Tool results`` user block.

    With sliding memory enabled, default is high so Gemini-scale models keep full **LSP**
    diagnostics and **browser console** payloads in the active (non-summarized) tail.
    """
    if sliding_window_memory_disabled():
        return int(os.environ.get("PLODDER_TOOL_OBS_MAX_CHARS_LEGACY", "14000") or "14000")
    raw = (os.environ.get("PLODDER_TOOL_OBS_MAX_CHARS") or "").strip()
    if raw.isdigit():
        return max(16_000, min(int(raw), 500_000))
    return 200_000


def _is_tool_observation_user(msg: ChatMessage) -> bool:
    if msg.get("role") != "user":
        return False
    c = str(msg.get("content") or "")
    return (
        "## Tool results" in c
        or "## Parse error" in c
        or "## Think-before-act" in c
        or "## Visual review" in c
        or "You set `status` to `continue` but provided no `tool_calls`" in c
    )


def _is_context_injection_user(msg: ChatMessage) -> bool:
    if msg.get("role") != "user":
        return False
    c = str(msg.get("content") or "")
    return c.startswith("## Episode continuity") or c.startswith("## Long-horizon continuity")


def _is_running_summary_user(msg: ChatMessage) -> bool:
    if msg.get("role") != "user":
        return False
    c = str(msg.get("content") or "")
    return "## Running summary (sliding-window memory" in c


def partition_messages_sliding_window(
    messages: list[ChatMessage],
    *,
    max_tool_rounds: int = SLIDING_WINDOW_TOOL_ROUNDS,
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
    """
    Split ``messages`` into immutable prefix, stale middle, and full-detail tail.

    - **prefix**: ``[system, first_user]`` — the Goal seed and system prompt; **never** summarized.
    - **stale**: older turns to fold into the running summary.
    - **tail**: the last ``max_tool_rounds`` tool-observation cycles (plus their paired assistants
      and any **context injection** user blocks immediately before each kept assistant).
    """
    if len(messages) < 2:
        return list(messages), [], []
    prefix = messages[:2]
    rest = messages[2:]
    if not rest:
        return prefix, [], []

    obs_idx = [i for i, m in enumerate(rest) if _is_tool_observation_user(m)]
    if len(obs_idx) <= max_tool_rounds:
        return prefix, [], list(rest)

    first_kept = obs_idx[len(obs_idx) - max_tool_rounds]
    tail_start = first_kept
    if tail_start > 0 and rest[tail_start - 1].get("role") == "assistant":
        tail_start -= 1
    else:
        j = tail_start - 1
        while j >= 0:
            if rest[j].get("role") == "assistant":
                tail_start = j
                break
            j -= 1
    j = tail_start - 1
    while j >= 0 and rest[j].get("role") == "user" and _is_context_injection_user(rest[j]):
        tail_start = j
        j -= 1

    stale = rest[:tail_start]
    tail = rest[tail_start:]
    return prefix, stale, tail


def flatten_messages_for_summary(messages: list[ChatMessage], *, max_chars: int = 120_000) -> str:
    """Linearize assistant/user messages for summarization (cap size for the summarizer call)."""
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", "") or "")
        lines.append(f"### {role.upper()}\n{content}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 80] + "\n\n…(stale segment truncated for summarizer input)…\n"
    return text


def build_running_summary_llm_messages(
    existing_summary: str,
    stale_segment_text: str,
) -> list[ChatMessage]:
    """Two-turn messages for the summarizer (same LLM as the agent; must return prose only)."""
    user = (
        "## Existing running summary\n"
        + (existing_summary.strip() or "(none — first compaction)")
        + "\n\n## New conversation segment to fold into the summary\n"
        + stale_segment_text
        + "\n\n## Instructions\n"
        "Write the **updated** running summary in markdown only. "
        "Keep **all** LSP diagnostic lines and browser console / page-error lines from the segment "
        "(or explicitly quote them). Do not exceed ~8000 words."
    )
    return [
        {"role": "system", "content": RUNNING_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def merge_prefix_summary_and_tail(
    prefix: list[ChatMessage],
    tail: list[ChatMessage],
    running_summary: str,
) -> list[ChatMessage]:
    """Rebuild ``messages``: system + Goal user + optional summary user + tail."""
    out: list[ChatMessage] = list(prefix)
    rs = running_summary.strip()
    if rs:
        out.append(
            {
                "role": "user",
                "content": (
                    "## Running summary (sliding-window memory; OpenHands-style)\n"
                    "The **first user message** in this thread (workspace + **Goal**) is unchanged and remains "
                    "the source of truth for the task.\n"
                    "Below is a distilled memory of **older** turns. **LSP errors** and **browser console / "
                    "page errors** from those turns are included here so they stay available alongside the "
                    "last "
                    f"{SLIDING_WINDOW_TOOL_ROUNDS} full tool rounds in the tail.\n\n"
                    + rs
                ),
            }
        )
    out.extend(tail)
    return out


async def async_refresh_running_summary(
    llm: LLMFn,
    existing_summary: str,
    stale_messages: list[ChatMessage],
) -> str:
    """Call the LLM once to fold ``stale_messages`` into ``existing_summary``."""
    if not stale_messages:
        return existing_summary
    stale_eff = [m for m in stale_messages if not _is_running_summary_user(m)]
    if not stale_eff:
        return existing_summary
    stale_text = flatten_messages_for_summary(stale_eff)
    msgs = build_running_summary_llm_messages(existing_summary, stale_text)
    raw = await llm(msgs)
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = _strip_json_fence(text)
    return text[:48_000]


async def apply_sliding_window_to_messages(
    messages: list[ChatMessage],
    llm: LLMFn,
    running_summary: str,
) -> str:
    """
    If the message list exceeds the sliding window, summarize ``stale`` via ``llm`` and
    replace ``messages`` contents with ``prefix + summary user + tail``.

    Returns the updated ``running_summary`` string.
    """
    if sliding_window_memory_disabled():
        return running_summary
    prefix, stale, tail = partition_messages_sliding_window(messages)
    if not stale:
        return running_summary
    try:
        new_summary = await async_refresh_running_summary(llm, running_summary, stale)
    except Exception:
        return running_summary
    merged = merge_prefix_summary_and_tail(prefix, tail, new_summary)
    messages.clear()
    messages.extend(merged)
    return new_summary


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

### Sliding-window memory (OpenHands-style)

Older turns may be folded into a **Running summary** user message (after the Goal). The **first
user message** (workspace + **Goal**) is never removed. The last **20** tool-observation rounds
stay verbatim; **LSP diagnostics** and **browser console** lines from older rounds are kept in
the summary text so they remain actionable with Gemini-scale context.
"""


def extract_shell_inner(argv: list[str]) -> str | None:
    """Return the inner script for ``sh -c`` / ``bash -lc``, else ``None``."""
    if len(argv) >= 3 and argv[0] in ("sh", "bash") and argv[1] in ("-c", "-lc"):
        return str(argv[2])
    return None


__all__ = [
    "SLIDING_WINDOW_TOOL_ROUNDS",
    "RUNNING_SUMMARY_SYSTEM_PROMPT",
    "REASONING_LOOP_SEED_SUFFIX",
    "TOOLS_REQUIRING_MONOLOGUE",
    "apply_sliding_window_to_messages",
    "async_refresh_running_summary",
    "build_agent_thought_text",
    "build_running_summary_llm_messages",
    "extract_shell_inner",
    "flatten_messages_for_summary",
    "goal_suggests_frontend_stack",
    "merge_prefix_summary_and_tail",
    "monologue_validation_error",
    "parse_driver_turn",
    "partition_messages_sliding_window",
    "path_suggests_ui_surface",
    "shell_argv_suggests_dev_server",
    "sliding_window_memory_disabled",
    "terminal_failure_followup_hints",
    "tool_observation_truncation_limit",
    "visual_review_done_gate",
    "worklog_has_ui_mutation",
]
