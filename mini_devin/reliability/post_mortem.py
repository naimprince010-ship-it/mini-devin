"""
Observation-style post-mortem payloads (OpenHands-aligned).

Structured records for forced diagnostics, failure streaks, and recovery paths.
Use :func:`format_post_mortem` for human-readable markdown (e.g. ``post_mortem.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiagnosticTriggerRecord:
    """Exact fingerprint that caused a forced diagnostic (3× identical tool error)."""

    error_fingerprint: str
    tool_name: str
    exit_code: int | None
    output_preview: str


@dataclass
class FailureStreakRecord:
    """A streak that reached the diagnostic threshold (or max tracked length)."""

    tool_name: str
    error_fingerprint: str
    streak_length: int


@dataclass
class RecoveryPathRecord:
    """Brief narrative of what the agent did after a diagnostic observation."""

    after_error_fingerprint: str
    summary: str
    next_tool: str | None = None
    next_action_preview: str | None = None


@dataclass
class PostMortemPayload:
    """Everything needed to render a session post-mortem."""

    goal_or_task_id: str = ""
    diagnostic_triggers: list[DiagnosticTriggerRecord] = field(default_factory=list)
    recovery_paths: list[RecoveryPathRecord] = field(default_factory=list)
    failure_streaks: list[FailureStreakRecord] = field(default_factory=list)


def _preview_action(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "terminal":
        return str(args.get("command", ""))[:220]
    if tool_name == "editor":
        p = args.get("path", "")
        act = args.get("action", args.get("command", ""))
        return f"{act} {p}"[:220].strip()
    return str(args)[:220]


def infer_recovery_path_summary(
    *,
    after_error_fingerprint: str,
    next_tool_name: str,
    next_args: dict[str, Any],
) -> RecoveryPathRecord:
    """Heuristic one-liner: how behavior changed after a forced diagnostic."""
    preview = _preview_action(next_tool_name, next_args)
    low = preview.lower()

    if next_tool_name == "terminal":
        if any(k in low for k in ("ls", "find", "pwd", "dir ", "dir\n")):
            summary = (
                "Agent shifted from repeating the failing command toward **directory / path verification** "
                "(ls, find, or pwd) after the diagnostic observation."
            )
        elif "pytest" in low or "unittest" in low or " py.test" in low:
            summary = (
                "Agent returned to **test execution** (pytest/unittest) after workspace grounding — "
                "likely with corrected paths or flags."
            )
        elif any(k in low for k in ("npm", "pnpm", "yarn", "node", "npx")):
            summary = (
                "Agent moved to **Node/npm-style checks** after diagnostic (install path or dependency sanity)."
            )
        elif "cat " in low or "package.json" in low:
            summary = "Agent inspected **package manifest** after diagnostic."
        else:
            summary = (
                "Agent continued with a **different terminal command** after diagnostic "
                "(see preview — not a straight repeat of the failing fingerprint)."
            )
    elif next_tool_name == "editor":
        summary = (
            "Agent resumed **editor/file operations** after diagnostic — typically with a corrected path."
        )
    else:
        summary = f"Next successful action used tool `{next_tool_name}` after diagnostic."

    return RecoveryPathRecord(
        after_error_fingerprint=after_error_fingerprint,
        summary=summary,
        next_tool=next_tool_name,
        next_action_preview=preview or None,
    )


def format_post_mortem(payload: PostMortemPayload) -> str:
    """Markdown report for tuning base prompts from real run data."""
    lines: list[str] = [
        "# Session post-mortem (observation-driven)",
        "",
        f"**Task / goal id:** `{payload.goal_or_task_id}`",
        "",
        "## Failure streaks (diagnostic threshold)",
        "",
    ]
    if not payload.failure_streaks:
        lines.append("_No streak reached the forced-diagnostic threshold._")
    else:
        for i, fs in enumerate(payload.failure_streaks, 1):
            lines.append(f"{i}. **{fs.tool_name}** — streak length **{fs.streak_length}**")
            lines.append(f"   - `error_fingerprint`: `{fs.error_fingerprint}`")
            lines.append("")

    lines.extend(
        [
            "## Diagnostic triggers",
            "",
            "Fingerprints that caused a **forced diagnostic** (workspace snapshot + system correction).",
            "",
        ]
    )
    if not payload.diagnostic_triggers:
        lines.append("_None — no forced diagnostic fired this session._")
    else:
        for i, d in enumerate(payload.diagnostic_triggers, 1):
            lines.append(f"{i}. **Tool:** `{d.tool_name}`  ·  **exit_code:** `{d.exit_code}`")
            lines.append(f"   - **error_fingerprint:** `{d.error_fingerprint}`")
            if d.output_preview.strip():
                lines.append("   - **output preview:**")
                lines.append(f"     ```\n{_short_block(d.output_preview, 600)}\n     ```")
            lines.append("")

    lines.extend(
        [
            "## Recovery path (after diagnostic)",
            "",
            "What changed in behavior following a diagnostic observation.",
            "",
        ]
    )
    if not payload.recovery_paths:
        lines.append(
            "_No recovery path recorded (either no diagnostic, or no successful tool call after diagnostic)._"
        )
    else:
        for i, r in enumerate(payload.recovery_paths, 1):
            lines.append(f"{i}. {r.summary}")
            lines.append(f"   - **Linked fingerprint:** `{r.after_error_fingerprint}`")
            if r.next_tool:
                lines.append(f"   - **Next tool:** `{r.next_tool}`")
            if r.next_action_preview:
                lines.append(f"   - **Preview:** `{_one_line(r.next_action_preview)}`")
            lines.append("")

    lines.extend(
        [
            "## Prompt-tuning notes",
            "",
            "- If the same **error_fingerprint** appears often, tighten the base prompt for that tool stack.",
            "- If **recovery paths** cluster on `ls`/`find`, add explicit “verify layout before pytest/npm” guidance.",
            "- If **diagnostic triggers** are rare but failures are high, lower the streak threshold or improve first-attempt hints.",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _short_block(s: str, max_len: int) -> str:
    t = s.strip().replace("\r\n", "\n")
    if len(t) <= max_len:
        return t
    return t[: max_len - 20] + "\n…(truncated)…"


def _one_line(s: str) -> str:
    return " ".join(s.split())[:300]
