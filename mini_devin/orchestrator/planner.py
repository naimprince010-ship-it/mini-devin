"""
Structured workspace planning (OpenHands-style).

Ensures ``PLAN.md`` exists at the workspace root so the agent can align actions
with explicit steps and keep the plan updated as work proceeds.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAN_FILENAME = "PLAN.md"


class Planner:
    """Create or extend ``PLAN.md`` with goal-linked steps."""

    @staticmethod
    def sync_plan_file(workspace: str | Path, goal: str) -> Path:
        """
        Write or append a structured plan for ``goal``.

        Returns the path to ``PLAN.md``.
        """
        root = Path(workspace).resolve()
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / PLAN_FILENAME
        stamp = datetime.now(timezone.utc).isoformat()
        goal_txt = (goal or "").strip()
        block = (
            f"# Execution plan\n\n"
            f"_Updated: {stamp}_\n\n"
            f"## Goal\n{goal_txt or '(no description)'}\n\n"
            f"## Steps\n"
            f"- [ ] **STEP-1**: Inspect repository layout and constraints.\n"
            f"- [ ] **STEP-2**: Implement changes (editor / terminal) tied to this plan.\n"
            f"- [ ] **STEP-3**: Verify (tests, linter, or minimal run); update this file when done.\n\n"
            f"> Each tool call should include **`plan_step`** (e.g. `\"STEP-2\"`) for traceability.\n"
        )

        if plan_path.is_file():
            existing = plan_path.read_text(encoding="utf-8", errors="replace")
            if goal_txt and goal_txt in existing[:6000]:
                return plan_path
            merged = (
                existing.rstrip()
                + f"\n\n---\n## Plan revision ({stamp})\n\n"
                + block
            )
            plan_path.write_text(merged, encoding="utf-8")
        else:
            plan_path.write_text(block, encoding="utf-8")
        return plan_path

    @staticmethod
    def append_checkpoint(
        workspace: str | Path,
        tool: str,
        exit_code: int | None,
        plan_step: Any = None,
    ) -> Path | None:
        """
        Append a short progress line to ``PLAN.md`` so the UI / git view stays fresh
        after major tool actions (terminal, editor).
        """
        root = Path(workspace).resolve()
        plan_path = root / PLAN_FILENAME
        if not plan_path.is_file():
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        ps = str(plan_step).strip() if plan_step is not None and str(plan_step).strip() else "—"
        line = f"\n- [{stamp}] **{tool}** (exit={exit_code}, plan_step={ps})"
        try:
            with plan_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            return None
        return plan_path
