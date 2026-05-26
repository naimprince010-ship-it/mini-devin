#!/usr/bin/env python3
"""
Bootstrap Plodder agent-memory training for a workspace.

What this does:
1) Creates workspace memory folders (.plodder, knowledge_base)
2) Initializes learned_patterns.md (if missing)
3) Appends a seed reflection block (so future prompts get memory guidance)
4) Writes a simple memory_training_plan.json for tracking progress

Usage:
  python scripts/start_agent_memory_training.py
  python scripts/start_agent_memory_training.py --workspace ./workspace
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SEED_BULLETS = [
    "Before repeating a failed command, inspect the exact error and change one variable.",
    "Prefer minimal, verifiable edits; run tests after each meaningful change.",
    "If dependency/tool is missing, fix environment first, then resume task logic.",
    "Record failure->fix pairs as short rules to prevent repeated mistakes.",
    "For long tasks, checkpoint progress every 2-3 actions.",
]


def ensure_learned_patterns(path: Path) -> bool:
    """Create knowledge_base/learned_patterns.md if it does not exist."""
    if path.exists():
        return False

    header = (
        "# Learned patterns\n\n"
        "Auto-generated insights from self-heal recovery and session reflections. "
        "Do not delete manually unless you intend to reset memory.\n"
    )
    path.write_text(header, encoding="utf-8")
    return True


def append_seed_reflection(path: Path) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    body = "\n".join(f"- {line}" for line in SEED_BULLETS)
    block = f"\n\n## Reflection {ts}\n\n{body}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


def write_plan(path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    plan = {
        "started_at": now,
        "status": "active",
        "phase": "bootstrap",
        "targets": {
            "sessions": 5,
            "documented_failure_fix_pairs": 20,
            "repeat_error_reduction_percent": 30,
        },
        "weekly_loop": [
            "Run real tasks and keep episode memory enabled.",
            "After each fixed failure, append one short reflection rule.",
            "Review learned patterns and remove noisy/duplicate rules.",
            "Track repeated error signatures and compare trend week over week.",
        ],
    }
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap agent memory training")
    p.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Workspace root (default: current directory)",
    )
    args = p.parse_args()

    ws = args.workspace.resolve()
    if not ws.exists() or not ws.is_dir():
        raise SystemExit(f"Workspace not found or not a directory: {ws}")

    plodder_dir = ws / ".plodder"
    kb_dir = ws / "knowledge_base"
    plodder_dir.mkdir(parents=True, exist_ok=True)
    kb_dir.mkdir(parents=True, exist_ok=True)

    patterns_path = kb_dir / "learned_patterns.md"
    created = ensure_learned_patterns(patterns_path)
    append_seed_reflection(patterns_path)

    plan_path = plodder_dir / "memory_training_plan.json"
    write_plan(plan_path)

    print("Agent memory training started.")
    print(f"workspace: {ws}")
    print(f"learned patterns: {patterns_path} ({'created' if created else 'updated'})")
    print(f"training plan: {plan_path}")
    print("next: run real tasks; keep adding concise failure->fix reflections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
