"""Post-recovery reflection LLM pass → ``learned_patterns.md``."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from plodder.memory.learned_patterns import append_learned_pattern

ChatMessage = dict[str, Any]
LLMFn = Callable[[list[ChatMessage]], Awaitable[str]]


async def run_self_heal_reflection(
    llm: LLMFn,
    *,
    goal: str,
    plan_brief: str,
    attempts: list[dict[str, Any]],
    final_code: str,
    workspace_root: str,
) -> str:
    """
    After a multi-attempt self-heal **success**, ask what went wrong and how to avoid it; persist markdown.
    """
    code_excerpt = (final_code or "")[:8000]
    att_json = json.dumps(attempts, indent=2, default=str, ensure_ascii=False)[:12000]
    user = (
        "You are reflecting on a self-heal loop that **eventually succeeded** after one or more "
        "sandbox failures.\n\n"
        f"## Goal\n{goal.strip()}\n\n"
        f"## Plan summary\n{plan_brief.strip()[:4000]}\n\n"
        f"## Sandbox attempts (chronological)\n```json\n{att_json}\n```\n\n"
        f"## Final code (excerpt)\n```text\n{code_excerpt}\n```\n\n"
        "Respond in **markdown** (max 10 short bullets):\n"
        "1. What likely caused the initial failure(s)?\n"
        "2. Concrete guardrails, tests, or patterns to avoid repeating this in future sessions.\n"
        "Do not paste the full program again — **insights only**."
    )
    messages: list[ChatMessage] = [
        {
            "role": "system",
            "content": "You write concise engineering retrospectives. No flattery.",
        },
        {"role": "user", "content": user},
    ]
    raw = (await llm(messages)).strip()
    if raw:
        append_learned_pattern(workspace_root, raw)
    return raw
