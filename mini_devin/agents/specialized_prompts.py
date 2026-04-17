"""
Optional specialization suffixes for the main agent system prompt.

Select with env AGENT_SPECIALIZATION=clean_code|performance|security|default
"""

from __future__ import annotations

SPECIALIZATIONS: dict[str, str] = {
    "clean_code": """
## Specialization: Clean code
- Prefer small, composable functions; meaningful names; early returns over deep nesting.
- Remove dead code and duplicate logic; add types/docstrings only where they clarify contracts.
- Keep edits minimal; match existing file style and patterns.
""",
    "performance": """
## Specialization: Performance
- Measure before optimizing; note Big-O and hot paths when changing loops or I/O.
- Avoid unnecessary allocations, N+1 queries, and unbounded in-memory buffering.
- Prefer streaming/async where the codebase already uses them; do not invent micro-optimizations without evidence.
""",
    "security": """
## Specialization: Security
- Treat all external input as untrusted; validate and encode outputs appropriately.
- Never commit secrets; reject logging tokens/passwords; use parameterized queries for SQL.
- Follow least privilege for commands and file operations; flag risky dependency changes.
""",
}


def get_specialization_system_suffix(mode: str) -> str:
    key = (mode or "default").strip().lower()
    if key in ("", "default", "general"):
        return ""
    return SPECIALIZATIONS.get(key, "").strip()
