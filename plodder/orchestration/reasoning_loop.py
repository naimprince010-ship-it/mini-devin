"""
OpenHands-style **Observe → Think → Act** contract for the unified JSON driver.

Structured phases are optional extra keys on the assistant JSON; the driver still requires
``status``, ``rationale``, and ``tool_calls`` for compatibility.
"""

from __future__ import annotations

REASONING_LOOP_SEED_SUFFIX = """
## Actor–critic loop (Eyes / Brain / Hands–Feet)

Each turn, mentally follow this loop (you may echo it in JSON for traceability):

1. **Observe (Eyes)** — What do tool outputs, **lsp_check**, or **playwright_observe** show? Do not claim success if diagnostics or UI still show errors.
2. **Think (Brain)** — Use episode memory + plan; avoid repeating failed commands.
3. **Act (Hands / Feet)** — Prefer **atomic_edit** for precise file changes; use **sandbox_shell** for commands (cwd + exports persist across shell calls in this session).

Optional JSON keys (same object as ``status`` / ``rationale`` / ``tool_calls``):
- ``observe``: one sentence on what you last saw.
- ``think``: one sentence on the next move.
- ``act_summary``: one phrase naming the tools you will call.

Before ``status: "done"``, run **lsp_check** on files you edited (and **playwright_observe** when the task is UI/web).
"""


def extract_shell_inner(argv: list[str]) -> str | None:
    """Return the inner script for ``sh -c`` / ``bash -lc``, else ``None``."""
    if len(argv) >= 3 and argv[0] in ("sh", "bash") and argv[1] in ("-c", "-lc"):
        return str(argv[2])
    return None
