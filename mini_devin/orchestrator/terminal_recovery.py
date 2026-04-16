"""
Heuristic recovery hints for failed terminal commands (OpenHands-style observation enrichment).
"""

from __future__ import annotations

import re
from typing import Any


def terminal_recovery_hint(
    exit_code: int | None,
    output: str,
    *,
    command: str | None = None,
) -> str | None:
    """
    Return a short markdown hint when ``exit_code`` is non-zero, or None if nothing to add.
    """
    if exit_code in (None, 0):
        return None
    text = (output or "").lower()
    cmd = (command or "").lower()

    if "not found" in text or "command not found" in text or "is not recognized" in text:
        return (
            "**Hint**: The shell could not find a binary. "
            "Confirm it is installed, on PATH, or use the full path / project-local runner "
            "(e.g. `npx`, `python -m`, `./venv/bin/python`)."
        )
    if exit_code == 127 or "enoent" in text:
        return "**Hint**: Exit 127 often means missing executable — check PATH and spelling."

    if "permission denied" in text or "eacces" in text:
        return "**Hint**: Permission denied — try a user-writable directory or adjust file modes (`chmod` / ACL)."

    if "npm err" in text or "npm error" in text:
        return "**Hint**: npm failure — try `npm ci` vs `npm install`, delete `node_modules` + lockfile mismatch, or check Node version."

    if (
        "eaddrinuse" in text
        or "address already in use" in text
        or "port 3000 is in use" in text
        or "port is already in use" in text
    ):
        return (
            "**Hint**: Dev server port conflict — retry on another port like `3001`, `3002`, `4173`, `5001`, `5002`, `5173`, or `8000` "
            "instead of looping on unavailable port-inspection tools."
        )

    if (
        "lsof" in text and "not found" in text
        or "netstat" in text and "not found" in text
        or "ss: command not found" in text
        or "fuser" in text and "not found" in text
    ):
        return (
            "**Hint**: Port diagnostic tools are missing in this environment — choose another dev-server port "
            "or check app startup another way instead of retrying the same shell command."
        )

    if "no such file or directory" in text and ("cd " in cmd or "cwd" in text):
        return (
            "**Hint**: Working directory mismatch — run `pwd` and `ls` first, then retry from the directory that "
            "actually contains the app or script."
        )

    if "err_connection" in text or "could not resolve host" in text:
        return "**Hint**: Network/DNS failure — retry, check proxy/VPN, or use offline mode if available."

    if "pytest" in cmd or "pytest" in text[:800]:
        if "fixture" in text and exit_code != 0:
            return "**Hint**: pytest fixture error — verify `conftest.py` scope and fixture names."
        if "assert" in text:
            return "**Hint**: Assertion failed — inspect expected vs actual in the traceback above."

    if "ruff" in cmd or "ruff" in text[:400]:
        return "**Hint**: Ruff reported issues — run `ruff check --fix` where safe, then re-run check."

    if "mypy" in cmd or "error:" in text[:200] and "mypy" in text[:800]:
        return "**Hint**: Type errors — narrow annotations or add guards; re-run mypy on the edited file only."

    if re.search(r"\berror c\d{4}", text) or "compilation terminated" in text:
        return "**Hint**: Compiler error — scroll to the first error line; fix includes/types before later errors."

    if "fatal:" in text and "git" in cmd:
        return "**Hint**: Git error — check branch, remote, auth, and whether the repo is in a merge/rebase state."

    if exit_code == 1 and len((output or "").strip()) < 80:
        return "**Hint**: Non-zero exit with little output — re-run with `set -x` / verbose flags or capture stderr separately."

    return f"**Hint**: Command exited with code **{exit_code}**. Read stderr/stdout above, fix the root cause, then retry a smaller command."
