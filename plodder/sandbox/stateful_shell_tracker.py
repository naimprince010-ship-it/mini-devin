"""
Stateful shell session for Plodder **Feet**: track cwd + exports across ``sandbox_shell`` invocations.

Docker still runs one-shot containers, but each command is wrapped with ``cd`` + ``export`` derived
from this tracker (persisted under ``.plodder/shell/session_state.json`` on the workspace).
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any


_STATE_DIR = ".plodder/shell"
_STATE_FILE = "session_state.json"


def _state_path(root: Path) -> Path:
    d = (root / _STATE_DIR).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d / _STATE_FILE


class StatefulShellTracker:
    """
    cwd is **relative POSIX path** from workspace root (``"."`` = workspace root).
    ``exports`` are applied as ``export K=V`` before each user command (POSIX shell).
    """

    def __init__(self, workspace_root: Path) -> None:
        self.root = Path(workspace_root).resolve()
        self.cwd = "."
        self.exports: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        p = _state_path(self.root)
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data.get("cwd"), str):
            self.cwd = data["cwd"].strip().replace("\\", "/") or "."
        ex = data.get("exports")
        if isinstance(ex, dict):
            self.exports = {str(k): str(v) for k, v in ex.items() if isinstance(k, str)}

    def _save(self) -> None:
        p = _state_path(self.root)
        p.write_text(
            json.dumps({"cwd": self.cwd, "exports": self.exports}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _normalize_cwd_token(self, raw: str) -> str:
        t = raw.strip().strip('"').strip("'").replace("\\", "/")
        if t in (".", "", "/workspace", "/workspace/"):
            return "."
        if t.startswith("/workspace"):
            inner = t[len("/workspace") :].strip("/")
            return inner or "."
        root = self.root.resolve()
        base = root if self.cwd in (".", "") else (root / self.cwd)
        try:
            joined = (base / t).resolve()
            rel = joined.relative_to(root)
            return rel.as_posix() or "."
        except (ValueError, OSError):
            return self.cwd

    def ingest_user_command(self, inner_cmd: str) -> None:
        """Update cwd/exports from a user shell snippet (best-effort static parse)."""
        for m in re.finditer(
            r"(?:^|[;&|]|\s&&\s)\s*cd\s+([^\s;&|]+)",
            inner_cmd,
            flags=re.MULTILINE,
        ):
            self.cwd = self._normalize_cwd_token(m.group(1))
        for m in re.finditer(
            r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$",
            inner_cmd,
            flags=re.MULTILINE,
        ):
            val = m.group(2).strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            self.exports[m.group(1)] = val
        self._save()

    def wrap_argv(self, argv: list[str]) -> list[str]:
        """
        Wrap ``["sh","-c","..."]`` / ``["bash","-lc","..."]`` with persisted cwd + exports.

        Call :meth:`ingest_user_command` **after** the shell runs (with the same inner string)
        so ``cd`` in this turn applies to the **next** turn. Other argv shapes are unchanged.
        """
        if len(argv) >= 3 and argv[0] in ("sh", "bash") and argv[1] in ("-c", "-lc"):
            inner = argv[2]
            ws = "/workspace"
            cd_arg = self.cwd if self.cwd in (".", "") else self.cwd
            cd_path = f"{ws}/{cd_arg}" if cd_arg != "." else ws
            exports = ""
            if self.exports:
                exports = "export " + " ".join(f"{shlex.quote(k)}={shlex.quote(v)}" for k, v in self.exports.items()) + " && "
            wrapped = (
                f"set +e; umask 022; cd {shlex.quote(cd_path)} 2>/dev/null || cd {ws} && "
                f"{exports}"
                f"{inner}"
            )
            return [argv[0], argv[1], wrapped]
        if len(argv) >= 2 and argv[0] in ("sh", "bash"):
            joined = " ".join(shlex.quote(a) for a in argv[1:])
            return self.wrap_argv(["sh", "-c", joined])
        return list(argv)
