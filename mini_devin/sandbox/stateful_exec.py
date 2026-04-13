"""
OpenHands-style **stateful shell** without a long-lived PTY: each invocation runs
``bash -lc`` but **cwd** (and light **env** snippets) persist via files under
``.mini_devin/_shell/`` on the workspace volume.

- ``cwd.txt`` — last ``pwd`` after each command (clamped under workspace root on the host).
- ``env.sh`` — optional ``export`` lines appended when the user command looks like ``export …``.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path


def noninteractive_export_block() -> str:
    """When ``MINIDEVIN_NONINTERACTIVE`` is set, export CI-style vars (apt/git prompts)."""
    v = (os.environ.get("MINIDEVIN_NONINTERACTIVE") or "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return ""
    return (
        "export DEBIAN_FRONTEND=noninteractive\n"
        "export CI=true\n"
        "export GIT_TERMINAL_PROMPT=0\n"
    )


_EXPORT_LINE_RE = re.compile(
    r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$",
    re.MULTILINE,
)


def shell_state_dir(workspace_root: str) -> Path:
    d = Path(workspace_root) / ".mini_devin" / "_shell"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_stateful_bash_script(
    user_command: str,
    *,
    workspace_posix: str,
    cwd_file_posix: str,
    env_file_posix: str,
    clamp_under_workspace: bool,
) -> str:
    """
    Bash script body: ``cd`` from persisted cwd, run user command, persist new cwd.

    ``workspace_posix`` must use ``/`` (container ``/workspace`` or host path for bash).
    """
    ws = shlex.quote(workspace_posix)
    cf = shlex.quote(cwd_file_posix)
    ef = shlex.quote(env_file_posix)
    if clamp_under_workspace:
        clamp_block = f"""
_new="$(cat {cf} 2>/dev/null || true)"
case "$_new" in
  {ws}|{ws}/*) ;;
  *) printf '%s' {ws} > {cf} ;;
esac
"""
    else:
        clamp_block = ""

    return f"""set +e
umask 022
mkdir -p "$(dirname {cf})"
[ -f {cf} ] || printf '%s' {ws} > {cf}
MDV_CWD="$(cat {cf})"
cd "$MDV_CWD" 2>/dev/null || cd {ws}
[ -f {ef} ] && . {ef}
{user_command}
_ec=$?
printf '%s' "$PWD" > {cf}.new && mv -f {cf}.new {cf} 2>/dev/null || printf '%s' {ws} > {cf}
{clamp_block}
exit $_ec
"""


def maybe_append_exports(workspace_root: str, user_command: str) -> None:
    """Append simple ``export VAR=value`` lines from *user_command* to ``env.sh``."""
    d = shell_state_dir(workspace_root)
    env_path = d / "env.sh"
    lines: list[str] = []
    for m in _EXPORT_LINE_RE.finditer(user_command.strip()):
        var, val = m.group(1), m.group(2).strip()
        if not var:
            continue
        # Strip matching quotes from value
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        lines.append(f"export {var}={shlex.quote(val)}\n")
    if not lines:
        return
    with env_path.open("a", encoding="utf-8") as fh:
        fh.writelines(lines)


def posix_paths_for_process_sandbox(project_root: str) -> tuple[str, str, str]:
    """Host paths as POSIX for bash (Git Bash on Windows uses forward slashes)."""
    root = Path(project_root).resolve().as_posix()
    d = shell_state_dir(str(Path(project_root).resolve()))
    return root, (d / "cwd.txt").as_posix(), (d / "env.sh").as_posix()


def posix_paths_for_docker_workspace() -> tuple[str, str, str]:
    """In-container paths under ``/workspace``."""
    return "/workspace", "/workspace/.mini_devin/_shell/cwd.txt", "/workspace/.mini_devin/_shell/env.sh"
