"""
Normalize agent-supplied paths when the host is POSIX (Linux containers).

Models often echo the user's Windows paths (e.g. ``G:\\myself``). Those are
invalid on Linux and cause mkdir/editor loops.
"""

from __future__ import annotations

import os
import re


_WIN_DRIVE = re.compile(r"^([A-Za-z]):[/\\](.*)$")


def looks_like_windows_drive_path(path: str) -> bool:
    s = path.strip()
    if len(s) < 2:
        return False
    return s[0].isalpha() and s[1] == ":"


def posix_tail_after_windows_drive(path: str) -> str:
    """
    ``G:\\myself\\a`` -> ``myself/a``; ``G:\\`` -> ``.``; ``G:`` -> ``.``
    """
    s = path.strip().replace("\\", "/")
    m = _WIN_DRIVE.match(s)
    if m:
        tail = (m.group(2) or "").strip("/")
        return tail or "."
    if looks_like_windows_drive_path(s) and len(s) >= 3 and s[2] in "/\\":
        tail = s[3:].replace("\\", "/").strip("/")
        return tail or "."
    if len(s) == 2 and s[1] == ":":
        return "."
    return s.lstrip("/")


def resolve_for_editor(path: str, working_directory: str) -> str:
    """
    Resolve editor paths: on POSIX, remap Windows drive paths into the workspace.
    """
    wd = working_directory or os.getcwd()
    raw = path.strip()
    if os.name != "nt" and looks_like_windows_drive_path(raw):
        tail = posix_tail_after_windows_drive(raw)
        return os.path.abspath(os.path.join(wd, tail))
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(wd, raw))


def command_uses_windows_drive_paths(command: str) -> bool:
    """
    Detect Windows-style paths inside a shell command on hosts where they fail.

    Uses ``X:\\`` (almost only Windows) and ``X:/`` but skips URL schemes ``*://``.
    """
    if re.search(r"[A-Za-z]:\\", command):
        return True
    for m in re.finditer(r"[A-Za-z]:/", command):
        after_colon = command[m.start() + 2 : m.start() + 4]
        if after_colon.startswith("//"):
            continue
        return True
    return False


def linux_workspace_hint(workspace: str) -> str:
    return (
        "This shell runs on Linux (container/cloud). Windows paths like G:\\\\ or C:\\\\ "
        "do not exist here.\n"
        f"Use the task workspace: {workspace}\n"
        "Examples: `mkdir -p ./myself` or create files with the `editor` write_file action "
        "(parent folders are created automatically)."
    )
