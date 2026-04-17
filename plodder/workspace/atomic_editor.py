"""
Atomic editor **Hands**: read → verify → write in one operation (str_replace or full write).
"""

from __future__ import annotations

from typing import Any, Literal

from plodder.workspace.session_workspace import SessionWorkspace

Mode = Literal["str_replace", "write_full"]


def atomic_edit(
    ws: SessionWorkspace,
    path: str,
    *,
    mode: Mode,
    old_string: str | None = None,
    new_string: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    """
    ``str_replace``: exactly one occurrence of ``old_string`` must exist unless ``occurrences`` > 1.

    ``write_full``: replace entire file with ``content``.
    """
    rel = path.strip().replace("\\", "/").lstrip("/")
    if mode == "write_full":
        if content is None:
            raise ValueError("write_full requires content")
        ws.write_file(rel, content)
        return {"ok": True, "mode": mode, "path": rel, "bytes": len(content.encode("utf-8"))}

    if mode == "str_replace":
        if old_string is None or new_string is None:
            raise ValueError("str_replace requires old_string and new_string")
        text = ws.read_file(rel)
        count = text.count(old_string)
        if count == 0:
            raise ValueError("old_string not found in file")
        if count > 1:
            raise ValueError(
                f"old_string is ambiguous ({count} occurrences); narrow old_string to be unique"
            )
        new_text = text.replace(old_string, new_string, 1)
        ws.write_file(rel, new_text)
        return {
            "ok": True,
            "mode": mode,
            "path": rel,
            "bytes": len(new_text.encode("utf-8")),
            "replaced_once": True,
        }

    raise ValueError(f"unknown mode: {mode}")
