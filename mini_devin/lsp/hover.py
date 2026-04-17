"""
Lightweight hover text from Tree-sitter symbol extraction (no Jedi / full LSP).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..memory.tree_sitter_symbols import extract_symbols_ast


def collect_hover(
    workspace_abs: str,
    rel_path: str,
    line: int,
    _column: int,
    content: str | None = None,
) -> str | None:
    """
    Return markdown/plain hover for 1-based ``line`` (Monaco), or None.
    """
    base = os.path.abspath(workspace_abs)
    target = os.path.abspath(os.path.join(base, rel_path))
    if not target.startswith(base):
        return None

    ext = Path(rel_path).suffix.lower()
    lang_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
    }
    lang = lang_map.get(ext)
    if not lang:
        return None

    if content is None:
        if not os.path.isfile(target):
            return None
        content = Path(target).read_text(encoding="utf-8", errors="replace")

    syms = extract_symbols_ast(rel_path, content, lang)
    for s in syms:
        if s.start_line <= line <= s.end_line:
            qn = f"{s.parent}.{s.name}" if s.parent else s.name
            parts = [f"**[{s.kind}]** `{qn}`", "", f"```{lang}", s.signature, "```"]
            if s.body_preview.strip():
                prev = s.body_preview.strip()[:800]
                if len(s.body_preview.strip()) > 800:
                    prev += "\n…"
                parts.extend(["", prev])
            return "\n".join(parts)
    return None
