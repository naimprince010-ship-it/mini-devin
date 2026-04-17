"""Append-only learned insights for system-prompt injection (``knowledge_base/learned_patterns.md``)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_MAX_CHARS = 8000


def _patterns_path(workspace_root: str | Path) -> Path:
    root = Path(workspace_root).resolve()
    kb = root / "knowledge_base"
    kb.mkdir(parents=True, exist_ok=True)
    return kb / "learned_patterns.md"


def load_learned_patterns_for_prompt(workspace_root: str | Path, *, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Markdown block to append to the system prompt (empty if missing)."""
    path = _patterns_path(workspace_root)
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[-max_chars:] + "\n\n…(truncated from head)…"
    return "## Learned patterns (prior self-heal / session reflections)\n\n" + text


def append_learned_pattern(workspace_root: str | Path, markdown_body: str) -> Path:
    """Append one reflection section to ``knowledge_base/learned_patterns.md``."""
    path = _patterns_path(workspace_root)
    ts = datetime.now(timezone.utc).isoformat()
    block = f"\n\n## Reflection {ts}\n\n{markdown_body.strip()}\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
    else:
        existing = (
            "# Learned patterns\n\n"
            "Auto-generated insights from self-heal recovery and session reflections. "
            "Do not delete manually unless you intend to reset memory.\n"
        )
    path.write_text(existing + block, encoding="utf-8")
    return path
