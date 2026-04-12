"""
Load hand-curated golden examples (JSONL) for few-shot style injection.

Each line: JSON object with keys "title" (optional), "messages" (list of {role, content})
or flat "instruction"/"output" for a single turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_golden_path() -> Path:
    return _repo_root() / "data" / "golden" / "examples.jsonl"


def load_golden_records(path: Path, *, max_records: int = 12) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= max_records:
                break
    return out


def _record_to_markdown(rec: dict[str, Any]) -> str:
    if "messages" in rec and isinstance(rec["messages"], list):
        title = rec.get("title", "Example")
        parts = [f"#### {title}\n"]
        for m in rec["messages"]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user"))
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            parts.append(f"**{role}:**\n{content}\n")
        return "\n".join(parts).strip()
    ins = str(rec.get("instruction", "")).strip()
    out = str(rec.get("output", "")).strip()
    if ins and out:
        return f"**Task:** {ins}\n\n**Ideal response sketch:**\n{out}".strip()
    return ""


def format_golden_context_for_prompt(
    path: Optional[str | Path] = None,
    *,
    max_records: int = 8,
    max_chars: int = 6000,
) -> str:
    p = Path(path) if path else default_golden_path()
    if not p.is_file():
        return ""
    records = load_golden_records(p, max_records=max_records)
    if not records:
        return ""
    blocks = ["## Golden examples (curated patterns)\n"]
    used = 0
    for rec in records:
        block = _record_to_markdown(rec)
        if not block:
            continue
        block = block + "\n\n---\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    text = "\n".join(blocks).strip()
    return text if len(text) > 80 else ""
