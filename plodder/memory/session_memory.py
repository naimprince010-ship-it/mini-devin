"""
OpenHands-style **episode memory**: append-only ``session_memory.jsonl`` + condensed continuity.

Stops blind repetition by surfacing recent actions/observations and duplicate-command warnings.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

MemoryKind = Literal["thought", "action", "observation", "meta"]

# Sliding window: beyond this many events, compress older rows into a short summary.
_CONDENSE_AFTER = 10
_KEEP_FULL_DETAIL = 5


def _memory_path(root: str | Path) -> Path:
    d = Path(root).resolve() / ".plodder"
    d.mkdir(parents=True, exist_ok=True)
    return d / "session_memory.jsonl"


class EpisodeMemory:
    """
    Real-time JSONL stream (thought / action / observation) under ``.plodder/session_memory.jsonl``.
    """

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).resolve()
        self._path = _memory_path(self.root)

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        kind: MemoryKind,
        payload: dict[str, Any],
        *,
        round_idx: int | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "round": round_idx,
            **payload,
        }
        line = json.dumps(row, default=str, ensure_ascii=False) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def read_events(self, *, max_lines: int = 500) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        tail = lines[-max_lines:]
        out: list[dict[str, Any]] = []
        for ln in tail:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    @staticmethod
    def _action_signature(ev: dict[str, Any]) -> str | None:
        if ev.get("kind") != "action":
            return None
        tool = str(ev.get("tool") or "")
        args = ev.get("args")
        if not isinstance(args, dict):
            args = {}
        try:
            blob = json.dumps({"t": tool, "a": args}, sort_keys=True, default=str)
        except TypeError:
            blob = f"{tool}:{args!r}"
        return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()[:16]

    def _duplicate_warnings(self, events: list[dict[str, Any]], window: int = 20) -> str:
        sigs = [self._action_signature(e) for e in events[-window:]]
        sigs = [s for s in sigs if s]
        if not sigs:
            return ""
        c = Counter(sigs)
        dups = [s for s, n in c.items() if n >= 2]
        if not dups:
            return ""
        return (
            "**Anti-loop**: The same tool+arguments signature was used **multiple times** in recent "
            f"steps (hashes: {', '.join(dups[:5])}). Try a **different** command or fix root cause "
            "before repeating.\n\n"
        )

    @staticmethod
    def _one_line(ev: dict[str, Any]) -> str:
        k = ev.get("kind", "?")
        if k == "thought":
            t = str(ev.get("text", ""))[:160]
            return f"- thought: {t}"
        if k == "action":
            return f"- action {ev.get('tool')}: {str(ev.get('args', {}))[:120]}"
        if k == "observation":
            ok = ev.get("ok", "?")
            return f"- observation ok={ok} tool={ev.get('tool')}"
        return f"- {k}: {str(ev)[:100]}"

    def get_condensed_context(
        self,
        *,
        condense_after: int = _CONDENSE_AFTER,
        keep_full: int = _KEEP_FULL_DETAIL,
        max_lines: int = 400,
    ) -> str:
        """
        Text block for the next LLM turn: ``Short History Summary`` + full detail for the tail.

        When total events > ``condense_after``, older rows are one-line summaries; the last
        ``keep_full`` events are included as JSON.
        """
        events = self.read_events(max_lines=max_lines)
        if not events:
            return ""
        warn = self._duplicate_warnings(events)
        if len(events) <= condense_after:
            body = "## Episode memory (full stream, condensed lines)\n" + "\n".join(
                self._one_line(e) for e in events
            )
            return warn + body

        older = events[:-keep_full]
        recent = events[-keep_full:]
        summary = (
            "## Short History Summary (older events)\n"
            + "\n".join(self._one_line(e) for e in older)
            + "\n\n## Recent events (full JSON, last "
            + str(len(recent))
            + ")\n```json\n"
            + json.dumps(recent, indent=2, default=str, ensure_ascii=False)[:12000]
            + "\n```"
        )
        return warn + summary
