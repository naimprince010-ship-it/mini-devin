"""
Unified timeline view over ``session_events.jsonl`` — Thought / Action / Observation export.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from .session_events import load_session_events


class TimelineEventKind(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    MESSAGE = "message"
    OTHER = "other"


class TimelineEvent(BaseModel):
    """One normalized row for dashboards or JSON export."""

    kind: TimelineEventKind
    ts: str | None = None
    summary: str = ""
    tool_name: str | None = None
    exit_code: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


def _classify_row(row: dict[str, Any]) -> TimelineEvent:
    ts = row.get("ts")
    kind_s = str(row.get("kind") or "").lower()
    legacy = str(row.get("type") or row.get("legacy_type") or "").lower()

    if kind_s == "message" or legacy == "message":
        return TimelineEvent(
            kind=TimelineEventKind.MESSAGE,
            ts=str(ts) if ts else None,
            summary=(row.get("text") or "")[:8000],
            raw=dict(row),
        )
    if kind_s == "tool_call" or legacy == "act":
        tool = row.get("tool_name") or row.get("tool")
        args = row.get("tool_args") if isinstance(row.get("tool_args"), dict) else {}
        summary = f"{tool} {json.dumps(args, ensure_ascii=False)[:2000]}"
        return TimelineEvent(
            kind=TimelineEventKind.ACTION,
            ts=str(ts) if ts else None,
            summary=summary,
            tool_name=str(tool) if tool else None,
            raw=dict(row),
        )
    if kind_s == "observation" or legacy == "observe":
        tool = row.get("tool_name") or row.get("tool")
        out = row.get("output") or row.get("summary") or ""
        return TimelineEvent(
            kind=TimelineEventKind.OBSERVATION,
            ts=str(ts) if ts else None,
            summary=str(out)[:8000],
            tool_name=str(tool) if tool else None,
            exit_code=row.get("exit_code") if isinstance(row.get("exit_code"), int) else None,
            raw=dict(row),
        )
    if kind_s == "status" or legacy in ("think", "skill_autoload", "task_start"):
        return TimelineEvent(
            kind=TimelineEventKind.THOUGHT,
            ts=str(ts) if ts else None,
            summary=(row.get("text") or json.dumps(row.get("meta") or {}, ensure_ascii=False))[:8000],
            raw=dict(row),
        )
    return TimelineEvent(
        kind=TimelineEventKind.OTHER,
        ts=str(ts) if ts else None,
        summary=json.dumps(row, default=str, ensure_ascii=False)[:4000],
        raw=dict(row),
    )


class EventStream:
    """
    Read-only timeline helper over the append-only JSONL written by the agent.
    """

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()

    def iter_normalized(self, *, max_lines: int = 5000) -> Iterator[TimelineEvent]:
        for row in load_session_events(self.workspace, max_lines=max_lines):
            if not isinstance(row, dict):
                continue
            yield _classify_row(row)

    def to_export_list(self, *, max_lines: int = 5000) -> list[dict[str, Any]]:
        return [e.model_dump() for e in self.iter_normalized(max_lines=max_lines)]

    def export_json(self, path: str | Path, *, max_lines: int = 5000) -> None:
        data = {
            "workspace": str(self.workspace),
            "events": self.to_export_list(max_lines=max_lines),
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
