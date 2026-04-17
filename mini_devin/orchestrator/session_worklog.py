"""
Persistent session worklog (``.plodder/worklog.json``) for plan / steps / resume after IDE restart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SessionWorklog:
    """OpenHands-style session state persisted on disk."""

    version: int = 1
    session_id: str | None = None
    last_task_id: str | None = None
    current_plan: list[str] = field(default_factory=list)
    finished_steps: list[str] = field(default_factory=list)
    remaining_tasks: list[str] = field(default_factory=list)
    current_step_idx: int = 0
    updated_at: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> SessionWorklog:
        return cls(
            version=int(data.get("version", 1)),
            session_id=data.get("session_id"),
            last_task_id=data.get("last_task_id"),
            current_plan=list(data.get("current_plan") or []),
            finished_steps=list(data.get("finished_steps") or []),
            remaining_tasks=list(data.get("remaining_tasks") or []),
            current_step_idx=int(data.get("current_step_idx", 0)),
            updated_at=str(data.get("updated_at") or ""),
        )


def worklog_path(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / ".plodder" / "worklog.json"


def load_worklog(workspace: str | Path) -> SessionWorklog | None:
    path = worklog_path(workspace)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return SessionWorklog.from_json_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def save_worklog(workspace: str | Path, log: SessionWorklog) -> None:
    root = Path(workspace).resolve()
    d = root / ".plodder"
    d.mkdir(parents=True, exist_ok=True)
    log.updated_at = datetime.now(timezone.utc).isoformat()
    path = d / "worklog.json"
    path.write_text(
        json.dumps(log.to_json_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
