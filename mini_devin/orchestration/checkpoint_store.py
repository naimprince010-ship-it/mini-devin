"""File-backed durable checkpoint store for the orchestration runtime."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from mini_devin.contracts.protocols import DurableCheckpoint, DurableCheckpointStore


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_checkpoint(row: dict[str, Any]) -> DurableCheckpoint:
    created_at_raw = row.get("created_at")
    if isinstance(created_at_raw, str) and created_at_raw:
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
    else:
        created_at = datetime.utcnow()
    return DurableCheckpoint(
        checkpoint_id=str(row.get("checkpoint_id", "")),
        scope_id=str(row.get("scope_id", "")),
        created_at=created_at,
        state=dict(row.get("state") or {}),
        metadata=dict(row.get("metadata") or {}),
    )


class JsonlCheckpointStore(DurableCheckpointStore):
    """Append-only checkpoint store under ``.plodder/checkpoints.jsonl``."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)

    def _path(self) -> Path:
        root = self.workspace.resolve()
        root.mkdir(parents=True, exist_ok=True)
        log_dir = root / ".plodder"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "checkpoints.jsonl"

    def save(self, checkpoint: DurableCheckpoint) -> None:
        path = self._path()
        row = {
            **asdict(checkpoint),
            "created_at": checkpoint.created_at.isoformat(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    def load(self, checkpoint_id: str) -> DurableCheckpoint | None:
        for checkpoint in reversed(self.list()):
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None

    def list(self, scope_id: str | None = None) -> list[DurableCheckpoint]:
        path = self._path()
        if not path.is_file():
            return []
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return []
        checkpoints = [_parse_checkpoint(row) for row in rows if isinstance(row, dict)]
        if scope_id is None:
            return checkpoints
        return [checkpoint for checkpoint in checkpoints if checkpoint.scope_id == scope_id]

    def delete(self, checkpoint_id: str) -> None:
        path = self._path()
        if not path.is_file():
            return
        checkpoints = [checkpoint for checkpoint in self.list() if checkpoint.checkpoint_id != checkpoint_id]
        if not checkpoints:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8") as handle:
            for checkpoint in checkpoints:
                row = {
                    **asdict(checkpoint),
                    "created_at": checkpoint.created_at.isoformat(),
                }
                handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def load_checkpoint_store(workspace: str | Path) -> JsonlCheckpointStore:
    return JsonlCheckpointStore(workspace)
