"""
Artifact Logger for Mini-Devin

Records and persists task execution artifacts including tool calls,
file modifications, commands executed, token usage, and final diffs.
Each task run is stored in its own directory under a configurable base dir.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class ArtifactLogger:
    """
    Logs and persists artifacts produced during a task run.

    Artifacts are stored under ``{base_dir}/{task_id}/`` with the
    following structure::

        runs/
        └── <task_id>/
            ├── run.json          # Run metadata and summary
            ├── tool_calls.jsonl  # One JSON object per line, each tool call
            └── diff.patch        # Final git diff (if available)
    """

    def __init__(self, base_dir: str, task_id: str, task_description: str):
        self._run_dir = Path(base_dir) / task_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._task_id = task_id
        self._task_description = task_description
        self._model: str = ""
        self._started_at: str = datetime.now(timezone.utc).isoformat()
        self._completed_at: Optional[str] = None
        self._status: str = "running"
        self._summary: str = ""
        self._total_tokens: int = 0
        self._iterations: int = 0
        self._commands_executed: list[str] = []
        self._files_modified: list[str] = []
        self._diff: str = ""

        self._tool_calls_path = self._run_dir / "tool_calls.jsonl"
        self._run_meta_path = self._run_dir / "run.json"

        # Write initial metadata so the directory is populated immediately.
        self._write_meta()

    # ------------------------------------------------------------------
    # Public API used by the agent
    # ------------------------------------------------------------------

    def set_model(self, model: str) -> None:
        """Record which LLM model is being used for this run."""
        self._model = model
        self._write_meta()

    def log_tool_call(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        success: bool,
        duration_ms: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Append a tool call record to the JSONL log."""
        record = {
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result if isinstance(result, str) else str(result),
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            record["error"] = error

        with self._tool_calls_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def add_command_executed(self, command: str) -> None:
        """Record a shell command that was executed."""
        if command not in self._commands_executed:
            self._commands_executed.append(command)

    def add_file_modified(self, file_path: str) -> None:
        """Record a file that was created or modified."""
        if file_path not in self._files_modified:
            self._files_modified.append(file_path)

    def increment_iteration(self) -> None:
        """Increment the agent iteration counter."""
        self._iterations += 1

    def update_tokens(self, total_tokens: int) -> None:
        """Update cumulative token usage."""
        self._total_tokens = total_tokens

    def set_diff(self, diff_text: str) -> None:
        """Store the final git diff for this run."""
        self._diff = diff_text
        (self._run_dir / "diff.patch").write_text(diff_text, encoding="utf-8")

    def complete(self, status: str, summary: str) -> None:
        """
        Finalise the run, writing all metadata to disk.

        Args:
            status: One of ``"completed"``, ``"failed"``, or ``"blocked"``.
            summary: Human-readable summary of what was done.
        """
        self._status = status
        self._summary = summary
        self._completed_at = datetime.now(timezone.utc).isoformat()
        self._write_meta()

    def get_run_dir(self) -> str:
        """Return the directory where all artifacts for this run are stored."""
        return str(self._run_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_meta(self) -> None:
        """Persist run metadata to ``run.json``."""
        meta = {
            "task_id": self._task_id,
            "task_description": self._task_description,
            "model": self._model,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "status": self._status,
            "summary": self._summary,
            "total_tokens": self._total_tokens,
            "iterations": self._iterations,
            "commands_executed": self._commands_executed,
            "files_modified": self._files_modified,
        }
        self._run_meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def create_artifact_logger(
    base_dir: str,
    task_id: str,
    task_description: str,
) -> ArtifactLogger:
    """
    Create an :class:`ArtifactLogger` for a new task run.

    Args:
        base_dir: Root directory under which per-task run directories are
            created (e.g. ``"runs"``).
        task_id: Unique identifier for this task (used as the subdirectory
            name).
        task_description: Human-readable description of the task, stored in
            the run metadata.

    Returns:
        A configured :class:`ArtifactLogger` instance.
    """
    return ArtifactLogger(
        base_dir=base_dir,
        task_id=task_id,
        task_description=task_description,
    )
