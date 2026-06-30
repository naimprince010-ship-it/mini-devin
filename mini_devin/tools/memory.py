"""
Long-Term Memory Tool for Plodder

Persists key-value memories across sessions in ~/.plodder/global_memory.json.
The agent uses this to remember user preferences, coding styles, and
architectural decisions that should survive between tasks and restarts.
"""

import asyncio
import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from ..core.tool_interface import BaseTool
from ..schemas.tools import BaseToolInput, BaseToolOutput, ToolStatus

# ── Module-level async lock prevents concurrent file corruption ───────────────
_MEMORY_LOCK: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _MEMORY_LOCK
    if _MEMORY_LOCK is None:
        _MEMORY_LOCK = asyncio.Lock()
    return _MEMORY_LOCK


# ── Storage path (overridable via env) ────────────────────────────────────────
def _memory_path() -> Path:
    raw = os.environ.get("PLODDER_MEMORY_FILE", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".plodder" / "global_memory.json"


# ── Key validation ─────────────────────────────────────────────────────────────
_KEY_RE = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")
_MAX_VALUE_CHARS = int(os.environ.get("PLODDER_MEMORY_MAX_VALUE_CHARS", "4000"))
_MAX_ENTRIES = int(os.environ.get("PLODDER_MEMORY_MAX_ENTRIES", "500"))


def _validate_key(key: str) -> str | None:
    """Return an error string if the key is invalid, else None."""
    if not key or not key.strip():
        return "key must not be empty"
    if not _KEY_RE.match(key.strip()):
        return (
            "key may only contain letters, digits, hyphens, underscores, and dots "
            "(max 128 chars)"
        )
    return None


# ── Low-level file helpers (run in thread to avoid blocking event loop) ────────
def _load_store(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            return {}
        return raw
    except (json.JSONDecodeError, OSError):
        return {}


def _save_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic rename


# ── Schema ─────────────────────────────────────────────────────────────────────

class MemoryAction(str, Enum):
    SAVE = "save"
    READ = "read"
    DELETE = "delete"


class MemoryInput(BaseToolInput):
    """Input for the long-term memory tool."""

    action: MemoryAction = Field(
        description=(
            "save — persist a key/value pair; "
            "read — return all stored memories; "
            "delete — remove a key"
        )
    )
    key: Optional[str] = Field(
        default=None,
        description="Memory key (required for save and delete). "
                    "Use descriptive names like 'preferred_language' or 'code_style'.",
    )
    value: Optional[str] = Field(
        default=None,
        description="Value to store (required for save). Plain text or JSON string.",
    )


class MemoryOutput(BaseToolOutput):
    """Output from the memory tool."""

    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Human-readable result summary")
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="For read action: dict of {key: {value, saved_at, updated_at}}",
    )


# ── Tool implementation ────────────────────────────────────────────────────────

class MemoryTool(BaseTool[MemoryInput, MemoryOutput]):
    """Persistent long-term memory that survives process restarts and new sessions."""

    name = "memory"
    description = (
        "Persistent long-term memory across ALL sessions and restarts. "
        "Store user preferences, coding conventions, and architectural decisions permanently.\n"
        "Actions:\n"
        "- save (key, value): Persist a named memory entry.\n"
        "- read (): Return every stored memory entry as a dict.\n"
        "- delete (key): Remove a stored entry."
    )
    input_schema = MemoryInput
    output_schema = MemoryOutput

    async def _execute(self, input_data: MemoryInput) -> MemoryOutput:
        loop = asyncio.get_event_loop()
        path = _memory_path()

        async with _get_lock():
            # ── READ ──────────────────────────────────────────────────────────
            if input_data.action == MemoryAction.READ:
                store: dict[str, Any] = await loop.run_in_executor(
                    None, _load_store, path
                )
                if not store:
                    return MemoryOutput(
                        status=ToolStatus.SUCCESS,
                        execution_time_ms=0,
                        success=True,
                        message="No memories stored yet.",
                        data={},
                    )
                return MemoryOutput(
                    status=ToolStatus.SUCCESS,
                    execution_time_ms=0,
                    success=True,
                    message=f"Loaded {len(store)} memory entry/entries.",
                    data=store,
                )

            # ── SAVE ──────────────────────────────────────────────────────────
            if input_data.action == MemoryAction.SAVE:
                key = (input_data.key or "").strip()
                err = _validate_key(key)
                if err:
                    return MemoryOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message=f"Invalid key: {err}",
                    )
                value = input_data.value
                if value is None:
                    return MemoryOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message="value is required for save.",
                    )
                if len(value) > _MAX_VALUE_CHARS:
                    return MemoryOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message=f"value too long ({len(value)} chars); limit is {_MAX_VALUE_CHARS}.",
                    )

                store = await loop.run_in_executor(None, _load_store, path)
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                is_new = key not in store
                entry: dict[str, Any] = {
                    "value": value,
                    "saved_at": store[key].get("saved_at", now_iso) if not is_new else now_iso,
                    "updated_at": now_iso,
                }
                if len(store) >= _MAX_ENTRIES and is_new:
                    return MemoryOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message=(
                            f"Memory store is full ({_MAX_ENTRIES} entries). "
                            "Delete some entries first."
                        ),
                    )
                store[key] = entry
                await loop.run_in_executor(None, _save_store, path, store)
                verb = "Created" if is_new else "Updated"
                return MemoryOutput(
                    status=ToolStatus.SUCCESS,
                    execution_time_ms=0,
                    success=True,
                    message=f"{verb} memory '{key}'.",
                )

            # ── DELETE ────────────────────────────────────────────────────────
            if input_data.action == MemoryAction.DELETE:
                key = (input_data.key or "").strip()
                err = _validate_key(key)
                if err:
                    return MemoryOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message=f"Invalid key: {err}",
                    )
                store = await loop.run_in_executor(None, _load_store, path)
                if key not in store:
                    return MemoryOutput(
                        status=ToolStatus.FAILURE,
                        execution_time_ms=0,
                        success=False,
                        message=f"Key '{key}' not found in memory.",
                    )
                del store[key]
                await loop.run_in_executor(None, _save_store, path, store)
                return MemoryOutput(
                    status=ToolStatus.SUCCESS,
                    execution_time_ms=0,
                    success=True,
                    message=f"Deleted memory '{key}'.",
                )

        # Unreachable but satisfies type checker
        return MemoryOutput(
            status=ToolStatus.FAILURE,
            execution_time_ms=0,
            success=False,
            message=f"Unknown action: {input_data.action}",
        )


def create_memory_tool() -> MemoryTool:
    """Factory — create a MemoryTool instance."""
    return MemoryTool()
