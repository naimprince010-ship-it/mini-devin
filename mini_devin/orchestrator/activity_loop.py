"""
OpenHands-style **Activity** layer: unified Action → Observation steps, state, validation metadata.

This module is UI/log oriented (no LLM). The :class:`Agent` wires each tool execution through
here for ``meta.activity`` on :class:`~mini_devin.orchestrator.standard_events.AgentStreamEvent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from mini_devin.reliability.self_correction import terminal_sanity_check

ActivitySource = Literal["agent", "user", "system"]

# OpenHands-style action labels for the event stream / UI
ACTION_CMD_RUN = "CmdRunAction"
ACTION_FILE_READ = "FileReadAction"
ACTION_FILE_WRITE = "FileWriteAction"
ACTION_FILE_EDIT = "FileEditAction"
ACTION_TOOL_GENERIC = "ToolAction"


def classify_action_type(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "terminal":
        return ACTION_CMD_RUN
    if tool_name != "editor":
        return ACTION_TOOL_GENERIC
    action = str(arguments.get("action", "read_file") or "read_file").lower()
    if action == "read_file":
        return ACTION_FILE_READ
    if action in ("list_directory", "get_diagnostics"):
        return "DirectoryListAction"
    if action == "write_file":
        return ACTION_FILE_WRITE
    if action in ("str_replace", "apply_patch"):
        return ACTION_FILE_EDIT
    return ACTION_FILE_EDIT


def build_activity_meta(
    *,
    thought: str,
    source: ActivitySource,
    action_type: str,
    step: int,
    tool_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standardized ``meta.activity`` payload for TOOL_CALL / OBSERVATION rows."""
    block: dict[str, Any] = {
        "schema": "mini_devin.activity/1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "thought": (thought or "")[:12000],
        "action_type": action_type,
        "step": step,
        "tool_name": tool_name,
    }
    if extra:
        block["extra"] = extra
    return {"activity": block}


@dataclass
class AgentActivityState:
    """Mutable per-task activity state (synced after each tool step)."""

    step_index: int = 0
    last_tool_name: str | None = None
    last_terminal_exit_code: int | None = None
    editor_focus_path: str | None = None
    last_action_type: str | None = None

    def bump_step(self) -> int:
        self.step_index += 1
        return self.step_index

    def record_tool(self, tool_name: str, action_type: str) -> None:
        self.last_tool_name = tool_name
        self.last_action_type = action_type

    def record_terminal_outcome(self, exit_code: int | None) -> None:
        self.last_terminal_exit_code = exit_code

    def record_editor_path(self, path: str | None) -> None:
        if path:
            self.editor_focus_path = path

    def to_meta_snapshot(self) -> dict[str, Any]:
        return {
            "last_terminal_exit_code": self.last_terminal_exit_code,
            "last_tool_name": self.last_tool_name,
            "editor_focus_path": self.editor_focus_path,
            "last_action_type": self.last_action_type,
            "step_index": self.step_index,
        }


def validate_action_pre_flight(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    is_windows: bool,
    command_safety_check: Callable[[str], Any] | None,
) -> tuple[bool, str, Any | None]:
    """
    Run **before** executing a tool: terminal sanity + optional safety policy.

    Returns ``(ok, user_message, violation_or_none)``. ``violation`` may have ``.blocked`` and ``.message``.
    """
    if tool_name != "terminal":
        return True, "", None
    command = str(arguments.get("command", ""))
    ok, msg = terminal_sanity_check(command, is_windows=is_windows)
    if not ok:
        return False, msg, None
    if command_safety_check is None:
        return True, "", None
    violation = command_safety_check(command)
    if violation and getattr(violation, "blocked", False):
        return False, getattr(violation, "message", str(violation)) or "blocked", violation
    return True, "", None
