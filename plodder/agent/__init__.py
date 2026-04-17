"""Autonomous agent loop — lazy exports avoid import cycles with ``session_driver``."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AgentLoopConfig",
    "AgentSessionDriver",
    "ToolCall",
    "PlodderWorklog",
    "SessionRecoveryTracker",
    "export_worklog_json",
    "build_workspace_tree_block",
    "format_post_mortem",
    "normalize_tool_call",
    "run_agent",
]

_ORCHESTRATOR_NAMES = frozenset(
    {
        "PlodderWorklog",
        "SessionRecoveryTracker",
        "export_worklog_json",
        "format_post_mortem",
    }
)
_MAIN_LOOP_NAMES = frozenset(
    {
        "AgentLoopConfig",
        "AgentSessionDriver",
        "ToolCall",
        "build_workspace_tree_block",
        "normalize_tool_call",
        "run_agent",
    }
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _ORCHESTRATOR_NAMES:
        mod = importlib.import_module("plodder.agent.orchestrator")
        return getattr(mod, name)
    if name in _MAIN_LOOP_NAMES:
        mod = importlib.import_module("plodder.agent.main_loop")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
