"""Append-only event history for reconstructing agent context."""

from __future__ import annotations

import asyncio
from typing import Any

from mini_devin.backbone.event_models import AgentEvent, is_action_event, is_observation_event


class AgentStateStore:
    """
    Single-writer-safe timeline of :class:`~mini_devin.backbone.event_models.AgentEvent`.

    Used by :class:`~mini_devin.backbone.event_stream.EventStream` and by controllers
    that need to rebuild prompts from the full trace.
    """

    def __init__(self, max_events: int | None = None) -> None:
        self._max_events = max_events
        self._events: list[AgentEvent] = []
        self._lock = asyncio.Lock()

    async def append(self, event: AgentEvent) -> None:
        async with self._lock:
            self._events.append(event)
            if self._max_events is not None and len(self._events) > self._max_events:
                overflow = len(self._events) - self._max_events
                del self._events[:overflow]

    async def snapshot(self) -> tuple[AgentEvent, ...]:
        async with self._lock:
            return tuple(self._events)

    async def count(self) -> int:
        async with self._lock:
            return len(self._events)

    async def reconstruct_openai_style_tail(self, tail: int = 200) -> list[dict[str, Any]]:
        """
        Best-effort mapping of the last ``tail`` events to OpenAI-style messages.

        This is a **migration helper**, not a full replacement for the LLM client's
        conversation format. Extend as you add event kinds.
        """
        snap = await self.snapshot()
        events = snap[-tail:] if tail > 0 else []
        messages: list[dict[str, Any]] = []
        for e in events:
            if e.type == "AgentMessageEvent":  # type: ignore[union-attr]
                role = e.role  # type: ignore[union-attr]
                content = e.content  # type: ignore[union-attr]
                api_role = "user" if role == "user" else "assistant"
                if role == "system":
                    api_role = "system"
                messages.append({"role": api_role, "content": content})
            elif is_action_event(e):
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": e.id,
                                "type": "function",
                                "function": {
                                    "name": _action_tool_name(e),
                                    "arguments": _action_arguments_json(e),
                                },
                            }
                        ],
                    }
                )
            elif is_observation_event(e):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(e, "source_action_id", e.id),
                        "content": format_observation_as_tool_result(e),
                    }
                )
        return messages


def _action_tool_name(e: AgentEvent) -> str:
    if e.type == "CmdRunAction":  # type: ignore[union-attr]
        return "terminal"
    if e.type == "FileWriteAction":  # type: ignore[union-attr]
        return "editor"
    if e.type == "GenericToolAction":  # type: ignore[union-attr]
        return e.tool_name  # type: ignore[union-attr]
    return "unknown"


def _action_arguments_json(e: AgentEvent) -> str:
    import json

    if e.type == "CmdRunAction":  # type: ignore[union-attr]
        payload = {
            "command": e.command,  # type: ignore[union-attr]
            "working_directory": e.cwd or ".",  # type: ignore[union-attr]
            "timeout_seconds": e.timeout_sec or 30,  # type: ignore[union-attr]
        }
    elif e.type == "FileWriteAction":  # type: ignore[union-attr]
        payload = {
            "action": "write_file",
            "path": e.path,  # type: ignore[union-attr]
            "content": e.content,  # type: ignore[union-attr]
        }
    elif e.type == "GenericToolAction":  # type: ignore[union-attr]
        payload = dict(e.arguments)  # type: ignore[union-attr]
    else:
        payload = {}
    return json.dumps(payload)


def format_observation_as_tool_result(e: AgentEvent) -> str:
    if e.type == "CommandOutputObservation":  # type: ignore[union-attr]
        parts = [e.stdout or ""]  # type: ignore[union-attr]
        if e.stderr:  # type: ignore[union-attr]
            parts.append("\n--- stderr ---\n" + e.stderr)  # type: ignore[union-attr]
        parts.append(f"\n(exit {e.exit_code})")  # type: ignore[union-attr]
        return "".join(parts)
    if e.type == "FileWriteObservation":  # type: ignore[union-attr]
        return f"wrote {e.bytes_written} bytes to {e.path}" + (  # type: ignore[union-attr]
            f" err={e.error}" if e.error else ""  # type: ignore[union-attr]
        )
    if e.type == "GenericToolObservation":  # type: ignore[union-attr]
        return e.content  # type: ignore[union-attr]
    if e.type == "ErrorObservation":  # type: ignore[union-attr]
        return f"[{e.error_type}] {e.message}"  # type: ignore[union-attr]
    return str(e)
