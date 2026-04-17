"""
Standard agent stream events (Action/Observation-style) for JSONL + SSE/UI.

This is a **contract** layer on top of the existing append-only JSONL log
(``session_events.py``). Payloads stay JSON-serializable.

Legacy rows used ``type`` in ``think`` / ``observe`` / ``auto_verify`` / ….
New rows add a stable ``kind`` (see :class:`AgentEventKind`) while optionally
keeping ``legacy_type`` for backward compatibility during migration.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentEventKind(str, Enum):
    """Unified event kinds for logging and UI (SSE/WebSocket)."""

    MESSAGE = "message"
    """User or agent natural-language message."""

    TOOL_CALL = "tool_call"
    """Request to run a tool (name + arguments)."""

    OBSERVATION = "observation"
    """Result of a tool: stdout, structured output, or error."""

    STATUS = "status"
    """Progress / plan / “thinking” line without a tool invocation."""


class AgentStreamEvent(BaseModel):
    """
    One logical event in the session timeline.

    Serialize with :meth:`to_log_dict` / :meth:`sse_payload_dict`; persist via :func:`append_standard_event`.
    """

    kind: AgentEventKind

    role: Literal["user", "agent", "system"] | None = None
    """For MESSAGE; optional for STATUS."""

    text: str | None = None
    """MESSAGE body or STATUS line."""

    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_args: dict[str, Any] | None = None

    output: str | None = None
    """OBSERVATION: primary text (stdout or summary)."""

    error: str | None = None
    """OBSERVATION: error message if failed."""

    exit_code: int | None = None

    meta: dict[str, Any] = Field(default_factory=dict)
    """Extra fields (plan_step, filesystem_delta, model, tokens, …)."""

    legacy_type: str | None = None
    """Original ``type`` string when bridging from old events."""

    model_config = {"extra": "forbid"}

    def to_log_dict(self) -> dict[str, Any]:
        """Flat dict for JSONL (timestamp added by ``append_session_event``)."""
        d: dict[str, Any] = {"kind": self.kind.value}
        if self.legacy_type:
            d["type"] = self.legacy_type
        if self.role is not None:
            d["role"] = self.role
        if self.text is not None:
            d["text"] = self.text
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
            d["tool"] = self.tool_name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_args is not None:
            d["tool_args"] = self.tool_args
            if self.kind == AgentEventKind.TOOL_CALL:
                reserved = frozenset(d) | {"meta", "output", "error", "text", "role"}
                for ak, av in self.tool_args.items():
                    if ak not in reserved:
                        d[ak] = av
        if self.output is not None:
            d["output"] = self.output
        if self.error is not None:
            d["error"] = self.error
        if self.exit_code is not None:
            d["exit_code"] = self.exit_code
        if self.meta:
            d["meta"] = dict(self.meta)
            if self.legacy_type == "observe":
                m = self.meta
                if "plan_step" in m:
                    d["plan_step"] = m["plan_step"]
                if "filesystem_delta" in m:
                    d["filesystem_delta"] = m["filesystem_delta"]
            elif self.legacy_type == "think" and "task_id" in self.meta:
                d["task_id"] = self.meta["task_id"]
            elif self.legacy_type == "task_start":
                for _k in ("task_id", "goal"):
                    if _k in self.meta:
                        d[_k] = self.meta[_k]
            elif self.legacy_type == "auto_verify" and "path" in self.meta:
                d["path"] = self.meta["path"]
        return d

    def sse_payload_dict(self) -> dict[str, Any]:
        """Same as log row; API layer may JSON-encode and fan out."""
        return self.to_log_dict()


def from_legacy_session_event(row: dict[str, Any]) -> AgentStreamEvent:
    """
    Best-effort map from historical ``session_events.jsonl`` rows to :class:`AgentStreamEvent`.

    Unknown shapes still yield a reasonable OBSERVATION or STATUS.
    """
    t = str(row.get("type") or row.get("legacy_type") or "").lower()
    meta = {k: v for k, v in row.items() if k not in ("type", "ts", "kind", "role", "text")}

    if t == "think":
        return AgentStreamEvent(
            kind=AgentEventKind.STATUS,
            role="agent",
            text=row.get("text") or row.get("summary"),
            legacy_type="think",
            meta=meta,
        )
    if t == "observe":
        return AgentStreamEvent(
            kind=AgentEventKind.OBSERVATION,
            tool_name=row.get("tool_name") or row.get("tool"),
            exit_code=row.get("exit_code"),
            output=row.get("output") or row.get("summary"),
            legacy_type="observe",
            meta=meta,
        )
    if row.get("kind") in AgentEventKind._value2member_map_:
        k = AgentEventKind(row["kind"])
        return AgentStreamEvent(
            kind=k,
            role=row.get("role"),
            text=row.get("text"),
            tool_name=row.get("tool_name"),
            tool_call_id=row.get("tool_call_id"),
            tool_args=row.get("tool_args"),
            output=row.get("output"),
            error=row.get("error"),
            exit_code=row.get("exit_code"),
            meta=dict(row.get("meta") or {}),
            legacy_type=row.get("legacy_type"),
        )

    return AgentStreamEvent(
        kind=AgentEventKind.STATUS,
        text=json.dumps(row, default=str, ensure_ascii=False)[:2000],
        legacy_type=t or "unknown",
        meta=meta,
    )


def append_standard_event(
    workspace: str | Path,
    event: AgentStreamEvent,
    *,
    flat_extras: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Write one row via :func:`append_session_event`, optionally fan-out to SSE subscribers.

    ``flat_extras`` are merged on top (e.g. LLM token/cost fields for STATUS/OBSERVATION).
    When ``session_id`` is set, the same dict persisted to JSONL (including ``ts``) is
    published via :class:`~mini_devin.orchestrator.event_broadcaster.AgentEventBroadcaster`
    using :meth:`AgentStreamEvent.sse_payload_dict` fields plus ``ts`` (no extra LLM work).
    """
    from .event_broadcaster import get_agent_event_broadcaster
    from .session_events import append_session_event

    body = event.sse_payload_dict()
    if flat_extras:
        body = {**body, **flat_extras}
    written = append_session_event(Path(workspace), body)
    if written and session_id:
        get_agent_event_broadcaster().publish(session_id, written)
    return written
