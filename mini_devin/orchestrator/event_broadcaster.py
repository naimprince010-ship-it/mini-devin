"""
In-process fan-out for :class:`~mini_devin.orchestrator.standard_events.AgentStreamEvent` rows.

Used by :func:`~mini_devin.orchestrator.standard_events.append_standard_event` (when ``session_id``
is set) and by SSE routes such as ``GET /api/events/{session_id}``. No LLM calls — queues only.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

_DEFAULT_QUEUE_MAX = 512


class AgentEventBroadcaster:
    """
    Per-``session_id`` subscriber queues; payloads are JSON strings of the same
    dict written to ``session_events.jsonl`` (includes ``ts``).
    """

    def __init__(self, *, queue_maxsize: int = _DEFAULT_QUEUE_MAX) -> None:
        self._queue_maxsize = max(32, queue_maxsize)
        self._subs: dict[str, list[asyncio.Queue[str]]] = {}

    def subscribe(self, session_id: str) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subs.setdefault(session_id, []).append(q)
        return q

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[str]) -> None:
        lst = self._subs.get(session_id)
        if not lst:
            return
        if queue in lst:
            lst.remove(queue)
        if not lst:
            del self._subs[session_id]

    def publish(self, session_id: str, row: dict[str, Any]) -> None:
        """Push ``row`` (already includes ``ts``) to all subscribers as one JSON line."""
        line = json.dumps(row, default=str, ensure_ascii=False)
        for q in list(self._subs.get(session_id, [])):
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                pass


_default_broadcaster: AgentEventBroadcaster | None = None


def get_agent_event_broadcaster() -> AgentEventBroadcaster:
    global _default_broadcaster
    if _default_broadcaster is None:
        _default_broadcaster = AgentEventBroadcaster()
    return _default_broadcaster
