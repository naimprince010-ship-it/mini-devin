"""SSE subscriber registry used alongside WebSocket session broadcasts."""

import asyncio
from typing import Any

_sse_subscribers: dict[str, list[asyncio.Queue[str]]] = {}


def fanout_sse(session_id: str, payload: str) -> None:
    for q in list(_sse_subscribers.get(session_id, [])):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def register_sse_queue(session_id: str, maxsize: int = 512) -> asyncio.Queue[str]:
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
    _sse_subscribers.setdefault(session_id, []).append(q)
    return q


def unregister_sse_queue(session_id: str, queue: asyncio.Queue[str]) -> None:
    lst = _sse_subscribers.get(session_id)
    if not lst:
        return
    if queue in lst:
        lst.remove(queue)
    if not lst:
        del _sse_subscribers[session_id]
