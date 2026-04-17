"""
Monkey-patch ConnectionManager.broadcast_to_session for SSE fan-out + register SSE helpers.

Avoids editing websocket.py when tooling cannot patch large files.
"""

from __future__ import annotations

import asyncio
from fastapi import WebSocket

from .sse_fanout import fanout_sse, register_sse_queue, unregister_sse_queue
from .websocket import ConnectionManager, WebSocketMessage

_installed = False


async def _broadcast_to_session_patched(
    self: ConnectionManager,
    session_id: str,
    message: WebSocketMessage,
) -> int:
    message.session_id = session_id
    payload = message.to_json()
    fanout_sse(session_id, payload)

    if session_id not in self.session_connections:
        return 0

    success_count = 0
    disconnected: list[WebSocket] = []

    for connection in self.session_connections[session_id]:
        try:
            await connection.send_text(payload)
            success_count += 1
        except Exception:
            disconnected.append(connection)

    for conn in disconnected:
        self.disconnect(conn)

    return success_count


def register_sse_queue_method(self: ConnectionManager, session_id: str, maxsize: int = 512) -> asyncio.Queue[str]:
    return register_sse_queue(session_id, maxsize)


def unregister_sse_queue_method(
    self: ConnectionManager, session_id: str, queue: asyncio.Queue[str]
) -> None:
    unregister_sse_queue(session_id, queue)


def install_streaming_patch() -> None:
    global _installed
    if _installed:
        return
    ConnectionManager.broadcast_to_session = _broadcast_to_session_patched
    ConnectionManager.register_sse_queue = register_sse_queue_method
    ConnectionManager.unregister_sse_queue = unregister_sse_queue_method
    _installed = True
