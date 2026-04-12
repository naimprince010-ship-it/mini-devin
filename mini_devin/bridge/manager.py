"""
In-memory bridge registry: cloud agent forwards terminal exec to a WebSocket connected
from the user's machine (scripts/local_bridge.py).
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from typing import Any

from fastapi import WebSocket


class BridgeManager:
    """Maps Plodder session_id ↔ active local bridge WebSocket."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sockets: dict[str, WebSocket] = {}
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._tokens: dict[str, tuple[str, float]] = {}

    async def create_token(self, session_id: str, ttl_seconds: float = 900.0) -> str:
        token = secrets.token_urlsafe(32)
        async with self._lock:
            self._tokens[token] = (session_id, time.time() + ttl_seconds)
        return token

    async def consume_token(self, token: str) -> str | None:
        now = time.time()
        async with self._lock:
            entry = self._tokens.pop(token, None)
        if not entry:
            return None
        session_id, exp = entry
        if now > exp:
            return None
        return session_id

    async def register(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            old = self._sockets.get(session_id)
            if old is not None and old is not ws:
                try:
                    await old.close(code=4000)
                except Exception:
                    pass
            self._sockets[session_id] = ws

    async def unregister(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            if self._sockets.get(session_id) is ws:
                del self._sockets[session_id]

    def is_connected(self, session_id: str) -> bool:
        return session_id in self._sockets

    async def handle_bridge_message(self, session_id: str, payload: dict[str, Any]) -> None:
        if payload.get("type") != "exec_response":
            return
        rid = payload.get("id")
        if not rid or not isinstance(rid, str):
            return
        async with self._lock:
            fut = self._pending.pop(rid, None)
        if fut is not None and not fut.done():
            fut.set_result(payload)

    async def forward_exec(
        self,
        session_id: str,
        command: str,
        cwd: str,
        timeout_seconds: float,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        loop = asyncio.get_event_loop()
        rid = str(uuid.uuid4())
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        async with self._lock:
            ws = self._sockets.get(session_id)
            if ws is None:
                return None
            self._pending[rid] = fut
        msg = {
            "type": "exec_request",
            "id": rid,
            "command": command,
            "cwd": cwd,
            "timeout_seconds": min(float(timeout_seconds), 600.0),
            "env": env or {},
        }
        try:
            await ws.send_text(json.dumps(msg))
            result = await asyncio.wait_for(
                fut,
                timeout=min(float(timeout_seconds) + 30.0, 630.0),
            )
            return result
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending.pop(rid, None)
            return {
                "type": "exec_response",
                "id": rid,
                "exit_code": -1,
                "stdout": "",
                "stderr": "Local bridge: execution timed out waiting for response.",
            }
        except Exception as e:
            async with self._lock:
                self._pending.pop(rid, None)
            return {
                "type": "exec_response",
                "id": rid,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Local bridge error: {e}",
            }


_bridge_manager: BridgeManager | None = None


def get_bridge_manager() -> BridgeManager:
    global _bridge_manager
    if _bridge_manager is None:
        _bridge_manager = BridgeManager()
    return _bridge_manager
