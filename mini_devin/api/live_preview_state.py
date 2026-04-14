"""
Per-session dev server port for Live Preview (reverse-proxied through the API).

The agent registers a port after ``live_preview`` probe/set; the UI loads the iframe
from same-origin ``/api/sessions/{id}/live-preview/...``.
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Final

_lock = asyncio.Lock()
_ports: dict[str, int] = {}

_DEFAULT_ALLOWED: Final[tuple[int, ...]] = (
    3000,
    3001,
    4200,
    5000,
    5173,
    5174,
    8000,
    8080,
    8888,
    9000,
)


def allowed_ports() -> frozenset[int]:
    raw = (os.environ.get("LIVE_PREVIEW_ALLOWED_PORTS") or "").strip()
    if not raw:
        return frozenset(_DEFAULT_ALLOWED)
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return frozenset(out) if out else frozenset(_DEFAULT_ALLOWED)


def tcp_port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        r = sock.connect_ex((host, port))
        sock.close()
        return r == 0
    except OSError:
        return False


async def set_session_preview_port(session_id: str, port: int) -> bool:
    if port not in allowed_ports():
        return False
    if not tcp_port_open("127.0.0.1", port):
        return False
    async with _lock:
        _ports[session_id] = port
    return True


async def get_session_preview_port(session_id: str) -> int | None:
    async with _lock:
        return _ports.get(session_id)


async def clear_session_preview_port(session_id: str) -> None:
    async with _lock:
        _ports.pop(session_id, None)


def probe_local_ports_sync(ports: list[int]) -> list[int]:
    """Return ports in ``allowed_ports()`` that accept TCP on 127.0.0.1."""
    allow = allowed_ports()
    listening: list[int] = []
    for p in ports:
        try:
            pi = int(p)
        except (TypeError, ValueError):
            continue
        if pi not in allow:
            continue
        if tcp_port_open("127.0.0.1", pi):
            listening.append(pi)
    return listening


def live_preview_probe_hints(*, listening_ports: list[int]) -> dict[str, Any]:
    """
    Extra JSON fields for ``live_preview`` **probe** so models do not treat Railway ``PORT``
    (this API process) as a user's dev UI, and do not use Live Preview for external domains.
    """
    hints: dict[str, Any] = {
        "what_live_preview_is_for": (
            "Reverse-proxy a **workspace dev server** you started on this host's 127.0.0.1 (e.g. Vite/npm). "
            "It does **not** open arbitrary public websites — for https://example.com use **browser_playwright** "
            "or **browser_fetch**."
        ),
    }
    raw = (os.environ.get("PORT") or "").strip()
    if raw.isdigit():
        pe = int(raw)
        if pe in listening_ports:
            hints["platform_bind_warning"] = (
                f"Port {pe} matches the service PORT env — usually **this** Plodder/API process (e.g. Railway), "
                "not a site the user asked to 'preview' by domain name. Registering it loads this app in the iframe."
            )
    return hints


def live_preview_set_port_warning(port: int) -> str | None:
    """Non-None when ``set_active_port`` likely points at this deployment's HTTP listener."""
    raw = (os.environ.get("PORT") or "").strip()
    if raw.isdigit() and int(raw) == port:
        return (
            "This port matches PORT (often this Plodder deployment). The Browser tab will show this service, "
            "not an external URL. Open external sites with browser_playwright or browser_fetch."
        )
    return None
