"""
API Module for Plodder

This module provides the FastAPI backend for the web UI with:
- Streaming tokens and live tool logs
- Multi-session support
- Task management endpoints
- WebSocket connections for real-time updates
"""

from .app import app, connection_manager, create_app, session_manager
from .routes import router
from .websocket import ConnectionManager

_orch_mounted = False


def _mount_orchestration_routes() -> None:
    global _orch_mounted
    if _orch_mounted:
        return
    from .orchestration_routes import build_orchestration_router

    app.include_router(
        build_orchestration_router(session_manager, connection_manager),
        prefix="/api",
    )
    _orch_mounted = True


_mount_orchestration_routes()

__all__ = [
    "create_app",
    "app",
    "router",
    "ConnectionManager",
]
