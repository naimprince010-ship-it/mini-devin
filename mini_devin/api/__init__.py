"""
API Module for Plodder.

Keep this package initializer lazy. Lower-level modules such as
``mini_devin.sessions.manager`` import ``mini_devin.api.websocket`` for message
types; eagerly importing the FastAPI app here would pull the session manager
back in and create a circular import.
"""

from __future__ import annotations

import importlib
from typing import Any

_orch_mounted = False


def _mount_orchestration_routes() -> None:
    global _orch_mounted
    if _orch_mounted:
        return

    from .app import app, connection_manager, session_manager
    from .orchestration_routes import build_orchestration_router

    if not getattr(app.state, "orchestration_routes_mounted", False):
        app.include_router(
            build_orchestration_router(session_manager, connection_manager),
            prefix="/api",
        )
        app.state.orchestration_routes_mounted = True
    _orch_mounted = True


def __getattr__(name: str) -> Any:
    if name in {"app", "connection_manager", "create_app", "session_manager"}:
        app_module = importlib.import_module(".app", __name__)

        _mount_orchestration_routes()
        return getattr(app_module, name)

    if name == "router":
        from .routes import router

        return router

    if name == "ConnectionManager":
        from .websocket import ConnectionManager

        return ConnectionManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_app",
    "app",
    "router",
    "ConnectionManager",
]
