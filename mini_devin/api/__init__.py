"""
API Module for Mini-Devin

This module provides the FastAPI backend for the web UI with:
- Streaming tokens and live tool logs
- Multi-session support
- Task management endpoints
- WebSocket connections for real-time updates
"""

from .app import create_app, app
from .routes import router
from .websocket import ConnectionManager

__all__ = [
    "create_app",
    "app",
    "router",
    "ConnectionManager",
]
