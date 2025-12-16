"""
Sessions Module for Mini-Devin

This module provides multi-session support for:
- Concurrent task execution
- Session lifecycle management
- Task tracking and results
- Resource isolation
"""

from .manager import SessionManager, Session, Task, TaskResult
from .db_manager import DatabaseSessionManager

__all__ = [
    "SessionManager",
    "DatabaseSessionManager",
    "Session",
    "Task",
    "TaskResult",
]
