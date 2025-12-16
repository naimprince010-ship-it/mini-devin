"""
Phase 3.5 Task Validation

This module contains the task runner and definitions for Phase 3.5 validation:
- 10 real-world tasks across Python, Node, and mixed repositories
- Success/failure reporting
- Infrastructure validation
"""

from .task_runner import (
    TaskStatus,
    TaskResult,
    TaskDefinition,
    Phase35TaskRunner,
    PHASE35_TASKS,
)

__all__ = [
    "TaskStatus",
    "TaskResult",
    "TaskDefinition",
    "Phase35TaskRunner",
    "PHASE35_TASKS",
]
