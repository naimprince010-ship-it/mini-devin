"""
Safety Module for Mini-Devin

This module provides safety hardening to prevent dangerous operations:
- Block deleting more than 1 file in one operation
- Block editing more than 300 lines in one iteration
- Block dependency bumps unless explicitly allowed
- Transition to BLOCKED state on safety violations
- Strong STOP/BLOCKED conditions for agent execution
"""

from .guards import (
    SafetyGuard,
    SafetyViolation,
    SafetyPolicy,
    create_safety_guard,
)

from .stop_conditions import (
    StopReason,
    StopSeverity,
    StopCondition,
    StopConditionChecker,
    create_missing_api_key_condition,
    create_permission_denied_condition,
    create_ambiguous_task_condition,
    create_task_completed_condition,
)

__all__ = [
    # Guards
    "SafetyGuard",
    "SafetyViolation",
    "SafetyPolicy",
    "create_safety_guard",
    # Stop conditions
    "StopReason",
    "StopSeverity",
    "StopCondition",
    "StopConditionChecker",
    "create_missing_api_key_condition",
    "create_permission_denied_condition",
    "create_ambiguous_task_condition",
    "create_task_completed_condition",
]
