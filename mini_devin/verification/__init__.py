"""
Verification Module for Mini-Devin

This module provides verification and recovery capabilities:
- VerificationRunner: Run verification checks (lint, typecheck, tests)
- GitManager: Git operations for diff inspection and rollback
- RepairLoop: Bounded repair loops with automatic recovery
"""

from .runner import VerificationRunner, create_verification_runner
from .git_manager import GitManager, create_git_manager
from .repair import RepairLoop, RepairResult, create_repair_loop

__all__ = [
    "VerificationRunner",
    "create_verification_runner",
    "GitManager",
    "create_git_manager",
    "RepairLoop",
    "RepairResult",
    "create_repair_loop",
]
