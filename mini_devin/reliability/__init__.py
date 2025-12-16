"""
Reliability Module for Mini-Devin

This module provides reliability improvements for Phase 3.6:
- Minimal reproduction: Extract exact failing test/error and focus fix
- Diff discipline: Smaller patches, prefer apply_patch, avoid large rewrites
- Verification defaults: Auto-detect project type and choose verify commands
- Repair signals: Classify failures and pick repair strategy per class
"""

from .minimal_reproduction import (
    FailureExtractor,
    FailureInfo,
    FailureType,
    extract_failure_info,
)

from .diff_discipline import (
    DiffAnalyzer,
    PatchStrategy,
    DiffMetrics,
    analyze_diff,
    suggest_patch_strategy,
)

from .verification_defaults import (
    ProjectType,
    ProjectDetector,
    VerificationConfig,
    detect_project_type,
    get_verification_commands,
)

from .repair_signals import (
    FailureClass,
    RepairStrategy,
    FailureClassifier,
    classify_failure,
    get_repair_strategy,
)

__all__ = [
    # Minimal reproduction
    "FailureExtractor",
    "FailureInfo",
    "FailureType",
    "extract_failure_info",
    # Diff discipline
    "DiffAnalyzer",
    "PatchStrategy",
    "DiffMetrics",
    "analyze_diff",
    "suggest_patch_strategy",
    # Verification defaults
    "ProjectType",
    "ProjectDetector",
    "VerificationConfig",
    "detect_project_type",
    "get_verification_commands",
    # Repair signals
    "FailureClass",
    "RepairStrategy",
    "FailureClassifier",
    "classify_failure",
    "get_repair_strategy",
]
