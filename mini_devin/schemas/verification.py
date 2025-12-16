"""
Verification Schemas for Mini-Devin

This module defines the Pydantic schemas for verification and "done" signals:
- VerificationCheck: Individual checks that must pass
- VerificationResult: Result of running verification
- DoneSignal: Signals that indicate task completion
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Verification Check Types
# =============================================================================


class CheckType(str, Enum):
    """Types of verification checks."""
    # Code quality checks
    LINT = "lint"
    TYPECHECK = "typecheck"
    FORMAT = "format"
    
    # Testing checks
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    E2E_TESTS = "e2e_tests"
    
    # Build checks
    BUILD = "build"
    COMPILE = "compile"
    
    # Git checks
    GIT_CLEAN = "git_clean"
    GIT_COMMITTED = "git_committed"
    NO_CONFLICTS = "no_conflicts"
    
    # Custom checks
    GREP_PATTERN = "grep_pattern"
    FILE_EXISTS = "file_exists"
    FILE_CONTAINS = "file_contains"
    COMMAND_SUCCESS = "command_success"
    API_RESPONSE = "api_response"
    
    # Manual checks
    MANUAL_REVIEW = "manual_review"


class CheckStatus(str, Enum):
    """Status of a verification check."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"  # Check itself errored


class CheckSeverity(str, Enum):
    """Severity of a check - determines if failure blocks completion."""
    BLOCKING = "blocking"  # Must pass for task to complete
    WARNING = "warning"    # Should pass but won't block
    INFO = "info"          # Informational only


# =============================================================================
# Verification Check Definitions
# =============================================================================


class VerificationCheck(BaseModel):
    """Definition of a single verification check."""
    check_id: str
    check_type: CheckType
    name: str = Field(description="Human-readable name of the check")
    description: str | None = None
    severity: CheckSeverity = CheckSeverity.BLOCKING
    
    # Check configuration
    command: str | None = Field(
        default=None,
        description="Command to run for this check"
    )
    working_directory: str | None = None
    timeout_seconds: int = Field(default=300)
    
    # For pattern-based checks
    pattern: str | None = Field(
        default=None,
        description="Pattern to search for (regex)"
    )
    file_path: str | None = Field(
        default=None,
        description="File path for file-based checks"
    )
    expected_content: str | None = None
    
    # For API checks
    url: str | None = None
    expected_status: int | None = None
    expected_response_contains: str | None = None
    
    # Retry configuration
    max_retries: int = Field(default=0)
    retry_delay_seconds: int = Field(default=5)


class CheckResult(BaseModel):
    """Result of running a single verification check."""
    check_id: str
    check_type: CheckType
    status: CheckStatus
    
    # Output
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    
    # Details
    message: str | None = Field(
        default=None,
        description="Human-readable result message"
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the result"
    )
    
    # For test results
    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_skipped: int | None = None
    
    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    
    # Retry info
    attempt_number: int = 1
    retries_remaining: int = 0


# =============================================================================
# Verification Suite
# =============================================================================


class VerificationSuite(BaseModel):
    """A collection of verification checks to run."""
    suite_id: str
    name: str
    description: str | None = None
    
    # Checks to run
    checks: list[VerificationCheck] = Field(default_factory=list)
    
    # Execution configuration
    parallel: bool = Field(
        default=False,
        description="Whether to run checks in parallel"
    )
    stop_on_first_failure: bool = Field(
        default=False,
        description="Whether to stop on first blocking failure"
    )
    
    # Environment setup
    setup_commands: list[str] = Field(
        default_factory=list,
        description="Commands to run before checks"
    )
    teardown_commands: list[str] = Field(
        default_factory=list,
        description="Commands to run after checks"
    )


class VerificationResult(BaseModel):
    """Result of running a verification suite."""
    suite_id: str
    task_id: str
    
    # Overall status
    passed: bool = Field(description="Whether all blocking checks passed")
    
    # Individual results
    check_results: list[CheckResult] = Field(default_factory=list)
    
    # Summary
    total_checks: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_skipped: int = 0
    checks_errored: int = 0
    
    # Blocking failures
    blocking_failures: list[str] = Field(
        default_factory=list,
        description="IDs of blocking checks that failed"
    )
    
    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_duration_ms: int | None = None


# =============================================================================
# Done Signals
# =============================================================================


class DoneSignalType(str, Enum):
    """Types of signals that indicate task completion."""
    # Verification-based
    ALL_TESTS_PASS = "all_tests_pass"
    LINT_CLEAN = "lint_clean"
    BUILD_SUCCESS = "build_success"
    
    # Git-based
    PR_CREATED = "pr_created"
    PR_MERGED = "pr_merged"
    COMMITTED = "committed"
    
    # Output-based
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    PATTERN_FOUND = "pattern_found"
    PATTERN_NOT_FOUND = "pattern_not_found"
    
    # API-based
    API_RETURNS_SUCCESS = "api_returns_success"
    
    # User-based
    USER_APPROVED = "user_approved"
    
    # Composite
    ALL_CRITERIA_MET = "all_criteria_met"


class DoneSignal(BaseModel):
    """A signal that indicates part of the task is complete."""
    signal_id: str
    signal_type: DoneSignalType
    description: str = Field(description="What this signal means")
    
    # Configuration based on type
    command: str | None = None
    file_path: str | None = None
    pattern: str | None = None
    url: str | None = None
    
    # Status
    triggered: bool = False
    triggered_at: datetime | None = None
    evidence: str | None = Field(
        default=None,
        description="Evidence that the signal was triggered"
    )


class CompletionCriteria(BaseModel):
    """Criteria that must be met for task completion."""
    criteria_id: str
    task_id: str
    
    # Required signals
    required_signals: list[DoneSignal] = Field(
        default_factory=list,
        description="Signals that must all be triggered"
    )
    
    # Optional signals
    optional_signals: list[DoneSignal] = Field(
        default_factory=list,
        description="Signals that are nice to have"
    )
    
    # Verification
    verification_suite: VerificationSuite | None = None
    
    # Status
    all_required_met: bool = False
    completion_percentage: float = 0.0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


# =============================================================================
# Pre-built Verification Suites
# =============================================================================


def create_python_verification_suite(
    project_path: str,
    test_command: str = "pytest",
    lint_command: str = "ruff check .",
    typecheck_command: str = "mypy .",
) -> VerificationSuite:
    """Create a standard verification suite for Python projects."""
    return VerificationSuite(
        suite_id="python_standard",
        name="Python Standard Verification",
        description="Standard checks for Python projects",
        checks=[
            VerificationCheck(
                check_id="lint",
                check_type=CheckType.LINT,
                name="Lint Check",
                description="Run linter to check code style",
                command=lint_command,
                working_directory=project_path,
                severity=CheckSeverity.BLOCKING,
            ),
            VerificationCheck(
                check_id="typecheck",
                check_type=CheckType.TYPECHECK,
                name="Type Check",
                description="Run type checker",
                command=typecheck_command,
                working_directory=project_path,
                severity=CheckSeverity.WARNING,
            ),
            VerificationCheck(
                check_id="tests",
                check_type=CheckType.UNIT_TESTS,
                name="Unit Tests",
                description="Run unit tests",
                command=test_command,
                working_directory=project_path,
                severity=CheckSeverity.BLOCKING,
                timeout_seconds=600,
            ),
        ],
    )


def create_javascript_verification_suite(
    project_path: str,
    test_command: str = "npm test",
    lint_command: str = "npm run lint",
    build_command: str = "npm run build",
) -> VerificationSuite:
    """Create a standard verification suite for JavaScript/TypeScript projects."""
    return VerificationSuite(
        suite_id="javascript_standard",
        name="JavaScript Standard Verification",
        description="Standard checks for JavaScript/TypeScript projects",
        checks=[
            VerificationCheck(
                check_id="lint",
                check_type=CheckType.LINT,
                name="Lint Check",
                description="Run ESLint",
                command=lint_command,
                working_directory=project_path,
                severity=CheckSeverity.BLOCKING,
            ),
            VerificationCheck(
                check_id="build",
                check_type=CheckType.BUILD,
                name="Build",
                description="Build the project",
                command=build_command,
                working_directory=project_path,
                severity=CheckSeverity.BLOCKING,
            ),
            VerificationCheck(
                check_id="tests",
                check_type=CheckType.UNIT_TESTS,
                name="Unit Tests",
                description="Run tests",
                command=test_command,
                working_directory=project_path,
                severity=CheckSeverity.BLOCKING,
                timeout_seconds=600,
            ),
        ],
    )


def create_git_verification_suite(project_path: str) -> VerificationSuite:
    """Create verification checks for git status."""
    return VerificationSuite(
        suite_id="git_checks",
        name="Git Verification",
        description="Verify git repository state",
        checks=[
            VerificationCheck(
                check_id="git_status",
                check_type=CheckType.GIT_CLEAN,
                name="Git Status Clean",
                description="Check that all changes are committed",
                command="git status --porcelain",
                working_directory=project_path,
                severity=CheckSeverity.WARNING,
            ),
            VerificationCheck(
                check_id="no_conflicts",
                check_type=CheckType.NO_CONFLICTS,
                name="No Merge Conflicts",
                description="Check for merge conflict markers",
                command="git diff --check",
                working_directory=project_path,
                severity=CheckSeverity.BLOCKING,
            ),
        ],
    )
