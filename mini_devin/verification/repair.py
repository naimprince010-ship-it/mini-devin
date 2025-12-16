"""
Repair Loop for Mini-Devin

This module implements bounded repair loops that automatically attempt
to fix issues found during verification, with a maximum number of
iterations before escalating to the user.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Awaitable

from ..schemas.verification import (
    CheckResult,
    CheckStatus,
    VerificationResult,
    VerificationSuite,
)
from .runner import VerificationRunner
from .git_manager import GitManager


class RepairStatus(str, Enum):
    """Status of a repair attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ESCALATED = "escalated"
    SKIPPED = "skipped"


@dataclass
class RepairAttempt:
    """Record of a single repair attempt."""
    attempt_number: int
    started_at: datetime
    completed_at: datetime | None = None
    
    # What was tried
    action_taken: str = ""
    files_modified: list[str] = field(default_factory=list)
    
    # Results
    status: RepairStatus = RepairStatus.FAILED
    verification_result: VerificationResult | None = None
    error_message: str | None = None
    
    # Diff of changes made
    diff: str = ""


@dataclass
class RepairResult:
    """Result of a repair loop."""
    status: RepairStatus
    message: str
    
    # Attempts made
    attempts: list[RepairAttempt] = field(default_factory=list)
    total_attempts: int = 0
    
    # Final state
    final_verification: VerificationResult | None = None
    all_issues_fixed: bool = False
    remaining_issues: list[str] = field(default_factory=list)
    
    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_duration_ms: int | None = None


# Type for repair functions
RepairFunction = Callable[[list[CheckResult], int], Awaitable[tuple[bool, str, list[str]]]]


class RepairLoop:
    """
    Bounded repair loop for automatic issue fixing.
    
    This class orchestrates the repair process:
    1. Run verification
    2. If issues found, attempt repair
    3. Re-run verification
    4. Repeat until fixed or max iterations reached
    5. Escalate to user if not fixed
    """
    
    DEFAULT_MAX_ITERATIONS = 3
    
    def __init__(
        self,
        verification_runner: VerificationRunner,
        git_manager: GitManager | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        verbose: bool = True,
    ):
        self.verification_runner = verification_runner
        self.git_manager = git_manager
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Registered repair functions for different check types
        self._repair_functions: dict[str, RepairFunction] = {}
        
        # Default repair function (can be overridden)
        self._default_repair_function: RepairFunction | None = None
    
    def register_repair_function(
        self,
        check_type: str,
        repair_fn: RepairFunction,
    ) -> None:
        """
        Register a repair function for a specific check type.
        
        The repair function should:
        - Take a list of failed CheckResults and attempt number
        - Return (success: bool, action_description: str, files_modified: list[str])
        """
        self._repair_functions[check_type] = repair_fn
    
    def set_default_repair_function(self, repair_fn: RepairFunction) -> None:
        """Set the default repair function for unhandled check types."""
        self._default_repair_function = repair_fn
    
    async def run(
        self,
        suite: VerificationSuite,
        task_id: str = "default",
        repair_fn: RepairFunction | None = None,
    ) -> RepairResult:
        """
        Run the repair loop.
        
        Args:
            suite: The verification suite to run
            task_id: Task identifier
            repair_fn: Optional repair function to use for all repairs
        
        Returns:
            RepairResult with the outcome of the repair loop
        """
        started_at = datetime.utcnow()
        attempts: list[RepairAttempt] = []
        checkpoint_id = f"repair_start_{task_id}"
        
        # Create initial checkpoint if git manager available
        if self.git_manager:
            await self.git_manager.create_checkpoint(
                checkpoint_id,
                "Pre-repair checkpoint",
            )
        
        # Initial verification
        if self.verbose:
            print("[Repair] Running initial verification...")
        
        verification_result = await self.verification_runner.run_suite(suite, task_id)
        
        if verification_result.passed:
            return RepairResult(
                status=RepairStatus.SUCCESS,
                message="All checks passed on initial verification",
                attempts=[],
                total_attempts=0,
                final_verification=verification_result,
                all_issues_fixed=True,
                remaining_issues=[],
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
        
        # Repair loop
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"[Repair] Iteration {iteration}/{self.max_iterations}")
            
            attempt_started = datetime.utcnow()
            
            # Get failed checks
            failed_checks = [
                r for r in verification_result.check_results
                if r.status == CheckStatus.FAILED
            ]
            
            if not failed_checks:
                break
            
            # Attempt repair
            repair_success = False
            action_taken = ""
            files_modified: list[str] = []
            error_message: str | None = None
            
            try:
                if repair_fn:
                    # Use provided repair function
                    repair_success, action_taken, files_modified = await repair_fn(
                        failed_checks, iteration
                    )
                else:
                    # Use registered repair functions
                    repair_success, action_taken, files_modified = await self._attempt_repairs(
                        failed_checks, iteration
                    )
            except Exception as e:
                error_message = str(e)
                if self.verbose:
                    print(f"[Repair] Error during repair: {e}")
            
            # Get diff of changes
            diff = ""
            if self.git_manager and files_modified:
                diff_result = await self.git_manager.get_diff()
                diff = diff_result.diff_text
            
            # Re-run verification
            verification_result = await self.verification_runner.run_suite(suite, task_id)
            
            attempt = RepairAttempt(
                attempt_number=iteration,
                started_at=attempt_started,
                completed_at=datetime.utcnow(),
                action_taken=action_taken,
                files_modified=files_modified,
                status=RepairStatus.SUCCESS if verification_result.passed else RepairStatus.FAILED,
                verification_result=verification_result,
                error_message=error_message,
                diff=diff,
            )
            attempts.append(attempt)
            
            if verification_result.passed:
                if self.verbose:
                    print(f"[Repair] All issues fixed after {iteration} attempt(s)")
                
                return RepairResult(
                    status=RepairStatus.SUCCESS,
                    message=f"All issues fixed after {iteration} attempt(s)",
                    attempts=attempts,
                    total_attempts=iteration,
                    final_verification=verification_result,
                    all_issues_fixed=True,
                    remaining_issues=[],
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )
        
        # Max iterations reached - escalate
        remaining_issues = [
            f"{r.check_id}: {r.message}"
            for r in verification_result.check_results
            if r.status == CheckStatus.FAILED
        ]
        
        if self.verbose:
            print(f"[Repair] Max iterations reached. Escalating with {len(remaining_issues)} remaining issues.")
        
        return RepairResult(
            status=RepairStatus.ESCALATED,
            message=f"Could not fix all issues after {self.max_iterations} attempts. Escalating to user.",
            attempts=attempts,
            total_attempts=self.max_iterations,
            final_verification=verification_result,
            all_issues_fixed=False,
            remaining_issues=remaining_issues,
            started_at=started_at,
            completed_at=datetime.utcnow(),
        )
    
    async def _attempt_repairs(
        self,
        failed_checks: list[CheckResult],
        iteration: int,
    ) -> tuple[bool, str, list[str]]:
        """
        Attempt to repair failed checks using registered repair functions.
        
        Returns:
            (success, action_description, files_modified)
        """
        actions_taken = []
        all_files_modified = []
        any_success = False
        
        # Group failed checks by type
        checks_by_type: dict[str, list[CheckResult]] = {}
        for check in failed_checks:
            check_type = check.check_type.value
            if check_type not in checks_by_type:
                checks_by_type[check_type] = []
            checks_by_type[check_type].append(check)
        
        # Attempt repair for each type
        for check_type, checks in checks_by_type.items():
            repair_fn = self._repair_functions.get(check_type) or self._default_repair_function
            
            if repair_fn:
                try:
                    success, action, files = await repair_fn(checks, iteration)
                    if success:
                        any_success = True
                    actions_taken.append(action)
                    all_files_modified.extend(files)
                except Exception as e:
                    actions_taken.append(f"Failed to repair {check_type}: {e}")
            else:
                actions_taken.append(f"No repair function for {check_type}")
        
        return any_success, "; ".join(actions_taken), all_files_modified
    
    async def rollback_to_start(self, task_id: str) -> bool:
        """Rollback to the pre-repair checkpoint."""
        if not self.git_manager:
            return False
        
        checkpoint_id = f"repair_start_{task_id}"
        result = await self.git_manager.rollback_to_checkpoint(checkpoint_id, hard=True)
        return result.status.value == "success"
    
    async def run_single_check_repair(
        self,
        check: CheckResult,
        repair_fn: RepairFunction,
        max_attempts: int = 3,
    ) -> RepairResult:
        """
        Run repair loop for a single check.
        
        Useful for targeted repairs of specific issues.
        """
        started_at = datetime.utcnow()
        attempts: list[RepairAttempt] = []
        
        for iteration in range(1, max_attempts + 1):
            attempt_started = datetime.utcnow()
            
            try:
                success, action, files = await repair_fn([check], iteration)
            except Exception as e:
                attempts.append(RepairAttempt(
                    attempt_number=iteration,
                    started_at=attempt_started,
                    completed_at=datetime.utcnow(),
                    action_taken=f"Error: {e}",
                    status=RepairStatus.FAILED,
                    error_message=str(e),
                ))
                continue
            
            attempts.append(RepairAttempt(
                attempt_number=iteration,
                started_at=attempt_started,
                completed_at=datetime.utcnow(),
                action_taken=action,
                files_modified=files,
                status=RepairStatus.SUCCESS if success else RepairStatus.FAILED,
            ))
            
            if success:
                return RepairResult(
                    status=RepairStatus.SUCCESS,
                    message=f"Check repaired after {iteration} attempt(s)",
                    attempts=attempts,
                    total_attempts=iteration,
                    all_issues_fixed=True,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )
        
        return RepairResult(
            status=RepairStatus.ESCALATED,
            message=f"Could not repair check after {max_attempts} attempts",
            attempts=attempts,
            total_attempts=max_attempts,
            all_issues_fixed=False,
            remaining_issues=[f"{check.check_id}: {check.message}"],
            started_at=started_at,
            completed_at=datetime.utcnow(),
        )


# Pre-built repair functions

async def lint_repair_function(
    failed_checks: list[CheckResult],
    iteration: int,
) -> tuple[bool, str, list[str]]:
    """
    Attempt to auto-fix lint issues.
    
    This is a placeholder - in practice, this would call the LLM
    to fix the specific lint errors.
    """
    # Extract lint errors from check output
    errors = []
    for check in failed_checks:
        if check.stdout:
            errors.append(check.stdout)
        if check.stderr:
            errors.append(check.stderr)
    
    # In a real implementation, this would:
    # 1. Parse the lint errors
    # 2. Call the LLM to generate fixes
    # 3. Apply the fixes
    # 4. Return success/failure
    
    return False, f"Lint repair attempted (iteration {iteration})", []


async def test_repair_function(
    failed_checks: list[CheckResult],
    iteration: int,
) -> tuple[bool, str, list[str]]:
    """
    Attempt to fix failing tests.
    
    This is a placeholder - in practice, this would call the LLM
    to analyze test failures and fix the code.
    """
    # Extract test failures from check output
    failures = []
    for check in failed_checks:
        if check.stdout:
            failures.append(check.stdout)
    
    # In a real implementation, this would:
    # 1. Parse the test failures
    # 2. Identify the failing tests and error messages
    # 3. Call the LLM to analyze and fix
    # 4. Apply the fixes
    # 5. Return success/failure
    
    return False, f"Test repair attempted (iteration {iteration})", []


def create_repair_loop(
    working_directory: str,
    max_iterations: int = 3,
    verbose: bool = True,
) -> RepairLoop:
    """Create a repair loop with default settings."""
    from .runner import create_verification_runner
    from .git_manager import create_git_manager
    
    verification_runner = create_verification_runner(working_directory, verbose)
    git_manager = create_git_manager(working_directory, verbose)
    
    repair_loop = RepairLoop(
        verification_runner=verification_runner,
        git_manager=git_manager,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    
    # Register default repair functions
    repair_loop.register_repair_function("lint", lint_repair_function)
    repair_loop.register_repair_function("unit_tests", test_repair_function)
    
    return repair_loop
