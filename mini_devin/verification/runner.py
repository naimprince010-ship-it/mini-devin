"""
Verification Runner for Mini-Devin

This module implements the verification runner that executes checks
like lint, typecheck, and tests after agent changes.
"""

import asyncio
import os
from datetime import datetime
from typing import Callable

from ..schemas.verification import (
    CheckResult,
    CheckSeverity,
    CheckStatus,
    CheckType,
    VerificationCheck,
    VerificationResult,
    VerificationSuite,
)


class VerificationRunner:
    """
    Runs verification checks against a codebase.
    
    This runner executes lint, typecheck, test, and custom checks
    to verify that agent changes are correct.
    """
    
    def __init__(
        self,
        working_directory: str,
        timeout_default: int = 300,
        verbose: bool = True,
    ):
        self.working_directory = working_directory
        self.timeout_default = timeout_default
        self.verbose = verbose
        self._custom_checkers: dict[str, Callable] = {}
    
    def register_custom_checker(
        self,
        check_id: str,
        checker: Callable[[VerificationCheck], CheckResult],
    ) -> None:
        """Register a custom checker function."""
        self._custom_checkers[check_id] = checker
    
    async def run_suite(
        self,
        suite: VerificationSuite,
        task_id: str = "default",
    ) -> VerificationResult:
        """Run all checks in a verification suite."""
        started_at = datetime.utcnow()
        check_results: list[CheckResult] = []
        blocking_failures: list[str] = []
        
        # Run setup commands
        for cmd in suite.setup_commands:
            await self._run_command(cmd, self.working_directory, timeout=60)
        
        # Run checks
        if suite.parallel:
            # Run all checks in parallel
            tasks = [
                self._run_check(check)
                for check in suite.checks
            ]
            check_results = await asyncio.gather(*tasks)
        else:
            # Run checks sequentially
            for check in suite.checks:
                result = await self._run_check(check)
                check_results.append(result)
                
                # Stop on first blocking failure if configured
                if (
                    suite.stop_on_first_failure
                    and result.status == CheckStatus.FAILED
                    and check.severity == CheckSeverity.BLOCKING
                ):
                    break
        
        # Run teardown commands
        for cmd in suite.teardown_commands:
            await self._run_command(cmd, self.working_directory, timeout=60)
        
        # Calculate summary
        completed_at = datetime.utcnow()
        total_checks = len(check_results)
        checks_passed = sum(1 for r in check_results if r.status == CheckStatus.PASSED)
        checks_failed = sum(1 for r in check_results if r.status == CheckStatus.FAILED)
        checks_skipped = sum(1 for r in check_results if r.status == CheckStatus.SKIPPED)
        checks_errored = sum(1 for r in check_results if r.status == CheckStatus.ERROR)
        
        # Find blocking failures
        for i, result in enumerate(check_results):
            if (
                result.status == CheckStatus.FAILED
                and suite.checks[i].severity == CheckSeverity.BLOCKING
            ):
                blocking_failures.append(result.check_id)
        
        passed = len(blocking_failures) == 0
        
        return VerificationResult(
            suite_id=suite.suite_id,
            task_id=task_id,
            passed=passed,
            check_results=check_results,
            total_checks=total_checks,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            checks_skipped=checks_skipped,
            checks_errored=checks_errored,
            blocking_failures=blocking_failures,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_ms=int((completed_at - started_at).total_seconds() * 1000),
        )
    
    async def run_check(self, check: VerificationCheck) -> CheckResult:
        """Run a single verification check."""
        return await self._run_check(check)
    
    async def _run_check(self, check: VerificationCheck) -> CheckResult:
        """Internal method to run a single check with retries."""
        last_result: CheckResult | None = None
        
        for attempt in range(check.max_retries + 1):
            result = await self._execute_check(check, attempt + 1)
            last_result = result
            
            if result.status == CheckStatus.PASSED:
                return result
            
            if attempt < check.max_retries:
                await asyncio.sleep(check.retry_delay_seconds)
        
        return last_result or CheckResult(
            check_id=check.check_id,
            check_type=check.check_type,
            status=CheckStatus.ERROR,
            message="No result produced",
        )
    
    async def _execute_check(
        self,
        check: VerificationCheck,
        attempt: int,
    ) -> CheckResult:
        """Execute a single check attempt."""
        started_at = datetime.utcnow()
        
        try:
            # Check if there's a custom checker
            if check.check_id in self._custom_checkers:
                result = self._custom_checkers[check.check_id](check)
                result.attempt_number = attempt
                return result
            
            # Handle different check types
            if check.check_type in (
                CheckType.LINT,
                CheckType.TYPECHECK,
                CheckType.FORMAT,
                CheckType.UNIT_TESTS,
                CheckType.INTEGRATION_TESTS,
                CheckType.E2E_TESTS,
                CheckType.BUILD,
                CheckType.COMPILE,
                CheckType.COMMAND_SUCCESS,
            ):
                return await self._run_command_check(check, attempt, started_at)
            
            elif check.check_type == CheckType.FILE_EXISTS:
                return self._run_file_exists_check(check, attempt, started_at)
            
            elif check.check_type == CheckType.FILE_CONTAINS:
                return self._run_file_contains_check(check, attempt, started_at)
            
            elif check.check_type == CheckType.GREP_PATTERN:
                return await self._run_grep_check(check, attempt, started_at)
            
            elif check.check_type in (CheckType.GIT_CLEAN, CheckType.GIT_COMMITTED, CheckType.NO_CONFLICTS):
                return await self._run_git_check(check, attempt, started_at)
            
            else:
                return CheckResult(
                    check_id=check.check_id,
                    check_type=check.check_type,
                    status=CheckStatus.ERROR,
                    message=f"Unsupported check type: {check.check_type}",
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    attempt_number=attempt,
                )
                
        except Exception as e:
            completed_at = datetime.utcnow()
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.ERROR,
                message=str(e),
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=int((completed_at - started_at).total_seconds() * 1000),
                attempt_number=attempt,
            )
    
    async def _run_command_check(
        self,
        check: VerificationCheck,
        attempt: int,
        started_at: datetime,
    ) -> CheckResult:
        """Run a command-based check."""
        if not check.command:
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.ERROR,
                message="No command specified for check",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                attempt_number=attempt,
            )
        
        working_dir = check.working_directory or self.working_directory
        timeout = check.timeout_seconds or self.timeout_default
        
        stdout, stderr, exit_code = await self._run_command(
            check.command,
            working_dir,
            timeout,
        )
        
        completed_at = datetime.utcnow()
        
        # Parse test results if this is a test check
        tests_passed = None
        tests_failed = None
        tests_skipped = None
        
        if check.check_type in (
            CheckType.UNIT_TESTS,
            CheckType.INTEGRATION_TESTS,
            CheckType.E2E_TESTS,
        ):
            tests_passed, tests_failed, tests_skipped = self._parse_test_output(stdout)
        
        # Determine status
        if exit_code == 0:
            status = CheckStatus.PASSED
            message = f"{check.name} passed"
        else:
            status = CheckStatus.FAILED
            message = f"{check.name} failed with exit code {exit_code}"
        
        return CheckResult(
            check_id=check.check_id,
            check_type=check.check_type,
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            message=message,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=int((completed_at - started_at).total_seconds() * 1000),
            attempt_number=attempt,
            retries_remaining=check.max_retries - attempt,
        )
    
    def _run_file_exists_check(
        self,
        check: VerificationCheck,
        attempt: int,
        started_at: datetime,
    ) -> CheckResult:
        """Check if a file exists."""
        if not check.file_path:
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.ERROR,
                message="No file path specified",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                attempt_number=attempt,
            )
        
        file_path = check.file_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.working_directory, file_path)
        
        exists = os.path.exists(file_path)
        completed_at = datetime.utcnow()
        
        return CheckResult(
            check_id=check.check_id,
            check_type=check.check_type,
            status=CheckStatus.PASSED if exists else CheckStatus.FAILED,
            message=f"File {'exists' if exists else 'does not exist'}: {check.file_path}",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=int((completed_at - started_at).total_seconds() * 1000),
            attempt_number=attempt,
        )
    
    def _run_file_contains_check(
        self,
        check: VerificationCheck,
        attempt: int,
        started_at: datetime,
    ) -> CheckResult:
        """Check if a file contains expected content."""
        if not check.file_path or not check.expected_content:
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.ERROR,
                message="File path or expected content not specified",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                attempt_number=attempt,
            )
        
        file_path = check.file_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.working_directory, file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            contains = check.expected_content in content
            completed_at = datetime.utcnow()
            
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.PASSED if contains else CheckStatus.FAILED,
                message=f"File {'contains' if contains else 'does not contain'} expected content",
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=int((completed_at - started_at).total_seconds() * 1000),
                attempt_number=attempt,
            )
        except FileNotFoundError:
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.FAILED,
                message=f"File not found: {check.file_path}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                attempt_number=attempt,
            )
    
    async def _run_grep_check(
        self,
        check: VerificationCheck,
        attempt: int,
        started_at: datetime,
    ) -> CheckResult:
        """Run a grep pattern check."""
        if not check.pattern:
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.ERROR,
                message="No pattern specified",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                attempt_number=attempt,
            )
        
        path = check.file_path or "."
        if not os.path.isabs(path):
            path = os.path.join(self.working_directory, path)
        
        cmd = f"rg -l '{check.pattern}' {path}"
        stdout, stderr, exit_code = await self._run_command(
            cmd,
            self.working_directory,
            check.timeout_seconds or 60,
        )
        
        completed_at = datetime.utcnow()
        found = exit_code == 0 and stdout.strip()
        
        return CheckResult(
            check_id=check.check_id,
            check_type=check.check_type,
            status=CheckStatus.PASSED if found else CheckStatus.FAILED,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            message=f"Pattern {'found' if found else 'not found'}: {check.pattern}",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=int((completed_at - started_at).total_seconds() * 1000),
            attempt_number=attempt,
        )
    
    async def _run_git_check(
        self,
        check: VerificationCheck,
        attempt: int,
        started_at: datetime,
    ) -> CheckResult:
        """Run a git-related check."""
        working_dir = check.working_directory or self.working_directory
        
        if check.check_type == CheckType.GIT_CLEAN:
            cmd = "git status --porcelain"
            stdout, stderr, exit_code = await self._run_command(cmd, working_dir, 30)
            is_clean = exit_code == 0 and not stdout.strip()
            
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.PASSED if is_clean else CheckStatus.FAILED,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                message="Working directory is clean" if is_clean else "Working directory has uncommitted changes",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                attempt_number=attempt,
            )
        
        elif check.check_type == CheckType.NO_CONFLICTS:
            cmd = "git diff --check"
            stdout, stderr, exit_code = await self._run_command(cmd, working_dir, 30)
            no_conflicts = exit_code == 0
            
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.PASSED if no_conflicts else CheckStatus.FAILED,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                message="No merge conflicts" if no_conflicts else "Merge conflicts detected",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                attempt_number=attempt,
            )
        
        else:
            return CheckResult(
                check_id=check.check_id,
                check_type=check.check_type,
                status=CheckStatus.ERROR,
                message=f"Unsupported git check type: {check.check_type}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                attempt_number=attempt,
            )
    
    async def _run_command(
        self,
        command: str,
        working_dir: str,
        timeout: int,
    ) -> tuple[str, str, int]:
        """Run a shell command and return stdout, stderr, exit_code."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode or 0
            
            if self.verbose:
                print(f"[Verification] Command: {command}")
                print(f"[Verification] Exit code: {exit_code}")
            
            return stdout, stderr, exit_code
            
        except asyncio.TimeoutError:
            return "", f"Command timed out after {timeout}s", -1
        except Exception as e:
            return "", str(e), -1
    
    def _parse_test_output(self, output: str) -> tuple[int | None, int | None, int | None]:
        """Parse test output to extract pass/fail/skip counts."""
        import re
        
        # Try pytest format: "X passed, Y failed, Z skipped"
        pytest_match = re.search(
            r"(\d+) passed.*?(\d+) failed.*?(\d+) skipped",
            output,
            re.IGNORECASE,
        )
        if pytest_match:
            return (
                int(pytest_match.group(1)),
                int(pytest_match.group(2)),
                int(pytest_match.group(3)),
            )
        
        # Try simpler pytest format
        passed_match = re.search(r"(\d+) passed", output, re.IGNORECASE)
        failed_match = re.search(r"(\d+) failed", output, re.IGNORECASE)
        skipped_match = re.search(r"(\d+) skipped", output, re.IGNORECASE)
        
        passed = int(passed_match.group(1)) if passed_match else None
        failed = int(failed_match.group(1)) if failed_match else None
        skipped = int(skipped_match.group(1)) if skipped_match else None
        
        if passed is not None or failed is not None:
            return passed, failed, skipped
        
        return None, None, None
    
    # Convenience methods for common checks
    
    async def run_lint(
        self,
        command: str = "ruff check .",
        severity: CheckSeverity = CheckSeverity.BLOCKING,
    ) -> CheckResult:
        """Run lint check."""
        check = VerificationCheck(
            check_id="lint",
            check_type=CheckType.LINT,
            name="Lint Check",
            command=command,
            working_directory=self.working_directory,
            severity=severity,
        )
        return await self._run_check(check)
    
    async def run_typecheck(
        self,
        command: str = "mypy .",
        severity: CheckSeverity = CheckSeverity.WARNING,
    ) -> CheckResult:
        """Run type check."""
        check = VerificationCheck(
            check_id="typecheck",
            check_type=CheckType.TYPECHECK,
            name="Type Check",
            command=command,
            working_directory=self.working_directory,
            severity=severity,
        )
        return await self._run_check(check)
    
    async def run_tests(
        self,
        command: str = "pytest",
        severity: CheckSeverity = CheckSeverity.BLOCKING,
        timeout: int = 600,
    ) -> CheckResult:
        """Run tests."""
        check = VerificationCheck(
            check_id="tests",
            check_type=CheckType.UNIT_TESTS,
            name="Unit Tests",
            command=command,
            working_directory=self.working_directory,
            severity=severity,
            timeout_seconds=timeout,
        )
        return await self._run_check(check)
    
    async def run_build(
        self,
        command: str,
        severity: CheckSeverity = CheckSeverity.BLOCKING,
    ) -> CheckResult:
        """Run build check."""
        check = VerificationCheck(
            check_id="build",
            check_type=CheckType.BUILD,
            name="Build",
            command=command,
            working_directory=self.working_directory,
            severity=severity,
        )
        return await self._run_check(check)


def create_verification_runner(
    working_directory: str,
    verbose: bool = True,
) -> VerificationRunner:
    """Create a verification runner with default settings."""
    return VerificationRunner(
        working_directory=working_directory,
        verbose=verbose,
    )
