"""
End-to-End Test Harness for Mini-Devin (Phase 10).

This module provides a test harness for running real tasks through Mini-Devin
and validating that all components work together correctly.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TestStatus(Enum):
    """Status of a test run."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    status: TestStatus
    duration_seconds: float = 0.0
    iterations: int = 0
    tokens_used: int = 0
    error_message: str | None = None
    output: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Result of a test suite run."""
    suite_name: str
    started_at: datetime
    completed_at: datetime | None = None
    results: list[TestResult] = field(default_factory=list)
    
    @property
    def total_tests(self) -> int:
        return len(self.results)
    
    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)
    
    @property
    def failed_tests(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
    
    @property
    def error_tests(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)
    
    @property
    def skipped_tests(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
    
    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests * 100
    
    @property
    def total_duration(self) -> float:
        return sum(r.duration_seconds for r in self.results)
    
    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.results)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "summary": {
                "total": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "errors": self.error_tests,
                "skipped": self.skipped_tests,
                "success_rate": f"{self.success_rate:.1f}%",
                "total_duration": f"{self.total_duration:.2f}s",
                "total_tokens": self.total_tokens,
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "duration": f"{r.duration_seconds:.2f}s",
                    "iterations": r.iterations,
                    "tokens": r.tokens_used,
                    "error": r.error_message,
                }
                for r in self.results
            ],
        }
    
    def to_markdown(self) -> str:
        lines = [
            f"# {self.suite_name} Test Report",
            "",
            f"**Started:** {self.started_at.isoformat()}",
            f"**Completed:** {self.completed_at.isoformat() if self.completed_at else 'In Progress'}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {self.total_tests} |",
            f"| Passed | {self.passed_tests} |",
            f"| Failed | {self.failed_tests} |",
            f"| Errors | {self.error_tests} |",
            f"| Skipped | {self.skipped_tests} |",
            f"| Success Rate | {self.success_rate:.1f}% |",
            f"| Total Duration | {self.total_duration:.2f}s |",
            f"| Total Tokens | {self.total_tokens} |",
            "",
            "## Test Results",
            "",
            "| Test | Status | Duration | Iterations | Tokens |",
            "|------|--------|----------|------------|--------|",
        ]
        
        for r in self.results:
            status_emoji = {
                TestStatus.PASSED: "PASS",
                TestStatus.FAILED: "FAIL",
                TestStatus.ERROR: "ERROR",
                TestStatus.SKIPPED: "SKIP",
                TestStatus.PENDING: "PEND",
                TestStatus.RUNNING: "RUN",
            }.get(r.status, "?")
            
            lines.append(
                f"| {r.test_name} | {status_emoji} | {r.duration_seconds:.2f}s | "
                f"{r.iterations} | {r.tokens_used} |"
            )
        
        if any(r.error_message for r in self.results):
            lines.extend([
                "",
                "## Errors",
                "",
            ])
            for r in self.results:
                if r.error_message:
                    lines.extend([
                        f"### {r.test_name}",
                        "",
                        "```",
                        r.error_message,
                        "```",
                        "",
                    ])
        
        return "\n".join(lines)


class E2ETestHarness:
    """
    End-to-End Test Harness for Mini-Devin.
    
    Runs real tasks through the agent and validates results.
    """
    
    def __init__(
        self,
        output_dir: str = "./e2e_results",
        timeout_seconds: int = 300,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose
        self.current_suite: TestSuiteResult | None = None
    
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[E2E] {message}")
    
    async def run_test(
        self,
        test_name: str,
        task_description: str,
        workspace_dir: str,
        validation_fn: callable | None = None,
        gates_enabled: bool = True,
    ) -> TestResult:
        """
        Run a single end-to-end test.
        
        Args:
            test_name: Name of the test
            task_description: Description of the task to run
            workspace_dir: Directory to run the task in
            validation_fn: Optional function to validate results
            gates_enabled: Whether to enable planner/reviewer gates
            
        Returns:
            TestResult with the outcome
        """
        self._log(f"Running test: {test_name}")
        start_time = time.time()
        
        result = TestResult(
            test_name=test_name,
            status=TestStatus.RUNNING,
        )
        
        try:
            from mini_devin.orchestrator.agent import Agent
            from mini_devin.config.settings import AgentGatesSettings
            from mini_devin.schemas.state import TaskState, TaskGoal, TaskStatus
            from mini_devin.llm.client import create_llm_client
            import uuid
            
            gates = AgentGatesSettings(
                planning_required=gates_enabled,
                review_required=gates_enabled,
                use_llm_planning=False,
            )
            
            llm = create_llm_client()
            
            agent = Agent(
                llm_client=llm,
                working_directory=workspace_dir,
                gates_settings=gates,
                max_iterations=20,
                verbose=self.verbose,
            )
            
            task = TaskState(
                task_id=str(uuid.uuid4()),
                goal=TaskGoal(
                    description=task_description,
                    acceptance_criteria=[],
                ),
            )
            
            task_result = await asyncio.wait_for(
                agent.run(task),
                timeout=self.timeout_seconds,
            )
            
            result.iterations = agent.state.iteration
            result.tokens_used = task_result.total_tokens_used
            
            if task_result.status == TaskStatus.COMPLETED:
                if validation_fn:
                    try:
                        validation_passed = validation_fn(workspace_dir)
                        if validation_passed:
                            result.status = TestStatus.PASSED
                        else:
                            result.status = TestStatus.FAILED
                            result.error_message = "Validation failed"
                    except Exception as e:
                        result.status = TestStatus.FAILED
                        result.error_message = f"Validation error: {str(e)}"
                else:
                    result.status = TestStatus.PASSED
            else:
                result.status = TestStatus.FAILED
                result.error_message = task_result.last_error or "Task did not complete"
                
        except asyncio.TimeoutError:
            result.status = TestStatus.ERROR
            result.error_message = f"Test timed out after {self.timeout_seconds}s"
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
        
        result.duration_seconds = time.time() - start_time
        self._log(f"Test {test_name}: {result.status.value} ({result.duration_seconds:.2f}s)")
        
        return result
    
    async def run_suite(
        self,
        suite_name: str,
        tests: list[dict[str, Any]],
    ) -> TestSuiteResult:
        """
        Run a suite of end-to-end tests.
        
        Args:
            suite_name: Name of the test suite
            tests: List of test configurations
            
        Returns:
            TestSuiteResult with all outcomes
        """
        self._log(f"Starting test suite: {suite_name}")
        
        suite_result = TestSuiteResult(
            suite_name=suite_name,
            started_at=datetime.utcnow(),
        )
        self.current_suite = suite_result
        
        for test_config in tests:
            result = await self.run_test(
                test_name=test_config["name"],
                task_description=test_config["task"],
                workspace_dir=test_config["workspace"],
                validation_fn=test_config.get("validation"),
                gates_enabled=test_config.get("gates_enabled", True),
            )
            suite_result.results.append(result)
        
        suite_result.completed_at = datetime.utcnow()
        
        report_path = self.output_dir / f"{suite_name.lower().replace(' ', '_')}_report.md"
        report_path.write_text(suite_result.to_markdown())
        
        json_path = self.output_dir / f"{suite_name.lower().replace(' ', '_')}_results.json"
        json_path.write_text(json.dumps(suite_result.to_dict(), indent=2))
        
        self._log(f"Suite complete: {suite_result.passed_tests}/{suite_result.total_tests} passed")
        self._log(f"Report saved to: {report_path}")
        
        return suite_result


def validate_file_exists(filepath: str) -> callable:
    """Create a validation function that checks if a file exists."""
    def validator(workspace: str) -> bool:
        return (Path(workspace) / filepath).exists()
    return validator


def validate_file_contains(filepath: str, content: str) -> callable:
    """Create a validation function that checks if a file contains content."""
    def validator(workspace: str) -> bool:
        file_path = Path(workspace) / filepath
        if not file_path.exists():
            return False
        return content in file_path.read_text()
    return validator


def validate_tests_pass(test_command: str = "pytest") -> callable:
    """Create a validation function that runs tests and checks they pass."""
    def validator(workspace: str) -> bool:
        import subprocess
        result = subprocess.run(
            test_command.split(),
            cwd=workspace,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    return validator
