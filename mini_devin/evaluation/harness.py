"""
Evaluation Harness for Mini-Devin

This module provides SWE-bench style evaluation with:
- Task execution in isolated containers
- Pass/fail determination
- Detailed result tracking
- Performance metrics
"""

import asyncio
import json
import shutil
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field

from ..orchestrator.agent import Agent


class TaskDifficulty(str, Enum):
    """Difficulty level of an evaluation task."""
    TRIVIAL = "trivial"  # Simple one-line fix
    EASY = "easy"  # Single file, clear fix
    MEDIUM = "medium"  # Multiple files, some exploration
    HARD = "hard"  # Complex logic, multiple components
    EXPERT = "expert"  # Requires deep understanding


class TaskCategory(str, Enum):
    """Category of an evaluation task."""
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    TEST = "test"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"


@dataclass
class EvaluationTask:
    """A task for evaluation."""
    task_id: str
    name: str
    description: str
    repo_url: str | None = None
    repo_path: str | None = None
    
    # Task details
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    category: TaskCategory = TaskCategory.BUG_FIX
    
    # Setup
    setup_commands: list[str] = field(default_factory=list)
    
    # Verification
    test_command: str = ""
    expected_files_modified: list[str] = field(default_factory=list)
    verification_script: str | None = None
    
    # Hints (not shown to agent)
    solution_hint: str = ""
    expected_changes: str = ""
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 600
    max_iterations: int = 30


@dataclass
class EvaluationResult:
    """Result of evaluating a task."""
    task_id: str
    passed: bool
    
    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    
    # Agent metrics
    iterations: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    
    # Verification
    tests_passed: bool = False
    test_output: str = ""
    verification_output: str = ""
    
    # Details
    files_modified: list[str] = field(default_factory=list)
    error_message: str | None = None
    agent_summary: str = ""
    
    # Stop reason
    stop_reason: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "iterations": self.iterations,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "tests_passed": self.tests_passed,
            "test_output": self.test_output,
            "verification_output": self.verification_output,
            "files_modified": self.files_modified,
            "error_message": self.error_message,
            "agent_summary": self.agent_summary,
            "stop_reason": self.stop_reason,
        }


class EvaluationHarness:
    """
    Harness for running evaluation tasks.
    
    Features:
    - Isolated execution environment
    - Automatic setup and teardown
    - Pass/fail determination
    - Detailed metrics collection
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        use_docker: bool = False,
        results_dir: str = "./evaluation_results",
        verbose: bool = True,
    ):
        self.model = model
        self.use_docker = use_docker
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        
        # Results storage
        self.results: list[EvaluationResult] = []
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_task(
        self,
        task: EvaluationTask,
        on_progress: Callable[[str], None] | None = None,
    ) -> EvaluationResult:
        """
        Run a single evaluation task.
        
        Args:
            task: The task to run
            on_progress: Optional callback for progress updates
            
        Returns:
            The evaluation result
        """
        result = EvaluationResult(task_id=task.task_id)
        
        # Create temporary workspace
        workspace = tempfile.mkdtemp(prefix=f"eval_{task.task_id}_")
        workspace_path = Path(workspace)
        
        try:
            # Setup workspace
            if on_progress:
                on_progress(f"Setting up workspace for {task.task_id}")
            
            await self._setup_workspace(task, workspace_path)
            
            # Run setup commands
            for cmd in task.setup_commands:
                if on_progress:
                    on_progress(f"Running setup: {cmd}")
                await self._run_command(cmd, workspace_path)
            
            # Create agent
            agent = Agent(
                working_directory=str(workspace_path),
                model=self.model,
                max_iterations=task.max_iterations,
            )
            
            # Run agent
            if on_progress:
                on_progress(f"Running agent for {task.task_id}")
            
            agent_result = await agent.run(
                task=task.description,
                timeout=task.timeout_seconds,
            )
            
            # Collect agent metrics
            result.iterations = agent_result.iterations if hasattr(agent_result, 'iterations') else 0
            result.total_tokens = agent_result.total_tokens if hasattr(agent_result, 'total_tokens') else 0
            result.tool_calls = agent_result.tool_calls if hasattr(agent_result, 'tool_calls') else 0
            result.files_modified = agent_result.files_modified if hasattr(agent_result, 'files_modified') else []
            result.agent_summary = agent_result.summary if hasattr(agent_result, 'summary') else ""
            result.stop_reason = agent_result.stop_reason if hasattr(agent_result, 'stop_reason') else ""
            
            # Run verification
            if on_progress:
                on_progress(f"Verifying {task.task_id}")
            
            # Run tests
            if task.test_command:
                test_passed, test_output = await self._run_tests(
                    task.test_command, workspace_path
                )
                result.tests_passed = test_passed
                result.test_output = test_output
            else:
                result.tests_passed = True
            
            # Run custom verification script
            if task.verification_script:
                verify_passed, verify_output = await self._run_verification(
                    task.verification_script, workspace_path
                )
                result.verification_output = verify_output
                result.passed = result.tests_passed and verify_passed
            else:
                result.passed = result.tests_passed
            
            # Check expected files modified
            if task.expected_files_modified:
                files_match = all(
                    f in result.files_modified
                    for f in task.expected_files_modified
                )
                if not files_match and result.passed:
                    result.passed = False
                    result.error_message = "Expected files not modified"
            
        except asyncio.TimeoutError:
            result.passed = False
            result.error_message = f"Task timed out after {task.timeout_seconds}s"
            result.stop_reason = "timeout"
            
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            result.stop_reason = "error"
            
        finally:
            # Calculate duration
            result.completed_at = datetime.now(timezone.utc)
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()
            
            # Cleanup workspace
            try:
                shutil.rmtree(workspace)
            except Exception:
                pass
        
        # Store result
        self.results.append(result)
        
        # Save result to file
        self._save_result(result)
        
        return result
    
    async def run_tasks(
        self,
        tasks: list[EvaluationTask],
        parallel: int = 1,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[EvaluationResult]:
        """
        Run multiple evaluation tasks.
        
        Args:
            tasks: List of tasks to run
            parallel: Number of parallel executions
            on_progress: Optional callback for progress updates
            
        Returns:
            List of evaluation results
        """
        if parallel == 1:
            # Sequential execution
            results = []
            for i, task in enumerate(tasks):
                if on_progress:
                    on_progress(f"Running task {i+1}/{len(tasks)}: {task.task_id}")
                result = await self.run_task(task, on_progress)
                results.append(result)
            return results
        else:
            # Parallel execution with semaphore
            semaphore = asyncio.Semaphore(parallel)
            
            async def run_with_semaphore(task: EvaluationTask) -> EvaluationResult:
                async with semaphore:
                    return await self.run_task(task, on_progress)
            
            return await asyncio.gather(*[
                run_with_semaphore(task) for task in tasks
            ])
    
    async def _setup_workspace(
        self,
        task: EvaluationTask,
        workspace: Path,
    ) -> None:
        """Setup the workspace for a task."""
        if task.repo_url:
            # Clone repository
            await self._run_command(
                f"git clone --depth 1 {task.repo_url} .",
                workspace,
            )
        elif task.repo_path:
            # Copy from local path
            src = Path(task.repo_path)
            if src.exists():
                shutil.copytree(src, workspace, dirs_exist_ok=True)
    
    async def _run_command(
        self,
        command: str,
        cwd: Path,
        timeout: int = 60,
    ) -> tuple[int, str, str]:
        """Run a shell command."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            
            return (
                proc.returncode or 0,
                stdout.decode() if stdout else "",
                stderr.decode() if stderr else "",
            )
        except asyncio.TimeoutError:
            proc.kill()
            return -1, "", "Command timed out"
    
    async def _run_tests(
        self,
        test_command: str,
        workspace: Path,
    ) -> tuple[bool, str]:
        """Run tests and return pass/fail with output."""
        returncode, stdout, stderr = await self._run_command(
            test_command, workspace, timeout=120
        )
        
        output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        passed = returncode == 0
        
        return passed, output
    
    async def _run_verification(
        self,
        script: str,
        workspace: Path,
    ) -> tuple[bool, str]:
        """Run custom verification script."""
        # Write script to file
        script_path = workspace / "_verify.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        
        returncode, stdout, stderr = await self._run_command(
            f"bash {script_path}", workspace, timeout=60
        )
        
        output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        passed = returncode == 0
        
        return passed, output
    
    def _save_result(self, result: EvaluationResult) -> None:
        """Save a result to file."""
        result_file = self.results_dir / f"{result.task_id}.json"
        result_file.write_text(json.dumps(result.to_dict(), indent=2))
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all results."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results),
            "avg_duration": sum(r.duration_seconds for r in self.results) / len(self.results),
            "avg_iterations": sum(r.iterations for r in self.results) / len(self.results),
            "avg_tokens": sum(r.total_tokens for r in self.results) / len(self.results),
            "by_difficulty": self._group_by_difficulty(),
            "by_category": self._group_by_category(),
        }
    
    def _group_by_difficulty(self) -> dict[str, dict[str, int]]:
        """Group results by difficulty."""
        # This would require storing difficulty in results
        return {}
    
    def _group_by_category(self) -> dict[str, dict[str, int]]:
        """Group results by category."""
        # This would require storing category in results
        return {}
    
    def generate_report(self, output_path: str | None = None) -> str:
        """Generate a markdown report of results."""
        summary = self.get_summary()
        
        report = f"""# Evaluation Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | {summary['total']} |
| Passed | {summary['passed']} |
| Failed | {summary['failed']} |
| Pass Rate | {summary['pass_rate']:.1%} |
| Avg Duration | {summary.get('avg_duration', 0):.1f}s |
| Avg Iterations | {summary.get('avg_iterations', 0):.1f} |

## Results by Task

| Task ID | Passed | Duration | Iterations | Stop Reason |
|---------|--------|----------|------------|-------------|
"""
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            report += f"| {result.task_id} | {status} | {result.duration_seconds:.1f}s | {result.iterations} | {result.stop_reason} |\n"
        
        report += "\n## Detailed Results\n\n"
        
        for result in self.results:
            report += f"""### {result.task_id}

- **Status**: {"PASSED" if result.passed else "FAILED"}
- **Duration**: {result.duration_seconds:.1f}s
- **Iterations**: {result.iterations}
- **Tokens**: {result.total_tokens}
- **Files Modified**: {', '.join(result.files_modified) or 'None'}

"""
            if result.error_message:
                report += f"**Error**: {result.error_message}\n\n"
            
            if result.agent_summary:
                report += f"**Agent Summary**: {result.agent_summary}\n\n"
        
        if output_path:
            Path(output_path).write_text(report)
        
        return report
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results.clear()
