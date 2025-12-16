"""
Acceptance Test Runner for Mini-Devin

This module runs acceptance tests against real repositories to verify
Mini-Devin works reliably in real-world scenarios.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .scenarios import (
    RepoType,
    ScenarioResult,
    ScenarioType,
    TestScenario,
    get_all_scenarios,
)


@dataclass
class AcceptanceTestResult:
    """Result of running the acceptance test suite."""
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    scenarios_with_repair: int
    results: list[ScenarioResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.passed_scenarios / self.total_scenarios
    
    @property
    def repair_success_rate(self) -> float:
        repair_scenarios = [r for r in self.results if r.repair_attempts > 0]
        if not repair_scenarios:
            return 0.0
        successful = sum(1 for r in repair_scenarios if r.repair_succeeded)
        return successful / len(repair_scenarios)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "failed_scenarios": self.failed_scenarios,
            "scenarios_with_repair": self.scenarios_with_repair,
            "pass_rate": self.pass_rate,
            "repair_success_rate": self.repair_success_rate,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "scenario_type": r.scenario_type.value,
                    "repo_type": r.repo_type.value,
                    "passed": r.passed,
                    "verification_passed": r.verification_passed,
                    "repair_attempts": r.repair_attempts,
                    "repair_succeeded": r.repair_succeeded,
                    "error_message": r.error_message,
                    "duration_seconds": r.duration_seconds,
                    "artifacts_path": r.artifacts_path,
                }
                for r in self.results
            ],
        }


class AcceptanceTestRunner:
    """
    Runner for acceptance tests.
    
    Runs Mini-Devin against real repositories to verify it works
    reliably in real-world scenarios.
    """
    
    def __init__(
        self,
        agent_factory=None,
        output_dir: str = "acceptance_results",
        use_docker: bool = False,
        verbose: bool = True,
    ):
        self.agent_factory = agent_factory
        self.output_dir = output_dir
        self.use_docker = use_docker
        self.verbose = verbose
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[AcceptanceTest] {message}")
    
    async def run_all(
        self,
        scenarios: list[TestScenario] | None = None,
        filter_repo_type: RepoType | None = None,
        filter_scenario_type: ScenarioType | None = None,
    ) -> AcceptanceTestResult:
        """
        Run all acceptance tests.
        
        Args:
            scenarios: Optional list of scenarios to run. If None, runs all.
            filter_repo_type: Optional filter by repo type.
            filter_scenario_type: Optional filter by scenario type.
            
        Returns:
            AcceptanceTestResult with all scenario results.
        """
        if scenarios is None:
            scenarios = get_all_scenarios()
        
        # Apply filters
        if filter_repo_type:
            scenarios = [s for s in scenarios if s.repo_type == filter_repo_type]
        if filter_scenario_type:
            scenarios = [s for s in scenarios if s.scenario_type == filter_scenario_type]
        
        self._log(f"Running {len(scenarios)} acceptance test scenarios")
        
        result = AcceptanceTestResult(
            total_scenarios=len(scenarios),
            passed_scenarios=0,
            failed_scenarios=0,
            scenarios_with_repair=0,
        )
        
        start_time = time.time()
        
        for scenario in scenarios:
            self._log(f"Running scenario: {scenario.name} ({scenario.scenario_id})")
            scenario_result = await self.run_scenario(scenario)
            result.results.append(scenario_result)
            
            if scenario_result.passed:
                result.passed_scenarios += 1
                self._log("  PASSED")
            else:
                result.failed_scenarios += 1
                self._log(f"  FAILED: {scenario_result.error_message}")
            
            if scenario_result.repair_attempts > 0:
                result.scenarios_with_repair += 1
                if scenario_result.repair_succeeded:
                    self._log(f"  Repair succeeded after {scenario_result.repair_attempts} attempts")
        
        result.completed_at = datetime.utcnow()
        result.duration_seconds = time.time() - start_time
        
        # Save results
        self._save_results(result)
        
        self._log(f"Acceptance tests complete: {result.passed_scenarios}/{result.total_scenarios} passed")
        
        return result
    
    async def run_scenario(self, scenario: TestScenario) -> ScenarioResult:
        """
        Run a single test scenario.
        
        Args:
            scenario: The scenario to run.
            
        Returns:
            ScenarioResult with the outcome.
        """
        start_time = time.time()
        
        # Create temporary working directory
        work_dir = tempfile.mkdtemp(prefix=f"mini-devin-{scenario.scenario_id}-")
        
        try:
            # Setup the scenario
            self._log(f"  Setting up scenario in {work_dir}")
            setup_success = await self._setup_scenario(scenario, work_dir)
            if not setup_success:
                return ScenarioResult(
                    scenario_id=scenario.scenario_id,
                    scenario_type=scenario.scenario_type,
                    repo_type=scenario.repo_type,
                    passed=False,
                    verification_passed=False,
                    error_message="Failed to setup scenario",
                    duration_seconds=time.time() - start_time,
                )
            
            # Run the agent
            self._log("  Running agent on task")
            agent_result = await self._run_agent(scenario, work_dir)
            
            # Verify completion
            self._log("  Verifying completion")
            verification_passed, verification_msg = scenario.verify_completion(work_dir)
            
            # Determine if repair was needed
            repair_attempts = agent_result.get("repair_attempts", 0)
            repair_succeeded = agent_result.get("repair_succeeded", False)
            
            # Copy artifacts
            artifacts_path = self._copy_artifacts(scenario, work_dir)
            
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                scenario_type=scenario.scenario_type,
                repo_type=scenario.repo_type,
                passed=verification_passed and agent_result.get("success", False),
                verification_passed=verification_passed,
                repair_attempts=repair_attempts,
                repair_succeeded=repair_succeeded,
                error_message=None if verification_passed else verification_msg,
                duration_seconds=time.time() - start_time,
                artifacts_path=artifacts_path,
            )
            
        except Exception as e:
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                scenario_type=scenario.scenario_type,
                repo_type=scenario.repo_type,
                passed=False,
                verification_passed=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )
        finally:
            # Cleanup
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
    
    async def _setup_scenario(self, scenario: TestScenario, work_dir: str) -> bool:
        """Setup a scenario in the working directory."""
        try:
            # Clone repo if URL provided
            if scenario.repo_url:
                result = subprocess.run(
                    ["git", "clone", scenario.repo_url, work_dir],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    self._log(f"  Failed to clone repo: {result.stderr}")
                    return False
            
            # Copy from local path if provided
            elif scenario.repo_path:
                shutil.copytree(scenario.repo_path, work_dir, dirs_exist_ok=True)
            
            # Run setup commands
            setup_script = scenario.get_setup_script()
            if setup_script:
                result = subprocess.run(
                    setup_script,
                    shell=True,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    self._log(f"  Setup failed: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            self._log(f"  Setup error: {e}")
            return False
    
    async def _run_agent(self, scenario: TestScenario, work_dir: str) -> dict[str, Any]:
        """Run the agent on a scenario."""
        if self.agent_factory is None:
            # Simulate agent run for testing
            return {
                "success": True,
                "repair_attempts": 0,
                "repair_succeeded": False,
            }
        
        try:
            # Create agent
            agent = await self.agent_factory(working_directory=work_dir)
            
            # Get task description
            task_description = scenario.get_task_description()
            
            # Run agent
            result = await agent.run_simple(task_description)
            
            # Check for repair attempts
            repair_loop = agent._get_repair_loop()
            repair_attempts = 0
            repair_succeeded = False
            
            if repair_loop and hasattr(repair_loop, "last_result"):
                last_result = repair_loop.last_result
                if last_result:
                    repair_attempts = last_result.total_attempts
                    repair_succeeded = last_result.status.value == "success"
            
            return {
                "success": agent.state.phase.value == "complete",
                "repair_attempts": repair_attempts,
                "repair_succeeded": repair_succeeded,
                "result": result,
            }
            
        except Exception as e:
            self._log(f"  Agent error: {e}")
            return {
                "success": False,
                "repair_attempts": 0,
                "repair_succeeded": False,
                "error": str(e),
            }
    
    def _copy_artifacts(self, scenario: TestScenario, work_dir: str) -> str | None:
        """Copy artifacts from the run."""
        try:
            artifacts_dir = os.path.join(
                self.output_dir,
                scenario.scenario_id,
                datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            )
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Copy runs directory if it exists
            runs_dir = os.path.join(work_dir, "runs")
            if os.path.exists(runs_dir):
                shutil.copytree(runs_dir, os.path.join(artifacts_dir, "runs"))
            
            # Save git diff
            result = subprocess.run(
                ["git", "diff"],
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                with open(os.path.join(artifacts_dir, "diff.patch"), "w") as f:
                    f.write(result.stdout)
            
            return artifacts_dir
            
        except Exception as e:
            self._log(f"  Failed to copy artifacts: {e}")
            return None
    
    def _save_results(self, result: AcceptanceTestResult) -> None:
        """Save test results to file."""
        results_file = os.path.join(
            self.output_dir,
            f"results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        )
        
        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self._log(f"Results saved to: {results_file}")


async def run_acceptance_tests(
    agent_factory=None,
    output_dir: str = "acceptance_results",
    filter_repo_type: RepoType | None = None,
    filter_scenario_type: ScenarioType | None = None,
) -> AcceptanceTestResult:
    """
    Convenience function to run acceptance tests.
    
    Args:
        agent_factory: Factory function to create agents.
        output_dir: Directory to save results.
        filter_repo_type: Optional filter by repo type.
        filter_scenario_type: Optional filter by scenario type.
        
    Returns:
        AcceptanceTestResult with all scenario results.
    """
    runner = AcceptanceTestRunner(
        agent_factory=agent_factory,
        output_dir=output_dir,
    )
    
    return await runner.run_all(
        filter_repo_type=filter_repo_type,
        filter_scenario_type=filter_scenario_type,
    )
