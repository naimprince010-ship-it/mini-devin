"""
Test Scenarios for Acceptance Testing

This module defines test scenarios that Mini-Devin must pass:
1. Fix failing test
2. Add small feature + test
3. Refactor + update tests
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ScenarioType(str, Enum):
    """Types of test scenarios."""
    FIX_FAILING_TEST = "fix_failing_test"
    ADD_FEATURE = "add_feature"
    REFACTOR = "refactor"


class RepoType(str, Enum):
    """Types of repositories."""
    PYTHON = "python"
    NODE = "node"
    MIXED = "mixed"


@dataclass
class ScenarioResult:
    """Result of running a test scenario."""
    scenario_id: str
    scenario_type: ScenarioType
    repo_type: RepoType
    passed: bool
    verification_passed: bool
    repair_attempts: int = 0
    repair_succeeded: bool = False
    error_message: str | None = None
    duration_seconds: float = 0.0
    artifacts_path: str | None = None


@dataclass
class TestScenario(ABC):
    """Base class for test scenarios."""
    scenario_id: str
    name: str
    description: str
    repo_type: RepoType
    scenario_type: ScenarioType
    repo_url: str | None = None
    repo_path: str | None = None
    setup_commands: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Get the task description for the agent."""
        pass
    
    @abstractmethod
    def get_setup_script(self) -> str:
        """Get the setup script to prepare the scenario."""
        pass
    
    @abstractmethod
    def verify_completion(self, working_dir: str) -> tuple[bool, str]:
        """Verify the scenario was completed successfully."""
        pass


@dataclass
class FixFailingTestScenario(TestScenario):
    """Scenario: Fix a failing test in a repository."""
    failing_test_file: str = ""
    failing_test_name: str = ""
    expected_fix_file: str = ""
    
    def __post_init__(self):
        self.scenario_type = ScenarioType.FIX_FAILING_TEST
    
    def get_task_description(self) -> str:
        return f"""Fix the failing test in this repository.

The test `{self.failing_test_name}` in `{self.failing_test_file}` is failing.
Investigate the cause and fix the underlying code issue (not the test itself).

Acceptance Criteria:
- All tests pass
- The fix is in the source code, not the test
- No new test failures introduced
"""
    
    def get_setup_script(self) -> str:
        return "\n".join(self.setup_commands) if self.setup_commands else ""
    
    def verify_completion(self, working_dir: str) -> tuple[bool, str]:
        import subprocess
        
        for cmd in self.verification_commands:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, f"Verification failed: {cmd}\n{result.stderr}"
        
        return True, "All verification commands passed"


@dataclass
class AddFeatureScenario(TestScenario):
    """Scenario: Add a small feature with tests."""
    feature_description: str = ""
    target_file: str = ""
    test_file: str = ""
    
    def __post_init__(self):
        self.scenario_type = ScenarioType.ADD_FEATURE
    
    def get_task_description(self) -> str:
        return f"""Add a new feature to this repository.

Feature: {self.feature_description}

Requirements:
- Implement the feature in `{self.target_file}`
- Add tests for the feature in `{self.test_file}`
- All existing tests must still pass
- New tests must pass

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in self.acceptance_criteria)}
"""
    
    def get_setup_script(self) -> str:
        return "\n".join(self.setup_commands) if self.setup_commands else ""
    
    def verify_completion(self, working_dir: str) -> tuple[bool, str]:
        import subprocess
        
        for cmd in self.verification_commands:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, f"Verification failed: {cmd}\n{result.stderr}"
        
        return True, "All verification commands passed"


@dataclass
class RefactorScenario(TestScenario):
    """Scenario: Refactor code and update tests."""
    refactor_description: str = ""
    target_files: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.scenario_type = ScenarioType.REFACTOR
    
    def get_task_description(self) -> str:
        files_str = ", ".join(f"`{f}`" for f in self.target_files)
        return f"""Refactor the code in this repository.

Refactoring Task: {self.refactor_description}

Target Files: {files_str}

Requirements:
- Refactor the code as described
- Update any affected tests
- All tests must pass after refactoring
- Code quality should improve (lint checks pass)

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in self.acceptance_criteria)}
"""
    
    def get_setup_script(self) -> str:
        return "\n".join(self.setup_commands) if self.setup_commands else ""
    
    def verify_completion(self, working_dir: str) -> tuple[bool, str]:
        import subprocess
        
        for cmd in self.verification_commands:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, f"Verification failed: {cmd}\n{result.stderr}"
        
        return True, "All verification commands passed"


# Pre-defined test scenarios for each repo type

def get_fixtures_dir() -> str:
    """Get the path to the fixtures directory."""
    import os
    return os.path.join(os.path.dirname(__file__), "fixtures")


def create_python_fix_test_scenario() -> FixFailingTestScenario:
    """Create a Python fix-failing-test scenario."""
    import os
    return FixFailingTestScenario(
        scenario_id="python_fix_test_001",
        name="Fix Failing Python Test",
        description="Fix a failing unit test in a Python calculator module",
        repo_type=RepoType.PYTHON,
        scenario_type=ScenarioType.FIX_FAILING_TEST,
        repo_path=os.path.join(get_fixtures_dir(), "python_calc"),
        failing_test_file="tests/test_calculator.py",
        failing_test_name="test_divide_by_zero",
        expected_fix_file="src/calculator.py",
        setup_commands=[
            "python -m venv .venv",
            "source .venv/bin/activate && pip install pytest",
        ],
        verification_commands=[
            "source .venv/bin/activate && pytest tests/ -v",
        ],
        acceptance_criteria=[
            "All tests pass",
            "Division by zero raises appropriate exception",
            "No changes to test file",
        ],
    )


def create_python_add_feature_scenario() -> AddFeatureScenario:
    """Create a Python add-feature scenario."""
    import os
    return AddFeatureScenario(
        scenario_id="python_add_feature_001",
        name="Add Python Feature",
        description="Add a power function to a Python calculator module",
        repo_type=RepoType.PYTHON,
        scenario_type=ScenarioType.ADD_FEATURE,
        repo_path=os.path.join(get_fixtures_dir(), "python_calc"),
        feature_description="Add a power(base, exponent) function that calculates base^exponent",
        target_file="src/calculator.py",
        test_file="tests/test_calculator.py",
        setup_commands=[
            "python -m venv .venv",
            "source .venv/bin/activate && pip install pytest",
        ],
        verification_commands=[
            "source .venv/bin/activate && pytest tests/ -v",
            "source .venv/bin/activate && python -c 'from src.calculator import power; assert power(2, 3) == 8'",
        ],
        acceptance_criteria=[
            "power() function exists and works correctly",
            "Tests for power() function exist and pass",
            "All existing tests still pass",
        ],
    )


def create_python_refactor_scenario() -> RefactorScenario:
    """Create a Python refactor scenario."""
    import os
    return RefactorScenario(
        scenario_id="python_refactor_001",
        name="Refactor Python Code",
        description="Refactor calculator to use a Calculator class instead of functions",
        repo_type=RepoType.PYTHON,
        scenario_type=ScenarioType.REFACTOR,
        repo_path=os.path.join(get_fixtures_dir(), "python_calc"),
        refactor_description="Convert standalone functions to methods of a Calculator class",
        target_files=["src/calculator.py", "tests/test_calculator.py"],
        setup_commands=[
            "python -m venv .venv",
            "source .venv/bin/activate && pip install pytest ruff",
        ],
        verification_commands=[
            "source .venv/bin/activate && pytest tests/ -v",
            "source .venv/bin/activate && ruff check src/",
        ],
        acceptance_criteria=[
            "Calculator class exists with add, subtract, multiply, divide methods",
            "All tests updated to use Calculator class",
            "All tests pass",
            "Lint checks pass",
        ],
    )


def create_node_fix_test_scenario() -> FixFailingTestScenario:
    """Create a Node.js fix-failing-test scenario."""
    import os
    return FixFailingTestScenario(
        scenario_id="node_fix_test_001",
        name="Fix Failing Node Test",
        description="Fix a failing Jest test in a Node.js string utility module",
        repo_type=RepoType.NODE,
        scenario_type=ScenarioType.FIX_FAILING_TEST,
        repo_path=os.path.join(get_fixtures_dir(), "node_utils"),
        failing_test_file="tests/stringUtils.test.js",
        failing_test_name="should handle empty string in capitalize",
        expected_fix_file="src/stringUtils.js",
        setup_commands=[
            "npm install",
        ],
        verification_commands=[
            "npm test",
        ],
        acceptance_criteria=[
            "All tests pass",
            "Empty string handling is correct",
            "No changes to test file",
        ],
    )


def create_node_add_feature_scenario() -> AddFeatureScenario:
    """Create a Node.js add-feature scenario."""
    import os
    return AddFeatureScenario(
        scenario_id="node_add_feature_001",
        name="Add Node Feature",
        description="Add a truncate function to a Node.js string utility module",
        repo_type=RepoType.NODE,
        scenario_type=ScenarioType.ADD_FEATURE,
        repo_path=os.path.join(get_fixtures_dir(), "node_utils"),
        feature_description="Add a truncate(str, maxLength) function that truncates string with '...'",
        target_file="src/stringUtils.js",
        test_file="tests/stringUtils.test.js",
        setup_commands=[
            "npm install",
        ],
        verification_commands=[
            "npm test",
        ],
        acceptance_criteria=[
            "truncate() function exists and works correctly",
            "Tests for truncate() function exist and pass",
            "All existing tests still pass",
        ],
    )


def create_mixed_refactor_scenario() -> RefactorScenario:
    """Create a mixed repo refactor scenario."""
    import os
    return RefactorScenario(
        scenario_id="mixed_refactor_001",
        name="Refactor Mixed Repo",
        description="Refactor a full-stack app with Python backend and Node frontend",
        repo_type=RepoType.MIXED,
        scenario_type=ScenarioType.REFACTOR,
        repo_path=os.path.join(get_fixtures_dir(), "mixed_app"),
        refactor_description="Extract common validation logic into shared utilities",
        target_files=[
            "backend/utils/validation.py",
            "frontend/src/utils/validation.js",
        ],
        setup_commands=[
            "cd backend && python -m venv .venv && source .venv/bin/activate && pip install pytest",
            "cd frontend && npm install",
        ],
        verification_commands=[
            "cd backend && source .venv/bin/activate && pytest tests/ -v",
            "cd frontend && npm test",
        ],
        acceptance_criteria=[
            "Validation logic is consistent between backend and frontend",
            "All backend tests pass",
            "All frontend tests pass",
        ],
    )


def get_all_scenarios() -> list[TestScenario]:
    """Get all pre-defined test scenarios."""
    return [
        create_python_fix_test_scenario(),
        create_python_add_feature_scenario(),
        create_python_refactor_scenario(),
        create_node_fix_test_scenario(),
        create_node_add_feature_scenario(),
        create_mixed_refactor_scenario(),
    ]
