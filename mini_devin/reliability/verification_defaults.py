"""
Verification Defaults Module for Mini-Devin

This module auto-detects project types and provides appropriate verification commands:
- Detect project type from files (Python, Node, mixed, etc.)
- Provide default lint, test, and typecheck commands
- Support fallback commands when primary commands fail
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ProjectType(str, Enum):
    """Types of projects that can be detected."""
    
    PYTHON = "python"
    NODE = "node"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class VerificationCommand:
    """A single verification command with fallbacks."""
    
    name: str
    primary: str
    fallbacks: list[str] = field(default_factory=list)
    required: bool = True
    timeout_seconds: int = 60
    
    def get_commands(self) -> list[str]:
        """Get all commands including fallbacks."""
        return [self.primary] + self.fallbacks


@dataclass
class VerificationConfig:
    """
    Configuration for project verification.
    
    Contains all commands needed to verify a project.
    """
    
    project_type: ProjectType
    lint_commands: list[VerificationCommand] = field(default_factory=list)
    test_commands: list[VerificationCommand] = field(default_factory=list)
    typecheck_commands: list[VerificationCommand] = field(default_factory=list)
    build_commands: list[VerificationCommand] = field(default_factory=list)
    custom_commands: list[VerificationCommand] = field(default_factory=list)
    
    def get_all_commands(self) -> list[VerificationCommand]:
        """Get all verification commands."""
        return (
            self.lint_commands +
            self.test_commands +
            self.typecheck_commands +
            self.build_commands +
            self.custom_commands
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "project_type": self.project_type.value,
            "lint_commands": [c.primary for c in self.lint_commands],
            "test_commands": [c.primary for c in self.test_commands],
            "typecheck_commands": [c.primary for c in self.typecheck_commands],
            "build_commands": [c.primary for c in self.build_commands],
        }


class ProjectDetector:
    """
    Detects project type from directory contents.
    
    Examines files and directories to determine the project type
    and appropriate verification commands.
    """
    
    # File patterns for project detection
    PYTHON_INDICATORS = [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "Pipfile",
        "poetry.lock",
        "*.py",
    ]
    
    NODE_INDICATORS = [
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "*.js",
        "*.jsx",
    ]
    
    TYPESCRIPT_INDICATORS = [
        "tsconfig.json",
        "*.ts",
        "*.tsx",
    ]
    
    RUST_INDICATORS = [
        "Cargo.toml",
        "Cargo.lock",
        "*.rs",
    ]
    
    GO_INDICATORS = [
        "go.mod",
        "go.sum",
        "*.go",
    ]
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
    
    def detect(self) -> ProjectType:
        """
        Detect the project type.
        
        Returns:
            ProjectType enum value
        """
        indicators = {
            ProjectType.PYTHON: self._check_indicators(self.PYTHON_INDICATORS),
            ProjectType.NODE: self._check_indicators(self.NODE_INDICATORS),
            ProjectType.TYPESCRIPT: self._check_indicators(self.TYPESCRIPT_INDICATORS),
            ProjectType.RUST: self._check_indicators(self.RUST_INDICATORS),
            ProjectType.GO: self._check_indicators(self.GO_INDICATORS),
        }
        
        # Count how many project types are detected
        detected = [pt for pt, found in indicators.items() if found]
        
        if len(detected) == 0:
            return ProjectType.UNKNOWN
        elif len(detected) == 1:
            return detected[0]
        elif ProjectType.TYPESCRIPT in detected and ProjectType.NODE in detected:
            return ProjectType.TYPESCRIPT
        else:
            return ProjectType.MIXED
    
    def _check_indicators(self, patterns: list[str]) -> bool:
        """Check if any indicator patterns match."""
        for pattern in patterns:
            if pattern.startswith("*"):
                # Glob pattern
                ext = pattern[1:]
                for root, _, files in os.walk(self.workspace_path):
                    for f in files:
                        if f.endswith(ext):
                            return True
            else:
                # Exact file
                if (self.workspace_path / pattern).exists():
                    return True
        return False
    
    def get_package_manager(self) -> str | None:
        """Detect the package manager used."""
        if (self.workspace_path / "poetry.lock").exists():
            return "poetry"
        elif (self.workspace_path / "Pipfile.lock").exists():
            return "pipenv"
        elif (self.workspace_path / "pnpm-lock.yaml").exists():
            return "pnpm"
        elif (self.workspace_path / "yarn.lock").exists():
            return "yarn"
        elif (self.workspace_path / "package-lock.json").exists():
            return "npm"
        elif (self.workspace_path / "requirements.txt").exists():
            return "pip"
        return None
    
    def get_test_framework(self) -> str | None:
        """Detect the test framework used."""
        # Check Python test frameworks
        if (self.workspace_path / "pytest.ini").exists():
            return "pytest"
        if (self.workspace_path / "pyproject.toml").exists():
            try:
                content = (self.workspace_path / "pyproject.toml").read_text()
                if "pytest" in content:
                    return "pytest"
                if "unittest" in content:
                    return "unittest"
            except Exception:
                pass
        
        # Check Node test frameworks
        if (self.workspace_path / "package.json").exists():
            try:
                import json
                content = json.loads((self.workspace_path / "package.json").read_text())
                dev_deps = content.get("devDependencies", {})
                deps = content.get("dependencies", {})
                all_deps = {**deps, **dev_deps}
                
                if "jest" in all_deps:
                    return "jest"
                if "mocha" in all_deps:
                    return "mocha"
                if "vitest" in all_deps:
                    return "vitest"
            except Exception:
                pass
        
        return None


# Default verification configurations for each project type
PYTHON_VERIFICATION = VerificationConfig(
    project_type=ProjectType.PYTHON,
    lint_commands=[
        VerificationCommand(
            name="ruff",
            primary="ruff check .",
            fallbacks=["flake8 .", "pylint **/*.py"],
            required=False,
        ),
    ],
    test_commands=[
        VerificationCommand(
            name="pytest",
            primary="pytest",
            fallbacks=["python -m pytest", "python -m unittest discover"],
            required=True,
            timeout_seconds=120,
        ),
    ],
    typecheck_commands=[
        VerificationCommand(
            name="mypy",
            primary="mypy .",
            fallbacks=["pyright ."],
            required=False,
        ),
    ],
)

NODE_VERIFICATION = VerificationConfig(
    project_type=ProjectType.NODE,
    lint_commands=[
        VerificationCommand(
            name="eslint",
            primary="npm run lint",
            fallbacks=["npx eslint .", "yarn lint"],
            required=False,
        ),
    ],
    test_commands=[
        VerificationCommand(
            name="npm test",
            primary="npm test",
            fallbacks=["yarn test", "npx jest", "npx mocha"],
            required=True,
            timeout_seconds=120,
        ),
    ],
    typecheck_commands=[],
)

TYPESCRIPT_VERIFICATION = VerificationConfig(
    project_type=ProjectType.TYPESCRIPT,
    lint_commands=[
        VerificationCommand(
            name="eslint",
            primary="npm run lint",
            fallbacks=["npx eslint . --ext .ts,.tsx", "yarn lint"],
            required=False,
        ),
    ],
    test_commands=[
        VerificationCommand(
            name="npm test",
            primary="npm test",
            fallbacks=["yarn test", "npx jest", "npx vitest"],
            required=True,
            timeout_seconds=120,
        ),
    ],
    typecheck_commands=[
        VerificationCommand(
            name="tsc",
            primary="npx tsc --noEmit",
            fallbacks=["npm run typecheck", "yarn typecheck"],
            required=False,
        ),
    ],
)

RUST_VERIFICATION = VerificationConfig(
    project_type=ProjectType.RUST,
    lint_commands=[
        VerificationCommand(
            name="clippy",
            primary="cargo clippy",
            fallbacks=[],
            required=False,
        ),
    ],
    test_commands=[
        VerificationCommand(
            name="cargo test",
            primary="cargo test",
            fallbacks=[],
            required=True,
            timeout_seconds=180,
        ),
    ],
    typecheck_commands=[
        VerificationCommand(
            name="cargo check",
            primary="cargo check",
            fallbacks=[],
            required=True,
        ),
    ],
    build_commands=[
        VerificationCommand(
            name="cargo build",
            primary="cargo build",
            fallbacks=[],
            required=False,
        ),
    ],
)

GO_VERIFICATION = VerificationConfig(
    project_type=ProjectType.GO,
    lint_commands=[
        VerificationCommand(
            name="golint",
            primary="golangci-lint run",
            fallbacks=["go vet ./..."],
            required=False,
        ),
    ],
    test_commands=[
        VerificationCommand(
            name="go test",
            primary="go test ./...",
            fallbacks=[],
            required=True,
            timeout_seconds=120,
        ),
    ],
    typecheck_commands=[],
    build_commands=[
        VerificationCommand(
            name="go build",
            primary="go build ./...",
            fallbacks=[],
            required=False,
        ),
    ],
)

UNKNOWN_VERIFICATION = VerificationConfig(
    project_type=ProjectType.UNKNOWN,
    lint_commands=[],
    test_commands=[],
    typecheck_commands=[],
)

# Mapping of project types to verification configs
VERIFICATION_CONFIGS: dict[ProjectType, VerificationConfig] = {
    ProjectType.PYTHON: PYTHON_VERIFICATION,
    ProjectType.NODE: NODE_VERIFICATION,
    ProjectType.TYPESCRIPT: TYPESCRIPT_VERIFICATION,
    ProjectType.RUST: RUST_VERIFICATION,
    ProjectType.GO: GO_VERIFICATION,
    ProjectType.UNKNOWN: UNKNOWN_VERIFICATION,
}


def detect_project_type(workspace_path: str) -> ProjectType:
    """
    Convenience function to detect project type.
    
    Args:
        workspace_path: Path to the workspace directory
        
    Returns:
        ProjectType enum value
    """
    detector = ProjectDetector(workspace_path)
    return detector.detect()


def get_verification_commands(
    workspace_path: str,
    project_type: ProjectType | None = None,
) -> VerificationConfig:
    """
    Get verification commands for a workspace.
    
    Args:
        workspace_path: Path to the workspace directory
        project_type: Optional project type (auto-detected if not provided)
        
    Returns:
        VerificationConfig with appropriate commands
    """
    if project_type is None:
        project_type = detect_project_type(workspace_path)
    
    # Get base config
    config = VERIFICATION_CONFIGS.get(project_type, UNKNOWN_VERIFICATION)
    
    # For mixed projects, combine Python and Node configs
    if project_type == ProjectType.MIXED:
        config = VerificationConfig(
            project_type=ProjectType.MIXED,
            lint_commands=(
                PYTHON_VERIFICATION.lint_commands +
                NODE_VERIFICATION.lint_commands
            ),
            test_commands=(
                PYTHON_VERIFICATION.test_commands +
                NODE_VERIFICATION.test_commands
            ),
            typecheck_commands=(
                PYTHON_VERIFICATION.typecheck_commands +
                NODE_VERIFICATION.typecheck_commands
            ),
        )
    
    return config


def get_quick_verification_command(workspace_path: str) -> str | None:
    """
    Get a single quick verification command for the workspace.
    
    This is useful for quick checks during development.
    
    Args:
        workspace_path: Path to the workspace directory
        
    Returns:
        A single verification command string or None
    """
    project_type = detect_project_type(workspace_path)
    
    quick_commands = {
        ProjectType.PYTHON: "pytest -x -q",  # Stop on first failure, quiet
        ProjectType.NODE: "npm test -- --bail",  # Stop on first failure
        ProjectType.TYPESCRIPT: "npm test -- --bail",
        ProjectType.RUST: "cargo test -- --test-threads=1",
        ProjectType.GO: "go test -failfast ./...",
    }
    
    return quick_commands.get(project_type)
