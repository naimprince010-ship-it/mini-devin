"""
Repair Signals Module for Mini-Devin

This module classifies failures and provides repair strategies:
- Classify failures into categories (lint, type, test, runtime)
- Provide specific repair strategies for each failure class
- Track repair success rates for learning
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .minimal_reproduction import FailureInfo, FailureType


class FailureClass(str, Enum):
    """High-level classification of failures."""
    
    LINT = "lint"
    """Linting errors (style, formatting, unused imports)."""
    
    TYPE = "type"
    """Type errors (mypy, tsc, type mismatches)."""
    
    TEST = "test"
    """Test failures (assertions, test errors)."""
    
    RUNTIME = "runtime"
    """Runtime errors (exceptions, crashes)."""
    
    BUILD = "build"
    """Build errors (compilation, bundling)."""
    
    IMPORT = "import"
    """Import errors (missing modules, circular imports)."""
    
    SYNTAX = "syntax"
    """Syntax errors (parsing failures)."""
    
    UNKNOWN = "unknown"
    """Unknown or unclassified errors."""


class RepairStrategy(str, Enum):
    """Strategies for repairing different failure classes."""
    
    # Lint strategies
    AUTO_FIX = "auto_fix"
    """Run auto-fix tool (ruff --fix, eslint --fix)."""
    
    REMOVE_UNUSED = "remove_unused"
    """Remove unused imports/variables."""
    
    FORMAT_CODE = "format_code"
    """Run code formatter (black, prettier)."""
    
    # Type strategies
    ADD_TYPE_ANNOTATION = "add_type_annotation"
    """Add missing type annotations."""
    
    FIX_TYPE_MISMATCH = "fix_type_mismatch"
    """Fix type mismatch by correcting the value or annotation."""
    
    ADD_TYPE_IGNORE = "add_type_ignore"
    """Add type: ignore comment (last resort)."""
    
    # Test strategies
    FIX_ASSERTION = "fix_assertion"
    """Fix the code to match the expected assertion."""
    
    UPDATE_TEST = "update_test"
    """Update the test to match new behavior (if intentional)."""
    
    FIX_TEST_SETUP = "fix_test_setup"
    """Fix test setup/fixtures."""
    
    # Runtime strategies
    ADD_ERROR_HANDLING = "add_error_handling"
    """Add try/except or error checking."""
    
    FIX_NULL_CHECK = "fix_null_check"
    """Add null/undefined checks."""
    
    FIX_LOGIC = "fix_logic"
    """Fix logical error in code."""
    
    # Import strategies
    ADD_IMPORT = "add_import"
    """Add missing import statement."""
    
    FIX_IMPORT_PATH = "fix_import_path"
    """Fix incorrect import path."""
    
    INSTALL_DEPENDENCY = "install_dependency"
    """Install missing dependency."""
    
    # Syntax strategies
    FIX_SYNTAX = "fix_syntax"
    """Fix syntax error (missing bracket, colon, etc.)."""
    
    # Build strategies
    FIX_BUILD_CONFIG = "fix_build_config"
    """Fix build configuration."""
    
    # General strategies
    ROLLBACK = "rollback"
    """Rollback to previous state and try different approach."""
    
    ASK_USER = "ask_user"
    """Ask user for guidance."""


@dataclass
class RepairPlan:
    """
    A plan for repairing a failure.
    
    Contains the strategy and specific actions to take.
    """
    
    failure_class: FailureClass
    primary_strategy: RepairStrategy
    fallback_strategies: list[RepairStrategy] = field(default_factory=list)
    specific_actions: list[str] = field(default_factory=list)
    auto_fixable: bool = False
    confidence: float = 0.5
    max_attempts: int = 3
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "failure_class": self.failure_class.value,
            "primary_strategy": self.primary_strategy.value,
            "fallback_strategies": [s.value for s in self.fallback_strategies],
            "specific_actions": self.specific_actions,
            "auto_fixable": self.auto_fixable,
            "confidence": self.confidence,
            "max_attempts": self.max_attempts,
        }
    
    def get_repair_prompt(self, failure: FailureInfo) -> str:
        """Generate a prompt for the repair."""
        parts = [
            f"Repair {self.failure_class.value} failure using {self.primary_strategy.value} strategy.",
            "",
            f"Failure: {failure.message}",
        ]
        
        if failure.location:
            parts.append(f"Location: {failure.location}")
        
        if self.specific_actions:
            parts.append("")
            parts.append("Specific actions:")
            for action in self.specific_actions:
                parts.append(f"  - {action}")
        
        if self.fallback_strategies:
            parts.append("")
            parts.append(f"If this doesn't work, try: {', '.join(s.value for s in self.fallback_strategies)}")
        
        return "\n".join(parts)


class FailureClassifier:
    """
    Classifies failures and provides repair strategies.
    
    Uses failure information to determine the best repair approach.
    """
    
    # Mapping from FailureType to FailureClass
    TYPE_TO_CLASS: dict[FailureType, FailureClass] = {
        FailureType.TEST_FAILURE: FailureClass.TEST,
        FailureType.TEST_ERROR: FailureClass.TEST,
        FailureType.ASSERTION_ERROR: FailureClass.TEST,
        FailureType.IMPORT_ERROR: FailureClass.IMPORT,
        FailureType.SYNTAX_ERROR: FailureClass.SYNTAX,
        FailureType.TYPE_ERROR: FailureClass.TYPE,
        FailureType.RUNTIME_ERROR: FailureClass.RUNTIME,
        FailureType.LINT_ERROR: FailureClass.LINT,
        FailureType.UNKNOWN: FailureClass.UNKNOWN,
    }
    
    # Default repair strategies for each failure class
    DEFAULT_STRATEGIES: dict[FailureClass, RepairPlan] = {
        FailureClass.LINT: RepairPlan(
            failure_class=FailureClass.LINT,
            primary_strategy=RepairStrategy.AUTO_FIX,
            fallback_strategies=[
                RepairStrategy.REMOVE_UNUSED,
                RepairStrategy.FORMAT_CODE,
            ],
            auto_fixable=True,
            confidence=0.9,
            max_attempts=2,
        ),
        FailureClass.TYPE: RepairPlan(
            failure_class=FailureClass.TYPE,
            primary_strategy=RepairStrategy.FIX_TYPE_MISMATCH,
            fallback_strategies=[
                RepairStrategy.ADD_TYPE_ANNOTATION,
                RepairStrategy.ADD_TYPE_IGNORE,
            ],
            auto_fixable=False,
            confidence=0.7,
            max_attempts=3,
        ),
        FailureClass.TEST: RepairPlan(
            failure_class=FailureClass.TEST,
            primary_strategy=RepairStrategy.FIX_ASSERTION,
            fallback_strategies=[
                RepairStrategy.FIX_LOGIC,
                RepairStrategy.FIX_TEST_SETUP,
            ],
            auto_fixable=False,
            confidence=0.6,
            max_attempts=3,
        ),
        FailureClass.RUNTIME: RepairPlan(
            failure_class=FailureClass.RUNTIME,
            primary_strategy=RepairStrategy.FIX_LOGIC,
            fallback_strategies=[
                RepairStrategy.ADD_ERROR_HANDLING,
                RepairStrategy.FIX_NULL_CHECK,
            ],
            auto_fixable=False,
            confidence=0.5,
            max_attempts=3,
        ),
        FailureClass.IMPORT: RepairPlan(
            failure_class=FailureClass.IMPORT,
            primary_strategy=RepairStrategy.ADD_IMPORT,
            fallback_strategies=[
                RepairStrategy.FIX_IMPORT_PATH,
                RepairStrategy.INSTALL_DEPENDENCY,
            ],
            auto_fixable=True,
            confidence=0.8,
            max_attempts=2,
        ),
        FailureClass.SYNTAX: RepairPlan(
            failure_class=FailureClass.SYNTAX,
            primary_strategy=RepairStrategy.FIX_SYNTAX,
            fallback_strategies=[RepairStrategy.ROLLBACK],
            auto_fixable=False,
            confidence=0.7,
            max_attempts=2,
        ),
        FailureClass.BUILD: RepairPlan(
            failure_class=FailureClass.BUILD,
            primary_strategy=RepairStrategy.FIX_BUILD_CONFIG,
            fallback_strategies=[RepairStrategy.ASK_USER],
            auto_fixable=False,
            confidence=0.4,
            max_attempts=2,
        ),
        FailureClass.UNKNOWN: RepairPlan(
            failure_class=FailureClass.UNKNOWN,
            primary_strategy=RepairStrategy.FIX_LOGIC,
            fallback_strategies=[
                RepairStrategy.ROLLBACK,
                RepairStrategy.ASK_USER,
            ],
            auto_fixable=False,
            confidence=0.3,
            max_attempts=2,
        ),
    }
    
    def classify(self, failure: FailureInfo) -> FailureClass:
        """
        Classify a failure into a failure class.
        
        Args:
            failure: The failure information
            
        Returns:
            FailureClass enum value
        """
        # First, try direct mapping from failure type
        if failure.failure_type in self.TYPE_TO_CLASS:
            return self.TYPE_TO_CLASS[failure.failure_type]
        
        # Analyze message for additional hints
        message_lower = failure.message.lower()
        
        # Check for lint-related keywords
        if any(kw in message_lower for kw in ["unused", "import", "style", "format", "indent"]):
            return FailureClass.LINT
        
        # Check for type-related keywords
        if any(kw in message_lower for kw in ["type", "expected", "incompatible", "annotation"]):
            return FailureClass.TYPE
        
        # Check for test-related keywords
        if any(kw in message_lower for kw in ["assert", "test", "expect", "should"]):
            return FailureClass.TEST
        
        # Check for import-related keywords
        if any(kw in message_lower for kw in ["import", "module", "not found", "cannot find"]):
            return FailureClass.IMPORT
        
        # Check for syntax-related keywords
        if any(kw in message_lower for kw in ["syntax", "parse", "unexpected token"]):
            return FailureClass.SYNTAX
        
        return FailureClass.UNKNOWN
    
    def get_repair_plan(self, failure: FailureInfo) -> RepairPlan:
        """
        Get a repair plan for a failure.
        
        Args:
            failure: The failure information
            
        Returns:
            RepairPlan with strategy and actions
        """
        failure_class = self.classify(failure)
        base_plan = self.DEFAULT_STRATEGIES.get(
            failure_class,
            self.DEFAULT_STRATEGIES[FailureClass.UNKNOWN],
        )
        
        # Create a copy with specific actions based on the failure
        plan = RepairPlan(
            failure_class=base_plan.failure_class,
            primary_strategy=base_plan.primary_strategy,
            fallback_strategies=base_plan.fallback_strategies.copy(),
            specific_actions=self._get_specific_actions(failure, failure_class),
            auto_fixable=base_plan.auto_fixable,
            confidence=base_plan.confidence,
            max_attempts=base_plan.max_attempts,
        )
        
        return plan
    
    def _get_specific_actions(
        self,
        failure: FailureInfo,
        failure_class: FailureClass,
    ) -> list[str]:
        """Generate specific actions based on the failure."""
        actions = []
        
        if failure_class == FailureClass.LINT:
            if "unused" in failure.message.lower():
                actions.append(f"Remove unused import/variable at {failure.location}")
            elif "format" in failure.message.lower():
                actions.append("Run code formatter on the file")
            else:
                actions.append("Run linter with --fix flag")
        
        elif failure_class == FailureClass.TYPE:
            if failure.expected and failure.actual:
                actions.append(f"Change type from {failure.actual} to {failure.expected}")
            else:
                actions.append(f"Fix type error at {failure.location}")
            actions.append("Check if the value needs to be converted or the annotation updated")
        
        elif failure_class == FailureClass.TEST:
            if failure.test_name:
                actions.append(f"Focus on fixing test: {failure.test_name}")
            if failure.expected and failure.actual:
                actions.append(f"Expected: {failure.expected}")
                actions.append(f"Actual: {failure.actual}")
                actions.append("Fix the implementation to return the expected value")
            if failure.location:
                actions.append(f"Check the code at {failure.location}")
        
        elif failure_class == FailureClass.IMPORT:
            if "not found" in failure.message.lower() or "no module" in failure.message.lower():
                # Extract module name
                import re
                match = re.search(r"['\"]([^'\"]+)['\"]", failure.message)
                if match:
                    module = match.group(1)
                    actions.append(f"Add import for '{module}'")
                    actions.append("Or install the package if it's a dependency")
        
        elif failure_class == FailureClass.SYNTAX:
            if failure.location:
                actions.append(f"Check syntax at {failure.location}")
            actions.append("Look for missing brackets, colons, or quotes")
        
        elif failure_class == FailureClass.RUNTIME:
            if "none" in failure.message.lower() or "null" in failure.message.lower():
                actions.append("Add null/None check before accessing the value")
            elif "index" in failure.message.lower():
                actions.append("Check array/list bounds before accessing")
            elif "key" in failure.message.lower():
                actions.append("Check if key exists before accessing")
        
        return actions
    
    def get_auto_fix_command(self, failure_class: FailureClass, project_type: str) -> str | None:
        """
        Get an auto-fix command for a failure class.
        
        Args:
            failure_class: The failure class
            project_type: The project type (python, node, etc.)
            
        Returns:
            Auto-fix command string or None
        """
        if failure_class == FailureClass.LINT:
            if project_type == "python":
                return "ruff check --fix ."
            elif project_type in ("node", "typescript"):
                return "npx eslint --fix ."
        
        elif failure_class == FailureClass.IMPORT:
            if project_type == "python":
                return "ruff check --fix --select I ."  # Fix import sorting
        
        return None


def classify_failure(failure: FailureInfo) -> FailureClass:
    """
    Convenience function to classify a failure.
    
    Args:
        failure: The failure information
        
    Returns:
        FailureClass enum value
    """
    classifier = FailureClassifier()
    return classifier.classify(failure)


def get_repair_strategy(failure: FailureInfo) -> RepairPlan:
    """
    Convenience function to get a repair strategy.
    
    Args:
        failure: The failure information
        
    Returns:
        RepairPlan with strategy and actions
    """
    classifier = FailureClassifier()
    return classifier.get_repair_plan(failure)
