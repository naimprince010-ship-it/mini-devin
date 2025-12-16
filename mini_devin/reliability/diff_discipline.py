"""
Diff Discipline Module for Mini-Devin

This module enforces smaller, focused patches:
- Analyze diff size and complexity
- Prefer apply_patch over full file rewrites
- Suggest breaking large changes into smaller patches
- Track diff metrics for reliability improvement
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PatchStrategy(str, Enum):
    """Strategies for applying code changes."""
    
    APPLY_PATCH = "apply_patch"
    """Use unified diff patch - preferred for small, focused changes."""
    
    WRITE_FILE = "write_file"
    """Write entire file - only for new files or complete rewrites."""
    
    SEARCH_REPLACE = "search_replace"
    """Search and replace - for simple text substitutions."""
    
    INSERT_LINES = "insert_lines"
    """Insert lines at specific location - for adding new code."""
    
    DELETE_LINES = "delete_lines"
    """Delete specific lines - for removing code."""


@dataclass
class DiffHunk:
    """A single hunk in a diff."""
    
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines_added: int
    lines_removed: int
    content: str
    
    @property
    def total_changes(self) -> int:
        """Total number of changed lines."""
        return self.lines_added + self.lines_removed
    
    @property
    def is_pure_addition(self) -> bool:
        """Check if this hunk only adds lines."""
        return self.lines_removed == 0 and self.lines_added > 0
    
    @property
    def is_pure_deletion(self) -> bool:
        """Check if this hunk only removes lines."""
        return self.lines_added == 0 and self.lines_removed > 0


@dataclass
class DiffMetrics:
    """
    Metrics for analyzing a diff.
    
    Used to determine if a change is too large or complex.
    """
    
    files_changed: int = 0
    total_additions: int = 0
    total_deletions: int = 0
    total_hunks: int = 0
    max_hunk_size: int = 0
    avg_hunk_size: float = 0.0
    files: list[str] = field(default_factory=list)
    hunks_per_file: dict[str, int] = field(default_factory=dict)
    
    @property
    def total_changes(self) -> int:
        """Total number of changed lines."""
        return self.total_additions + self.total_deletions
    
    @property
    def is_large_change(self) -> bool:
        """Check if this is a large change (>100 lines)."""
        return self.total_changes > 100
    
    @property
    def is_complex_change(self) -> bool:
        """Check if this is a complex change (many files or hunks)."""
        return self.files_changed > 3 or self.total_hunks > 10
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "files_changed": self.files_changed,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "total_changes": self.total_changes,
            "total_hunks": self.total_hunks,
            "max_hunk_size": self.max_hunk_size,
            "avg_hunk_size": self.avg_hunk_size,
            "is_large_change": self.is_large_change,
            "is_complex_change": self.is_complex_change,
        }


@dataclass
class PatchSuggestion:
    """Suggestion for how to apply a change."""
    
    strategy: PatchStrategy
    reason: str
    confidence: float  # 0.0 to 1.0
    warnings: list[str] = field(default_factory=list)
    alternative_strategies: list[PatchStrategy] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "strategy": self.strategy.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "alternative_strategies": [s.value for s in self.alternative_strategies],
        }


class DiffAnalyzer:
    """
    Analyzes diffs to enforce diff discipline.
    
    Provides metrics and suggestions for keeping changes small and focused.
    """
    
    # Thresholds for diff discipline
    MAX_LINES_PER_CHANGE = 300
    MAX_LINES_PER_HUNK = 50
    MAX_FILES_PER_CHANGE = 5
    MAX_HUNKS_PER_FILE = 5
    
    # Patterns for parsing unified diffs
    FILE_HEADER_PATTERN = re.compile(r"^diff --git a/(.+) b/(.+)$")
    HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    
    def __init__(
        self,
        max_lines: int = MAX_LINES_PER_CHANGE,
        max_hunk_size: int = MAX_LINES_PER_HUNK,
        max_files: int = MAX_FILES_PER_CHANGE,
    ):
        self.max_lines = max_lines
        self.max_hunk_size = max_hunk_size
        self.max_files = max_files
    
    def analyze(self, diff: str) -> DiffMetrics:
        """
        Analyze a unified diff and return metrics.
        
        Args:
            diff: Unified diff string
            
        Returns:
            DiffMetrics with analysis results
        """
        metrics = DiffMetrics()
        lines = diff.split("\n")
        
        current_file = None
        current_hunk_size = 0
        hunk_sizes = []
        
        for line in lines:
            # Check for file header
            match = self.FILE_HEADER_PATTERN.match(line)
            if match:
                current_file = match.group(2)
                metrics.files.append(current_file)
                metrics.files_changed += 1
                metrics.hunks_per_file[current_file] = 0
                continue
            
            # Check for hunk header
            match = self.HUNK_HEADER_PATTERN.match(line)
            if match:
                if current_hunk_size > 0:
                    hunk_sizes.append(current_hunk_size)
                current_hunk_size = 0
                metrics.total_hunks += 1
                if current_file:
                    metrics.hunks_per_file[current_file] = metrics.hunks_per_file.get(current_file, 0) + 1
                continue
            
            # Count additions and deletions
            if line.startswith("+") and not line.startswith("+++"):
                metrics.total_additions += 1
                current_hunk_size += 1
            elif line.startswith("-") and not line.startswith("---"):
                metrics.total_deletions += 1
                current_hunk_size += 1
        
        # Add last hunk
        if current_hunk_size > 0:
            hunk_sizes.append(current_hunk_size)
        
        # Calculate hunk statistics
        if hunk_sizes:
            metrics.max_hunk_size = max(hunk_sizes)
            metrics.avg_hunk_size = sum(hunk_sizes) / len(hunk_sizes)
        
        return metrics
    
    def check_discipline(self, diff: str) -> tuple[bool, list[str]]:
        """
        Check if a diff follows diff discipline rules.
        
        Args:
            diff: Unified diff string
            
        Returns:
            Tuple of (passes_discipline, list of violations)
        """
        metrics = self.analyze(diff)
        violations = []
        
        if metrics.total_changes > self.max_lines:
            violations.append(
                f"Total changes ({metrics.total_changes} lines) exceeds maximum ({self.max_lines} lines). "
                "Consider breaking into smaller changes."
            )
        
        if metrics.max_hunk_size > self.max_hunk_size:
            violations.append(
                f"Largest hunk ({metrics.max_hunk_size} lines) exceeds maximum ({self.max_hunk_size} lines). "
                "Consider more focused changes."
            )
        
        if metrics.files_changed > self.max_files:
            violations.append(
                f"Files changed ({metrics.files_changed}) exceeds maximum ({self.max_files}). "
                "Consider changing fewer files at once."
            )
        
        for file, hunks in metrics.hunks_per_file.items():
            if hunks > self.MAX_HUNKS_PER_FILE:
                violations.append(
                    f"File '{file}' has {hunks} hunks (max {self.MAX_HUNKS_PER_FILE}). "
                    "Consider more focused changes to this file."
                )
        
        return len(violations) == 0, violations
    
    def suggest_strategy(
        self,
        change_description: str,
        target_file: str | None = None,
        estimated_lines: int | None = None,
        is_new_file: bool = False,
    ) -> PatchSuggestion:
        """
        Suggest the best strategy for applying a change.
        
        Args:
            change_description: Description of the intended change
            target_file: Target file path (if known)
            estimated_lines: Estimated number of lines to change
            is_new_file: Whether this is a new file
            
        Returns:
            PatchSuggestion with recommended strategy
        """
        warnings = []
        
        # New files should use write_file
        if is_new_file:
            return PatchSuggestion(
                strategy=PatchStrategy.WRITE_FILE,
                reason="New file creation requires write_file",
                confidence=1.0,
            )
        
        # Large changes should be broken down
        if estimated_lines and estimated_lines > self.max_lines:
            warnings.append(
                f"Estimated change size ({estimated_lines} lines) is large. "
                "Consider breaking into smaller changes."
            )
        
        # Analyze change description for hints
        description_lower = change_description.lower()
        
        # Simple text substitutions
        if any(word in description_lower for word in ["rename", "replace", "change name"]):
            return PatchSuggestion(
                strategy=PatchStrategy.SEARCH_REPLACE,
                reason="Simple text substitution detected",
                confidence=0.8,
                warnings=warnings,
                alternative_strategies=[PatchStrategy.APPLY_PATCH],
            )
        
        # Adding new code
        if any(word in description_lower for word in ["add", "insert", "create", "implement"]):
            if estimated_lines and estimated_lines < 20:
                return PatchSuggestion(
                    strategy=PatchStrategy.INSERT_LINES,
                    reason="Small addition detected",
                    confidence=0.7,
                    warnings=warnings,
                    alternative_strategies=[PatchStrategy.APPLY_PATCH],
                )
        
        # Removing code
        if any(word in description_lower for word in ["remove", "delete", "drop"]):
            return PatchSuggestion(
                strategy=PatchStrategy.DELETE_LINES,
                reason="Code removal detected",
                confidence=0.7,
                warnings=warnings,
                alternative_strategies=[PatchStrategy.APPLY_PATCH],
            )
        
        # Default to apply_patch for most changes
        return PatchSuggestion(
            strategy=PatchStrategy.APPLY_PATCH,
            reason="Unified diff patch is the safest and most precise method",
            confidence=0.9,
            warnings=warnings,
            alternative_strategies=[PatchStrategy.WRITE_FILE],
        )
    
    def split_large_diff(self, diff: str) -> list[str]:
        """
        Split a large diff into smaller, focused diffs.
        
        Args:
            diff: Large unified diff string
            
        Returns:
            List of smaller diff strings
        """
        # Split by file
        file_diffs = []
        current_diff_lines = []
        
        for line in diff.split("\n"):
            if self.FILE_HEADER_PATTERN.match(line):
                if current_diff_lines:
                    file_diffs.append("\n".join(current_diff_lines))
                current_diff_lines = [line]
            else:
                current_diff_lines.append(line)
        
        if current_diff_lines:
            file_diffs.append("\n".join(current_diff_lines))
        
        return file_diffs
    
    def create_minimal_patch(
        self,
        original: str,
        modified: str,
        file_path: str,
        context_lines: int = 3,
    ) -> str:
        """
        Create a minimal unified diff patch.
        
        Args:
            original: Original file content
            modified: Modified file content
            file_path: Path to the file
            context_lines: Number of context lines to include
            
        Returns:
            Unified diff string
        """
        import difflib
        
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
        )
        
        return "".join(diff)


def analyze_diff(diff: str) -> DiffMetrics:
    """
    Convenience function to analyze a diff.
    
    Args:
        diff: Unified diff string
        
    Returns:
        DiffMetrics with analysis results
    """
    analyzer = DiffAnalyzer()
    return analyzer.analyze(diff)


def suggest_patch_strategy(
    change_description: str,
    target_file: str | None = None,
    estimated_lines: int | None = None,
    is_new_file: bool = False,
) -> PatchSuggestion:
    """
    Convenience function to suggest a patch strategy.
    
    Args:
        change_description: Description of the intended change
        target_file: Target file path (if known)
        estimated_lines: Estimated number of lines to change
        is_new_file: Whether this is a new file
        
    Returns:
        PatchSuggestion with recommended strategy
    """
    analyzer = DiffAnalyzer()
    return analyzer.suggest_strategy(
        change_description,
        target_file,
        estimated_lines,
        is_new_file,
    )
