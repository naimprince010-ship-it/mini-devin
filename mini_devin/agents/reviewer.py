"""
Reviewer Agent for Mini-Devin

This module implements a reviewer agent that critiques diffs and proposes
smaller, more focused patches. It helps improve diff discipline and reduce
regressions by providing intelligent feedback on code changes.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.llm_client import LLMClient, create_llm_client
from ..reliability.diff_discipline import (
    DiffAnalyzer,
    DiffMetrics,
    PatchStrategy,
    PatchSuggestion,
)


class ReviewSeverity(str, Enum):
    """Severity levels for review feedback."""
    
    CRITICAL = "critical"
    """Must be fixed before merging - security issues, breaking changes."""
    
    HIGH = "high"
    """Should be fixed - bugs, significant code quality issues."""
    
    MEDIUM = "medium"
    """Recommended to fix - style issues, minor improvements."""
    
    LOW = "low"
    """Nice to have - suggestions, optional improvements."""
    
    INFO = "info"
    """Informational - observations, no action required."""


class ReviewCategory(str, Enum):
    """Categories for review feedback."""
    
    SECURITY = "security"
    """Security vulnerabilities or concerns."""
    
    BUG = "bug"
    """Potential bugs or logic errors."""
    
    PERFORMANCE = "performance"
    """Performance issues or improvements."""
    
    STYLE = "style"
    """Code style and formatting issues."""
    
    COMPLEXITY = "complexity"
    """Code complexity concerns."""
    
    DIFF_SIZE = "diff_size"
    """Diff size and scope concerns."""
    
    TESTING = "testing"
    """Testing coverage or quality."""
    
    DOCUMENTATION = "documentation"
    """Documentation issues."""
    
    BEST_PRACTICE = "best_practice"
    """Best practice violations."""
    
    SUGGESTION = "suggestion"
    """General suggestions for improvement."""


@dataclass
class ReviewComment:
    """A single review comment on a specific location."""
    
    file_path: str
    line_number: int | None
    message: str
    severity: ReviewSeverity
    category: ReviewCategory
    suggested_fix: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class PatchImprovement:
    """A suggested improvement to make a patch smaller or more focused."""
    
    original_patch: str
    improved_patch: str
    explanation: str
    lines_saved: int
    strategy: PatchStrategy
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_lines": len(self.original_patch.split("\n")),
            "improved_lines": len(self.improved_patch.split("\n")),
            "lines_saved": self.lines_saved,
            "explanation": self.explanation,
            "strategy": self.strategy.value,
        }


@dataclass
class ReviewFeedback:
    """Complete review feedback for a diff."""
    
    approved: bool
    summary: str
    comments: list[ReviewComment] = field(default_factory=list)
    patch_improvements: list[PatchImprovement] = field(default_factory=list)
    diff_metrics: DiffMetrics | None = None
    discipline_violations: list[str] = field(default_factory=list)
    overall_quality_score: float = 0.0
    
    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for c in self.comments if c.severity == ReviewSeverity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        """Count of high severity issues."""
        return sum(1 for c in self.comments if c.severity == ReviewSeverity.HIGH)
    
    @property
    def has_blocking_issues(self) -> bool:
        """Check if there are blocking issues."""
        return self.critical_count > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approved": self.approved,
            "summary": self.summary,
            "comments": [c.to_dict() for c in self.comments],
            "patch_improvements": [p.to_dict() for p in self.patch_improvements],
            "diff_metrics": self.diff_metrics.to_dict() if self.diff_metrics else None,
            "discipline_violations": self.discipline_violations,
            "overall_quality_score": self.overall_quality_score,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "has_blocking_issues": self.has_blocking_issues,
        }
    
    def format_report(self) -> str:
        """Format the review as a human-readable report."""
        lines = []
        
        status = "APPROVED" if self.approved else "CHANGES REQUESTED"
        lines.append(f"## Review Status: {status}")
        lines.append("")
        lines.append(f"**Summary:** {self.summary}")
        lines.append("")
        lines.append(f"**Quality Score:** {self.overall_quality_score:.1f}/10")
        lines.append("")
        
        if self.diff_metrics:
            lines.append("### Diff Metrics")
            lines.append(f"- Files changed: {self.diff_metrics.files_changed}")
            lines.append(f"- Lines added: {self.diff_metrics.total_additions}")
            lines.append(f"- Lines removed: {self.diff_metrics.total_deletions}")
            lines.append(f"- Total hunks: {self.diff_metrics.total_hunks}")
            lines.append("")
        
        if self.discipline_violations:
            lines.append("### Diff Discipline Violations")
            for violation in self.discipline_violations:
                lines.append(f"- {violation}")
            lines.append("")
        
        if self.comments:
            lines.append("### Review Comments")
            for comment in sorted(self.comments, key=lambda c: (
                0 if c.severity == ReviewSeverity.CRITICAL else
                1 if c.severity == ReviewSeverity.HIGH else
                2 if c.severity == ReviewSeverity.MEDIUM else
                3 if c.severity == ReviewSeverity.LOW else 4
            )):
                severity_icon = {
                    ReviewSeverity.CRITICAL: "[CRITICAL]",
                    ReviewSeverity.HIGH: "[HIGH]",
                    ReviewSeverity.MEDIUM: "[MEDIUM]",
                    ReviewSeverity.LOW: "[LOW]",
                    ReviewSeverity.INFO: "[INFO]",
                }[comment.severity]
                
                location = f"{comment.file_path}"
                if comment.line_number:
                    location += f":{comment.line_number}"
                
                lines.append(f"\n**{severity_icon}** `{location}`")
                lines.append(f"  {comment.message}")
                if comment.suggested_fix:
                    lines.append(f"  *Suggested fix:* {comment.suggested_fix}")
            lines.append("")
        
        if self.patch_improvements:
            lines.append("### Suggested Patch Improvements")
            for improvement in self.patch_improvements:
                lines.append(f"\n- {improvement.explanation}")
                lines.append(f"  Lines saved: {improvement.lines_saved}")
            lines.append("")
        
        return "\n".join(lines)


REVIEWER_SYSTEM_PROMPT = """You are a code reviewer agent for Mini-Devin. Your job is to review diffs and provide constructive feedback.

## Your Responsibilities
1. Identify potential bugs, security issues, and logic errors
2. Check for code quality and best practices
3. Suggest ways to make patches smaller and more focused
4. Ensure changes are well-tested and documented
5. Enforce diff discipline (small, focused changes)

## Review Guidelines
- Be constructive and specific
- Prioritize issues by severity
- Suggest concrete fixes when possible
- Consider the context and intent of the change
- Don't nitpick on minor style issues unless they affect readability

## Output Format
Provide your review as a JSON object with the following structure:
{
    "approved": boolean,
    "summary": "Brief summary of the review",
    "quality_score": number (0-10),
    "comments": [
        {
            "file_path": "path/to/file",
            "line_number": number or null,
            "message": "Description of the issue",
            "severity": "critical|high|medium|low|info",
            "category": "security|bug|performance|style|complexity|diff_size|testing|documentation|best_practice|suggestion",
            "suggested_fix": "Optional suggested fix"
        }
    ],
    "patch_improvements": [
        {
            "explanation": "How to improve the patch",
            "strategy": "apply_patch|write_file|search_replace|insert_lines|delete_lines"
        }
    ]
}"""


class ReviewerAgent:
    """
    A reviewer agent that critiques diffs and proposes improvements.
    
    The reviewer analyzes code changes and provides feedback on:
    - Code quality and best practices
    - Potential bugs and security issues
    - Diff discipline (size, scope, focus)
    - Suggestions for smaller, more focused patches
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        diff_analyzer: DiffAnalyzer | None = None,
        strict_mode: bool = False,
        auto_suggest_improvements: bool = True,
    ):
        """
        Initialize the reviewer agent.
        
        Args:
            llm_client: LLM client for intelligent review
            diff_analyzer: Analyzer for diff metrics
            strict_mode: If True, be stricter about diff discipline
            auto_suggest_improvements: If True, automatically suggest patch improvements
        """
        self.llm = llm_client or create_llm_client()
        self.diff_analyzer = diff_analyzer or DiffAnalyzer()
        self.strict_mode = strict_mode
        self.auto_suggest_improvements = auto_suggest_improvements
        
        self.llm.set_system_prompt(REVIEWER_SYSTEM_PROMPT)
    
    async def review_diff(
        self,
        diff: str,
        context: str | None = None,
        task_description: str | None = None,
    ) -> ReviewFeedback:
        """
        Review a diff and provide feedback.
        
        Args:
            diff: The unified diff to review
            context: Optional context about the codebase
            task_description: Optional description of what the change is trying to do
            
        Returns:
            ReviewFeedback with the review results
        """
        metrics = self.diff_analyzer.analyze(diff)
        passes_discipline, violations = self.diff_analyzer.check_discipline(diff)
        
        prompt = self._build_review_prompt(diff, context, task_description, metrics, violations)
        
        response = await self.llm.chat(prompt)
        
        feedback = self._parse_review_response(response, metrics, violations)
        
        if self.auto_suggest_improvements and not passes_discipline:
            improvements = self._suggest_improvements(diff, metrics, violations)
            feedback.patch_improvements.extend(improvements)
        
        return feedback
    
    async def review_file_change(
        self,
        original_content: str,
        modified_content: str,
        file_path: str,
        context: str | None = None,
    ) -> ReviewFeedback:
        """
        Review a file change by comparing original and modified content.
        
        Args:
            original_content: Original file content
            modified_content: Modified file content
            file_path: Path to the file
            context: Optional context about the change
            
        Returns:
            ReviewFeedback with the review results
        """
        diff = self.diff_analyzer.create_minimal_patch(
            original_content,
            modified_content,
            file_path,
        )
        
        return await self.review_diff(diff, context)
    
    async def suggest_smaller_patch(
        self,
        diff: str,
        max_lines: int = 50,
    ) -> list[str]:
        """
        Suggest how to break a large diff into smaller patches.
        
        Args:
            diff: The large diff to split
            max_lines: Maximum lines per suggested patch
            
        Returns:
            List of suggested smaller diffs
        """
        metrics = self.diff_analyzer.analyze(diff)
        
        if metrics.total_changes <= max_lines:
            return [diff]
        
        file_diffs = self.diff_analyzer.split_large_diff(diff)
        
        if len(file_diffs) > 1:
            return file_diffs
        
        prompt = f"""The following diff is too large ({metrics.total_changes} lines). 
Please suggest how to break it into smaller, logical chunks of at most {max_lines} lines each.

Diff:
```
{diff}
```

Provide your suggestions as a JSON array of objects, each with:
- "description": What this chunk does
- "files": Which files/sections to include
- "estimated_lines": Approximate size
"""
        
        # Call LLM for suggestions (response used for future enhancement)
        _ = await self.llm.chat(prompt)
        
        return file_diffs
    
    async def critique_patch_strategy(
        self,
        change_description: str,
        proposed_strategy: PatchStrategy,
        file_path: str | None = None,
    ) -> PatchSuggestion:
        """
        Critique a proposed patch strategy and suggest alternatives.
        
        Args:
            change_description: Description of the intended change
            proposed_strategy: The strategy being considered
            file_path: Target file path
            
        Returns:
            PatchSuggestion with recommendation
        """
        suggestion = self.diff_analyzer.suggest_strategy(
            change_description,
            file_path,
        )
        
        if suggestion.strategy != proposed_strategy:
            suggestion.warnings.append(
                f"Consider using {suggestion.strategy.value} instead of {proposed_strategy.value}: "
                f"{suggestion.reason}"
            )
        
        return suggestion
    
    def quick_review(self, diff: str) -> tuple[bool, list[str]]:
        """
        Perform a quick, synchronous review without LLM.
        
        This checks diff discipline and basic patterns only.
        
        Args:
            diff: The diff to review
            
        Returns:
            Tuple of (approved, list of issues)
        """
        issues = []
        
        # Analyze diff (metrics available for future use in extended checks)
        _ = self.diff_analyzer.analyze(diff)
        passes_discipline, violations = self.diff_analyzer.check_discipline(diff)
        
        issues.extend(violations)
        
        patterns = self._check_dangerous_patterns(diff)
        issues.extend(patterns)
        
        approved = len(issues) == 0 or (not self.strict_mode and all(
            "Consider" in issue or "Warning" in issue for issue in issues
        ))
        
        return approved, issues
    
    def _build_review_prompt(
        self,
        diff: str,
        context: str | None,
        task_description: str | None,
        metrics: DiffMetrics,
        violations: list[str],
    ) -> str:
        """Build the prompt for LLM review."""
        parts = []
        
        parts.append("Please review the following diff:")
        parts.append("")
        
        if task_description:
            parts.append(f"**Task:** {task_description}")
            parts.append("")
        
        if context:
            parts.append(f"**Context:** {context}")
            parts.append("")
        
        parts.append("**Diff Metrics:**")
        parts.append(f"- Files changed: {metrics.files_changed}")
        parts.append(f"- Lines added: {metrics.total_additions}")
        parts.append(f"- Lines removed: {metrics.total_deletions}")
        parts.append(f"- Total hunks: {metrics.total_hunks}")
        parts.append("")
        
        if violations:
            parts.append("**Diff Discipline Violations:**")
            for v in violations:
                parts.append(f"- {v}")
            parts.append("")
        
        parts.append("**Diff:**")
        parts.append("```diff")
        parts.append(diff[:10000])
        if len(diff) > 10000:
            parts.append("... (truncated)")
        parts.append("```")
        parts.append("")
        parts.append("Provide your review as JSON.")
        
        return "\n".join(parts)
    
    def _parse_review_response(
        self,
        response: str,
        metrics: DiffMetrics,
        violations: list[str],
    ) -> ReviewFeedback:
        """Parse the LLM response into ReviewFeedback."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}
        
        comments = []
        for c in data.get("comments", []):
            try:
                comments.append(ReviewComment(
                    file_path=c.get("file_path", "unknown"),
                    line_number=c.get("line_number"),
                    message=c.get("message", ""),
                    severity=ReviewSeverity(c.get("severity", "info")),
                    category=ReviewCategory(c.get("category", "suggestion")),
                    suggested_fix=c.get("suggested_fix"),
                ))
            except (ValueError, KeyError):
                continue
        
        improvements = []
        for p in data.get("patch_improvements", []):
            try:
                improvements.append(PatchImprovement(
                    original_patch="",
                    improved_patch="",
                    explanation=p.get("explanation", ""),
                    lines_saved=0,
                    strategy=PatchStrategy(p.get("strategy", "apply_patch")),
                ))
            except (ValueError, KeyError):
                continue
        
        return ReviewFeedback(
            approved=data.get("approved", len(comments) == 0),
            summary=data.get("summary", "Review completed"),
            comments=comments,
            patch_improvements=improvements,
            diff_metrics=metrics,
            discipline_violations=violations,
            overall_quality_score=data.get("quality_score", 5.0),
        )
    
    def _suggest_improvements(
        self,
        diff: str,
        metrics: DiffMetrics,
        violations: list[str],
    ) -> list[PatchImprovement]:
        """Suggest improvements based on diff analysis."""
        improvements = []
        
        if metrics.total_changes > self.diff_analyzer.max_lines:
            file_diffs = self.diff_analyzer.split_large_diff(diff)
            if len(file_diffs) > 1:
                improvements.append(PatchImprovement(
                    original_patch=diff,
                    improved_patch=file_diffs[0],
                    explanation=f"Split into {len(file_diffs)} separate patches, one per file",
                    lines_saved=metrics.total_changes - len(file_diffs[0].split("\n")),
                    strategy=PatchStrategy.APPLY_PATCH,
                ))
        
        if metrics.files_changed > self.diff_analyzer.max_files:
            improvements.append(PatchImprovement(
                original_patch=diff,
                improved_patch="",
                explanation=f"Consider splitting changes across {metrics.files_changed} files into separate commits",
                lines_saved=0,
                strategy=PatchStrategy.APPLY_PATCH,
            ))
        
        return improvements
    
    def _check_dangerous_patterns(self, diff: str) -> list[str]:
        """Check for dangerous patterns in the diff."""
        issues = []
        
        dangerous_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'eval\s*\(', "Use of eval() detected - potential security risk"),
            (r'exec\s*\(', "Use of exec() detected - potential security risk"),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell=True in subprocess - potential injection risk"),
            (r'os\.system\s*\(', "Use of os.system() - consider subprocess instead"),
            (r'# ?TODO', "TODO comment found - ensure it's intentional"),
            (r'# ?FIXME', "FIXME comment found - should this be addressed?"),
            (r'# ?HACK', "HACK comment found - consider refactoring"),
            (r'print\s*\(', "Print statement found - consider using logging"),
            (r'console\.log\s*\(', "Console.log found - consider removing for production"),
            (r'debugger;?', "Debugger statement found - remove before merging"),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, diff, re.IGNORECASE):
                issues.append(f"Warning: {message}")
        
        return issues


def create_reviewer_agent(
    strict_mode: bool = False,
    auto_suggest_improvements: bool = True,
) -> ReviewerAgent:
    """
    Create a reviewer agent with default configuration.
    
    Args:
        strict_mode: If True, be stricter about diff discipline
        auto_suggest_improvements: If True, automatically suggest patch improvements
        
    Returns:
        Configured ReviewerAgent instance
    """
    return ReviewerAgent(
        strict_mode=strict_mode,
        auto_suggest_improvements=auto_suggest_improvements,
    )
