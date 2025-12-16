"""Unit tests for the reviewer agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mini_devin.agents.reviewer import (
    ReviewerAgent,
    ReviewFeedback,
    ReviewSeverity,
    ReviewCategory,
    ReviewComment,
    PatchImprovement,
)
from mini_devin.reliability.diff_discipline import PatchStrategy


class TestReviewSeverity:
    """Tests for ReviewSeverity enum."""

    def test_severity_values(self):
        """Test that all severity values exist."""
        assert ReviewSeverity.LOW is not None
        assert ReviewSeverity.MEDIUM is not None
        assert ReviewSeverity.HIGH is not None
        assert ReviewSeverity.CRITICAL is not None
        assert ReviewSeverity.INFO is not None

    def test_severity_ordering(self):
        """Test severity ordering."""
        severities = [
            ReviewSeverity.INFO,
            ReviewSeverity.LOW,
            ReviewSeverity.MEDIUM,
            ReviewSeverity.HIGH,
            ReviewSeverity.CRITICAL,
        ]
        assert len(severities) == 5


class TestReviewCategory:
    """Tests for ReviewCategory enum."""

    def test_category_values(self):
        """Test that all category values exist."""
        assert ReviewCategory.SECURITY is not None
        assert ReviewCategory.BUG is not None
        assert ReviewCategory.PERFORMANCE is not None
        assert ReviewCategory.STYLE is not None
        assert ReviewCategory.COMPLEXITY is not None
        assert ReviewCategory.DIFF_SIZE is not None
        assert ReviewCategory.TESTING is not None
        assert ReviewCategory.DOCUMENTATION is not None
        assert ReviewCategory.BEST_PRACTICE is not None
        assert ReviewCategory.SUGGESTION is not None


class TestReviewComment:
    """Tests for ReviewComment dataclass."""

    def test_comment_creation(self):
        """Test creating a ReviewComment."""
        comment = ReviewComment(
            file_path="src/main.py",
            line_number=42,
            message="Potential SQL injection",
            severity=ReviewSeverity.HIGH,
            category=ReviewCategory.SECURITY,
            suggested_fix="Use parameterized queries",
        )
        assert comment.file_path == "src/main.py"
        assert comment.line_number == 42
        assert comment.severity == ReviewSeverity.HIGH
        assert comment.category == ReviewCategory.SECURITY
        assert comment.suggested_fix == "Use parameterized queries"

    def test_comment_without_line_number(self):
        """Test ReviewComment without line number."""
        comment = ReviewComment(
            file_path="src/utils.py",
            line_number=None,
            message="Missing docstring",
            severity=ReviewSeverity.LOW,
            category=ReviewCategory.DOCUMENTATION,
        )
        assert comment.line_number is None

    def test_comment_to_dict(self):
        """Test converting comment to dict."""
        comment = ReviewComment(
            file_path="test.py",
            line_number=10,
            message="Test message",
            severity=ReviewSeverity.MEDIUM,
            category=ReviewCategory.STYLE,
        )
        d = comment.to_dict()
        assert d["file_path"] == "test.py"
        assert d["line_number"] == 10
        assert d["severity"] == "medium"
        assert d["category"] == "style"


class TestPatchImprovement:
    """Tests for PatchImprovement dataclass."""

    def test_improvement_creation(self):
        """Test creating a PatchImprovement."""
        improvement = PatchImprovement(
            original_patch="large diff",
            improved_patch="smaller diff",
            explanation="Split into smaller patches",
            lines_saved=50,
            strategy=PatchStrategy.APPLY_PATCH,
        )
        assert improvement.lines_saved == 50
        assert improvement.strategy == PatchStrategy.APPLY_PATCH

    def test_improvement_to_dict(self):
        """Test converting improvement to dict."""
        improvement = PatchImprovement(
            original_patch="original",
            improved_patch="improved",
            explanation="Explanation",
            lines_saved=10,
            strategy=PatchStrategy.SEARCH_REPLACE,
        )
        d = improvement.to_dict()
        assert d["lines_saved"] == 10
        assert d["explanation"] == "Explanation"


class TestReviewFeedback:
    """Tests for ReviewFeedback dataclass."""

    def test_feedback_creation(self):
        """Test creating ReviewFeedback."""
        comments = [
            ReviewComment(
                file_path="src/main.py",
                line_number=10,
                message="Issue found",
                severity=ReviewSeverity.MEDIUM,
                category=ReviewCategory.BUG,
            ),
        ]
        feedback = ReviewFeedback(
            approved=False,
            summary="Code needs improvement",
            comments=comments,
        )
        assert feedback.approved is False
        assert len(feedback.comments) == 1
        assert feedback.summary == "Code needs improvement"

    def test_feedback_approved(self):
        """Test approved ReviewFeedback."""
        feedback = ReviewFeedback(
            approved=True,
            summary="Code looks good",
            comments=[],
        )
        assert feedback.approved is True
        assert len(feedback.comments) == 0

    def test_feedback_critical_count(self):
        """Test critical count property."""
        comments = [
            ReviewComment(
                file_path="test.py",
                line_number=1,
                message="Critical issue",
                severity=ReviewSeverity.CRITICAL,
                category=ReviewCategory.SECURITY,
            ),
            ReviewComment(
                file_path="test.py",
                line_number=2,
                message="Minor issue",
                severity=ReviewSeverity.LOW,
                category=ReviewCategory.STYLE,
            ),
        ]
        feedback = ReviewFeedback(
            approved=False,
            summary="Issues found",
            comments=comments,
        )
        assert feedback.critical_count == 1

    def test_feedback_high_count(self):
        """Test high count property."""
        comments = [
            ReviewComment(
                file_path="test.py",
                line_number=1,
                message="High issue 1",
                severity=ReviewSeverity.HIGH,
                category=ReviewCategory.BUG,
            ),
            ReviewComment(
                file_path="test.py",
                line_number=2,
                message="High issue 2",
                severity=ReviewSeverity.HIGH,
                category=ReviewCategory.PERFORMANCE,
            ),
        ]
        feedback = ReviewFeedback(
            approved=False,
            summary="Issues found",
            comments=comments,
        )
        assert feedback.high_count == 2

    def test_feedback_has_blocking_issues(self):
        """Test has_blocking_issues property."""
        # With critical issue
        critical_feedback = ReviewFeedback(
            approved=False,
            summary="Critical issue",
            comments=[
                ReviewComment(
                    file_path="test.py",
                    line_number=1,
                    message="Critical",
                    severity=ReviewSeverity.CRITICAL,
                    category=ReviewCategory.SECURITY,
                ),
            ],
        )
        assert critical_feedback.has_blocking_issues is True
        
        # Without critical issue
        non_critical_feedback = ReviewFeedback(
            approved=True,
            summary="Minor issues",
            comments=[
                ReviewComment(
                    file_path="test.py",
                    line_number=1,
                    message="Minor",
                    severity=ReviewSeverity.LOW,
                    category=ReviewCategory.STYLE,
                ),
            ],
        )
        assert non_critical_feedback.has_blocking_issues is False

    def test_feedback_to_dict(self):
        """Test converting feedback to dict."""
        feedback = ReviewFeedback(
            approved=True,
            summary="All good",
            comments=[],
            overall_quality_score=8.5,
        )
        d = feedback.to_dict()
        assert d["approved"] is True
        assert d["summary"] == "All good"
        assert d["overall_quality_score"] == 8.5

    def test_feedback_format_report(self):
        """Test formatting feedback as report."""
        feedback = ReviewFeedback(
            approved=True,
            summary="Code looks good",
            comments=[],
            overall_quality_score=9.0,
        )
        report = feedback.format_report()
        assert "APPROVED" in report
        assert "Code looks good" in report


class TestReviewerAgent:
    """Tests for ReviewerAgent class."""

    def test_reviewer_initialization(self):
        """Test ReviewerAgent initialization."""
        reviewer = ReviewerAgent()
        assert reviewer is not None
        assert reviewer.strict_mode is False
        assert reviewer.auto_suggest_improvements is True

    def test_reviewer_with_custom_settings(self):
        """Test ReviewerAgent with custom settings."""
        reviewer = ReviewerAgent(
            strict_mode=True,
            auto_suggest_improvements=False,
        )
        assert reviewer.strict_mode is True
        assert reviewer.auto_suggest_improvements is False

    def test_reviewer_with_llm_client(self):
        """Test ReviewerAgent with LLM client."""
        mock_client = MagicMock()
        reviewer = ReviewerAgent(llm_client=mock_client)
        assert reviewer.llm == mock_client

    def test_quick_review_safe_diff(self):
        """Test quick review of a safe diff."""
        reviewer = ReviewerAgent()
        diff = """
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
+# New comment
 def main():
     pass
"""
        approved, issues = reviewer.quick_review(diff)
        # Small diff should generally be approved
        assert isinstance(approved, bool)
        assert isinstance(issues, list)

    def test_quick_review_empty_diff(self):
        """Test quick review of empty diff."""
        reviewer = ReviewerAgent()
        approved, issues = reviewer.quick_review("")
        # Empty diff should be approved
        assert approved is True or len(issues) == 0


class TestReviewerAgentPatterns:
    """Pattern detection tests for ReviewerAgent."""

    def test_reviewer_detects_large_diff(self):
        """Test that reviewer flags large diffs."""
        reviewer = ReviewerAgent(strict_mode=True)
        
        # Create a large diff
        large_diff = "diff --git a/file.py b/file.py\n"
        large_diff += "--- a/file.py\n+++ b/file.py\n"
        for i in range(200):
            large_diff += f"+line {i}\n"
        
        approved, issues = reviewer.quick_review(large_diff)
        # Large diff should have issues in strict mode
        assert len(issues) > 0 or not approved
