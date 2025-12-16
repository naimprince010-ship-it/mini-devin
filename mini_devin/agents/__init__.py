"""
Multi-Agent Module for Mini-Devin

This module provides specialized agents that work together:
- ReviewerAgent: Critiques diffs and proposes improvements
- More agents can be added in the future
"""

from .reviewer import (
    ReviewerAgent,
    ReviewFeedback,
    ReviewSeverity,
    ReviewCategory,
    create_reviewer_agent,
)

__all__ = [
    "ReviewerAgent",
    "ReviewFeedback",
    "ReviewSeverity",
    "ReviewCategory",
    "create_reviewer_agent",
]
