"""
Evaluation Module for Mini-Devin

This module provides SWE-bench style evaluation for:
- Task validation and scoring
- Benchmark suite management
- Result tracking and reporting
- Regression testing
"""

from .harness import (
    EvaluationHarness,
    EvaluationTask,
    EvaluationResult,
    TaskDifficulty,
)
from .benchmark import Benchmark, BenchmarkSuite

__all__ = [
    "EvaluationHarness",
    "EvaluationTask",
    "EvaluationResult",
    "TaskDifficulty",
    "Benchmark",
    "BenchmarkSuite",
]
