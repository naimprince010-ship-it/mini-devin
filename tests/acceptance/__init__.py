"""
Acceptance Test Suite for Mini-Devin

This module provides acceptance tests that verify Mini-Devin works
reliably on real-world repositories:
- Python repos
- Node.js repos
- Mixed repos

Test scenarios:
1. Fix failing test
2. Add small feature + test
3. Refactor + update tests
"""

from .test_runner import AcceptanceTestRunner, AcceptanceTestResult
from .scenarios import (
    TestScenario,
    FixFailingTestScenario,
    AddFeatureScenario,
    RefactorScenario,
)

__all__ = [
    "AcceptanceTestRunner",
    "AcceptanceTestResult",
    "TestScenario",
    "FixFailingTestScenario",
    "AddFeatureScenario",
    "RefactorScenario",
]
