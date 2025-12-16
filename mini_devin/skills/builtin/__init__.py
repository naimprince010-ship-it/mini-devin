"""
Built-in Skills for Mini-Devin

This module provides a collection of pre-built skills for common
development tasks.
"""

from .add_endpoint import AddEndpointSkill
from .fix_failing_test import FixFailingTestSkill
from .refactor_function import RefactorFunctionSkill
from .add_dependency import AddDependencySkill
from .create_test import CreateTestSkill
from .update_documentation import UpdateDocumentationSkill
from .setup_linting import SetupLintingSkill
from .debug_error import DebugErrorSkill

__all__ = [
    "AddEndpointSkill",
    "FixFailingTestSkill",
    "RefactorFunctionSkill",
    "AddDependencySkill",
    "CreateTestSkill",
    "UpdateDocumentationSkill",
    "SetupLintingSkill",
    "DebugErrorSkill",
]
