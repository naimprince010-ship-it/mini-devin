"""
Skills Library for Mini-Devin

This module provides reusable procedures (skills) that the agent can invoke
to perform common development tasks consistently and reliably.

Skills are high-level, composable actions that combine multiple tool calls
into coherent workflows. Examples include:
- Adding a new API endpoint with tests
- Fixing a failing test
- Refactoring code with proper test updates
- Setting up project configurations
"""

from .base import Skill, SkillContext, SkillResult, SkillStatus
from .registry import SkillRegistry, get_registry, register_skill
from .builtin import (
    AddEndpointSkill,
    FixFailingTestSkill,
    RefactorFunctionSkill,
    AddDependencySkill,
    CreateTestSkill,
    UpdateDocumentationSkill,
    SetupLintingSkill,
    DebugErrorSkill,
)

__all__ = [
    # Base classes
    "Skill",
    "SkillContext",
    "SkillResult",
    "SkillStatus",
    # Registry
    "SkillRegistry",
    "get_registry",
    "register_skill",
    # Built-in skills
    "AddEndpointSkill",
    "FixFailingTestSkill",
    "RefactorFunctionSkill",
    "AddDependencySkill",
    "CreateTestSkill",
    "UpdateDocumentationSkill",
    "SetupLintingSkill",
    "DebugErrorSkill",
]
