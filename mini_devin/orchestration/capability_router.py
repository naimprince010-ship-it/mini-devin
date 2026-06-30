"""Specialist role router for team-style sub-agent orchestration.

This module classifies a sub-task into a best-fit specialist role and provides
role-specific execution guidance text for worker prompts.
"""

from __future__ import annotations

from enum import Enum

from .task_scheduler import SchedulableUnit


class SpecialistRole(str, Enum):
    FRONTEND = "frontend"
    BACKEND = "backend"
    QA = "qa"
    DEVOPS = "devops"
    GENERALIST = "generalist"


_ROLE_HINTS: dict[SpecialistRole, tuple[str, ...]] = {
    SpecialistRole.FRONTEND: (
        "react",
        "vite",
        "next.js",
        "nextjs",
        "ui",
        "ux",
        "css",
        "component",
        "tailwind",
        "frontend",
        "browser",
        "typescript",
        "tsx",
        "visual",
    ),
    SpecialistRole.BACKEND: (
        "api",
        "backend",
        "fastapi",
        "database",
        "postgres",
        "sql",
        "orm",
        "session",
        "auth",
        "endpoint",
        "python",
        "llm",
        "orchestrator",
        "tool",
    ),
    SpecialistRole.QA: (
        "test",
        "pytest",
        "qa",
        "validation",
        "verify",
        "assert",
        "regression",
        "smoke",
        "coverage",
        "e2e",
    ),
    SpecialistRole.DEVOPS: (
        "deploy",
        "docker",
        "compose",
        "kubernetes",
        "ci",
        "cd",
        "infra",
        "nginx",
        "systemd",
        "monitor",
        "observability",
        "logging",
        "production",
    ),
}


def route_specialist(unit: SchedulableUnit) -> SpecialistRole:
    """Choose a specialist role for a sub-task from text hints."""
    text = (unit.goal + "\n" + "\n".join(unit.acceptance_criteria)).lower()
    best = SpecialistRole.GENERALIST
    best_score = 0
    for role, hints in _ROLE_HINTS.items():
        score = sum(1 for h in hints if h in text)
        if score > best_score:
            best = role
            best_score = score
    return best


def specialist_prompt(role: SpecialistRole) -> str:
    """Prompt guidance for each specialist role."""
    if role == SpecialistRole.FRONTEND:
        return (
            "You are FRONTEND specialist. Prioritize UI correctness, responsive behavior, "
            "TypeScript safety, accessibility, and visual regressions."
        )
    if role == SpecialistRole.BACKEND:
        return (
            "You are BACKEND specialist. Prioritize API correctness, data integrity, "
            "error handling, concurrency safety, and migration-safe changes."
        )
    if role == SpecialistRole.QA:
        return (
            "You are QA specialist. Prioritize reproducible test cases, minimal flaky behavior, "
            "clear assertions, and evidence-driven verification."
        )
    if role == SpecialistRole.DEVOPS:
        return (
            "You are DEVOPS specialist. Prioritize deploy safety, rollbackability, service health, "
            "logs/metrics visibility, and runtime resilience."
        )
    return (
        "You are GENERALIST specialist. Make pragmatic, minimal-risk changes and verify outcomes "
        "before completion."
    )
