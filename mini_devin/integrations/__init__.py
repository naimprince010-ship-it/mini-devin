"""
Integrations package for Plodder
Provides integration with external services like GitHub, Vercel, Railway, etc.
"""

from .github import GitHubIntegration

try:
    from .deployment import DeploymentManager
except Exception:  # pragma: no cover - optional dependency surface
    DeploymentManager = None  # type: ignore[assignment]

try:
    from .playwright_agent import PlaywrightAgent
except Exception:  # pragma: no cover - optional dependency surface
    PlaywrightAgent = None  # type: ignore[assignment]

try:
    from .persistent_memory import PersistentMemory
except Exception:  # pragma: no cover - optional dependency surface
    PersistentMemory = None  # type: ignore[assignment]

try:
    from .test_fix_loop import TestFixRerunLoop
except Exception:  # pragma: no cover - optional dependency surface
    TestFixRerunLoop = None  # type: ignore[assignment]

__all__ = [
    "GitHubIntegration",
    "DeploymentManager", 
    "PlaywrightAgent",
    "PersistentMemory",
    "TestFixRerunLoop"
]
