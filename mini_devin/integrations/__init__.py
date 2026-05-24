"""
Integrations package for Plodder
Provides integration with external services like GitHub, Vercel, Railway, etc.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "GitHubIntegration": (".github", "GitHubIntegration"),
    "DeploymentManager": (".deployment", "DeploymentManager"),
    "PlaywrightAgent": (".playwright_agent", "PlaywrightAgent"),
    "PersistentMemory": (".persistent_memory", "PersistentMemory"),
    "TestFixRerunLoop": (".test_fix_loop", "TestFixRerunLoop"),
    "render_travel_booking_prompt": (".travel_booking", "render_travel_booking_prompt"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "GitHubIntegration",
    "DeploymentManager", 
    "PlaywrightAgent",
    "PersistentMemory",
    "TestFixRerunLoop",
    "render_travel_booking_prompt",
]
