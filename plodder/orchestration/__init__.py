"""
Plodder multi-step session orchestration.

Exports are **lazy** (PEP 562) so ``import plodder.orchestration.reasoning_loop`` does not
eagerly load ``self_heal`` → ``session_driver`` (avoids circular imports with ``agent``).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SelfHealBundle",
    "SelfHealLoop",
    "SelfHealResult",
    "UnifiedSessionDriver",
    "UnifiedSessionResult",
    "build_static_fix_prompt",
    "extract_code_fence",
    "merge_static_and_runtime",
    "parse_pseudo_plan_json",
]

_LAZY_MODULE = "plodder.orchestration.self_heal"


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(_LAZY_MODULE)
    return getattr(mod, name)
