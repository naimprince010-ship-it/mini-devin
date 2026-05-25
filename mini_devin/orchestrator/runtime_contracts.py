"""Compatibility shim for runtime-contract imports.

Historically some orchestrator modules imported runtime contracts from
`mini_devin.orchestrator.runtime_contracts`. The implementation now lives in
`mini_devin.orchestration.runtime_contracts`.
"""

from __future__ import annotations

from ..orchestration.runtime_contracts import *  # noqa: F401,F403
