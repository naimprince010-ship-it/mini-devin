"""
LSP-style IDE features (diagnostics, hover) without a full language server process.

Uses Pyright/basedpyright when available; falls back to Python syntax checks.
TypeScript/JavaScript: optional ``tsc`` via npx when Node is available.
"""

from .diagnostics import collect_diagnostics
from .hover import collect_hover

__all__ = ["collect_diagnostics", "collect_hover"]
