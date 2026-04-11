"""
Mini-Devin: An autonomous AI software engineer agent (terminal, editor, browser tools).

This package root stays **minimal on import**: loading `schemas` + `core` (LiteLLM, etc.)
on every `import mini_devin.*` used to delay or OOM API cold starts (Railway 502).

Import what you need explicitly, for example:
  ``from mini_devin.schemas.state import AgentState, TaskState``
  ``from mini_devin.core.tool_interface import ToolRegistry, get_global_registry``
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
