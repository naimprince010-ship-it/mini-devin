"""
Base classes for Browser Tools

This module provides base classes and utilities shared across browser tools.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any = None
    message: str = ""
    error: str | None = None


@dataclass
class BaseBrowserTool:
    """Base class for browser tools."""
    name: str = ""
    description: str = ""
    
    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description
