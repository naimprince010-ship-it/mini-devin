"""
Run Modes Configuration for Mini-Devin

This module defines the different run modes and their capabilities:
- offline: Terminal + Editor only (no web access)
- browse: Terminal + Editor + Web Search + Fetch (read-only web)
- interactive: All tools including interactive browser
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunMode(str, Enum):
    """Available run modes for Mini-Devin."""
    
    OFFLINE = "offline"
    """Terminal and Editor tools only. No web access."""
    
    BROWSE = "browse"
    """Terminal, Editor, Web Search, and Fetch tools. Read-only web access."""
    
    INTERACTIVE = "interactive"
    """All tools including interactive browser with Selenium."""


@dataclass
class RunModeConfig:
    """Configuration for a specific run mode."""
    
    mode: RunMode
    description: str
    
    # Tool availability
    terminal_enabled: bool = True
    editor_enabled: bool = True
    browser_search_enabled: bool = False
    browser_fetch_enabled: bool = False
    browser_interactive_enabled: bool = False
    
    # Safety settings
    allow_network_access: bool = False
    allow_external_apis: bool = False
    require_selenium: bool = False
    
    # Resource limits
    max_iterations: int = 50
    max_repair_iterations: int = 3
    timeout_seconds: int = 300
    
    # Additional settings
    extra_settings: dict[str, Any] = field(default_factory=dict)
    
    def get_enabled_tools(self) -> list[str]:
        """Get list of enabled tool names."""
        tools = []
        if self.terminal_enabled:
            tools.append("terminal")
        if self.editor_enabled:
            tools.append("editor")
        if self.browser_search_enabled:
            tools.append("browser_search")
        if self.browser_fetch_enabled:
            tools.append("browser_fetch")
        if self.browser_interactive_enabled:
            tools.append("browser_interactive")
        return tools
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        return tool_name in self.get_enabled_tools()


# Predefined run mode configurations
OFFLINE_CONFIG = RunModeConfig(
    mode=RunMode.OFFLINE,
    description="Offline mode: Terminal and Editor only. No web access.",
    terminal_enabled=True,
    editor_enabled=True,
    browser_search_enabled=False,
    browser_fetch_enabled=False,
    browser_interactive_enabled=False,
    allow_network_access=False,
    allow_external_apis=False,
    require_selenium=False,
)

BROWSE_CONFIG = RunModeConfig(
    mode=RunMode.BROWSE,
    description="Browse mode: Terminal, Editor, and read-only web access.",
    terminal_enabled=True,
    editor_enabled=True,
    browser_search_enabled=True,
    browser_fetch_enabled=True,
    browser_interactive_enabled=False,
    allow_network_access=True,
    allow_external_apis=True,
    require_selenium=False,
)

INTERACTIVE_CONFIG = RunModeConfig(
    mode=RunMode.INTERACTIVE,
    description="Interactive mode: All tools including interactive browser.",
    terminal_enabled=True,
    editor_enabled=True,
    browser_search_enabled=True,
    browser_fetch_enabled=True,
    browser_interactive_enabled=True,
    allow_network_access=True,
    allow_external_apis=True,
    require_selenium=True,
)

# Mode configuration mapping
RUN_MODE_CONFIGS: dict[RunMode, RunModeConfig] = {
    RunMode.OFFLINE: OFFLINE_CONFIG,
    RunMode.BROWSE: BROWSE_CONFIG,
    RunMode.INTERACTIVE: INTERACTIVE_CONFIG,
}


def get_run_mode_config(mode: str | RunMode) -> RunModeConfig:
    """
    Get the configuration for a specific run mode.
    
    Args:
        mode: Run mode name or enum value
        
    Returns:
        RunModeConfig for the specified mode
        
    Raises:
        ValueError: If mode is not recognized
    """
    if isinstance(mode, str):
        try:
            mode = RunMode(mode.lower())
        except ValueError:
            valid_modes = [m.value for m in RunMode]
            raise ValueError(
                f"Invalid run mode: '{mode}'. Valid modes: {valid_modes}"
            )
    
    return RUN_MODE_CONFIGS[mode]


def validate_run_mode(mode: str) -> bool:
    """Check if a run mode string is valid."""
    try:
        RunMode(mode.lower())
        return True
    except ValueError:
        return False


def get_available_modes() -> list[str]:
    """Get list of available run mode names."""
    return [m.value for m in RunMode]
