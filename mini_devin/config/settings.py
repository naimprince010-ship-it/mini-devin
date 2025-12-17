"""
Settings Configuration for Mini-Devin

This module provides centralized settings management with:
- Environment variable loading
- Default values
- Validation
"""

import os
from dataclasses import dataclass, field
from typing import Any

from .run_modes import RunMode, get_run_mode_config, RunModeConfig


@dataclass
class SafetySettings:
    """Safety-related settings."""
    
    max_iterations: int = 50
    max_repair_iterations: int = 3
    allow_dependency_bump: bool = False
    max_lines_edit: int = 300
    max_files_delete: int = 1
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf ~/*",
        ":(){ :|:& };:",  # Fork bomb
        "mkfs",
        "dd if=/dev/zero",
        "chmod -R 777 /",
        "> /dev/sda",
        "wget | sh",
        "curl | sh",
        "git push --force",
        "git push -f",
    ])
    
    @classmethod
    def from_env(cls) -> "SafetySettings":
        """Load safety settings from environment variables."""
        return cls(
            max_iterations=int(os.environ.get("MAX_ITERATIONS", "50")),
            max_repair_iterations=int(os.environ.get("MAX_REPAIR_ITERATIONS", "3")),
            allow_dependency_bump=os.environ.get("ALLOW_DEPENDENCY_BUMP", "false").lower() == "true",
            max_lines_edit=int(os.environ.get("MAX_LINES_EDIT", "300")),
            max_files_delete=int(os.environ.get("MAX_FILES_DELETE", "1")),
        )


@dataclass
class LLMSettings:
    """LLM-related settings."""
    
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    
    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Load LLM settings from environment variables."""
        return cls(
            model=os.environ.get("LLM_MODEL", "gpt-4o"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    
    def has_valid_api_key(self) -> bool:
        """Check if at least one API key is configured."""
        return bool(self.openai_api_key or self.anthropic_api_key)


@dataclass
class BrowserSettings:
    """Browser-related settings."""
    
    tavily_api_key: str | None = None
    serpapi_api_key: str | None = None
    selenium_url: str | None = None
    headless: bool = True
    page_load_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "BrowserSettings":
        """Load browser settings from environment variables."""
        return cls(
            tavily_api_key=os.environ.get("TAVILY_API_KEY"),
            serpapi_api_key=os.environ.get("SERPAPI_API_KEY"),
            selenium_url=os.environ.get("SELENIUM_URL", "http://selenium:4444"),
            headless=os.environ.get("BROWSER_HEADLESS", "true").lower() == "true",
            page_load_timeout=int(os.environ.get("PAGE_LOAD_TIMEOUT", "30")),
        )
    
    def has_search_api_key(self) -> bool:
        """Check if at least one search API key is configured."""
        return bool(self.tavily_api_key or self.serpapi_api_key)


@dataclass
class ArtifactSettings:
    """Artifact logging settings."""
    
    artifact_dir: str = "./runs"
    verbose: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "ArtifactSettings":
        """Load artifact settings from environment variables."""
        return cls(
            artifact_dir=os.environ.get("ARTIFACT_DIR", "./runs"),
            verbose=os.environ.get("VERBOSE", "true").lower() == "true",
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


@dataclass
class E2ESettings:
    """End-to-End testing settings."""
    
    e2e_required: bool = False
    e2e_timeout: int = 300
    e2e_report_dir: str = "./tests/e2e"
    
    @classmethod
    def from_env(cls) -> "E2ESettings":
        """Load E2E settings from environment variables."""
        return cls(
            e2e_required=os.environ.get("E2E_REQUIRED", "false").lower() == "true",
            e2e_timeout=int(os.environ.get("E2E_TIMEOUT", "300")),
            e2e_report_dir=os.environ.get("E2E_REPORT_DIR", "./tests/e2e"),
        )


@dataclass
class Settings:
    """
    Centralized settings for Mini-Devin.
    
    Combines all setting categories and provides validation.
    """
    
    run_mode: RunMode = RunMode.OFFLINE
    run_mode_config: RunModeConfig | None = None
    safety: SafetySettings = field(default_factory=SafetySettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    browser: BrowserSettings = field(default_factory=BrowserSettings)
    artifacts: ArtifactSettings = field(default_factory=ArtifactSettings)
    e2e: E2ESettings = field(default_factory=E2ESettings)
    workspace_dir: str = "/workspace"
    
    def __post_init__(self):
        """Initialize run mode config if not provided."""
        if self.run_mode_config is None:
            self.run_mode_config = get_run_mode_config(self.run_mode)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load all settings from environment variables."""
        run_mode_str = os.environ.get("RUN_MODE", "offline")
        try:
            run_mode = RunMode(run_mode_str.lower())
        except ValueError:
            run_mode = RunMode.OFFLINE
        
        return cls(
            run_mode=run_mode,
            run_mode_config=get_run_mode_config(run_mode),
            safety=SafetySettings.from_env(),
            llm=LLMSettings.from_env(),
            browser=BrowserSettings.from_env(),
            artifacts=ArtifactSettings.from_env(),
            e2e=E2ESettings.from_env(),
            workspace_dir=os.environ.get("WORKSPACE_DIR", "/workspace"),
        )
    
    def validate(self) -> list[str]:
        """
        Validate settings and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check LLM API key
        if not self.llm.has_valid_api_key():
            errors.append("No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        
        # Check browser API keys for browse/interactive modes
        if self.run_mode in (RunMode.BROWSE, RunMode.INTERACTIVE):
            if not self.browser.has_search_api_key():
                errors.append(
                    f"Run mode '{self.run_mode.value}' requires a search API key. "
                    "Set TAVILY_API_KEY or SERPAPI_API_KEY, or use 'offline' mode."
                )
        
        # Check safety settings
        if self.safety.max_iterations < 1:
            errors.append("MAX_ITERATIONS must be at least 1.")
        
        if self.safety.max_repair_iterations < 0:
            errors.append("MAX_REPAIR_ITERATIONS must be non-negative.")
        
        if self.safety.max_lines_edit < 1:
            errors.append("MAX_LINES_EDIT must be at least 1.")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if settings are valid."""
        return len(self.validate()) == 0
    
    def get_enabled_tools(self) -> list[str]:
        """Get list of enabled tools based on run mode."""
        if self.run_mode_config:
            return self.run_mode_config.get_enabled_tools()
        return ["terminal", "editor"]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary for logging."""
        return {
            "run_mode": self.run_mode.value,
            "enabled_tools": self.get_enabled_tools(),
            "safety": {
                "max_iterations": self.safety.max_iterations,
                "max_repair_iterations": self.safety.max_repair_iterations,
                "allow_dependency_bump": self.safety.allow_dependency_bump,
                "max_lines_edit": self.safety.max_lines_edit,
                "max_files_delete": self.safety.max_files_delete,
            },
            "llm": {
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "has_api_key": self.llm.has_valid_api_key(),
            },
            "browser": {
                "has_search_api_key": self.browser.has_search_api_key(),
                "headless": self.browser.headless,
            },
            "artifacts": {
                "artifact_dir": self.artifacts.artifact_dir,
                "verbose": self.artifacts.verbose,
                "log_level": self.artifacts.log_level,
            },
            "e2e": {
                "e2e_required": self.e2e.e2e_required,
                "e2e_timeout": self.e2e.e2e_timeout,
                "e2e_report_dir": self.e2e.e2e_report_dir,
            },
            "workspace_dir": self.workspace_dir,
        }


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Settings are loaded from environment variables on first access.
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global _settings
    _settings = Settings.from_env()
    return _settings
