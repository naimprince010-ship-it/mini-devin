"""
Multi-Model Provider Support for Plodder (Phase 12).

This module provides support for multiple LLM providers:
- Groq (OpenAI-compatible API, LiteLLM ``groq/...`` models — fast inference)
- Google Gemini (via LiteLLM, e.g. ``gemini/gemini-2.0-flash`` for Google AI Studio)
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic (Claude 3.5, Claude 3)
- Ollama (local models)
- Azure OpenAI
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provider(str, Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE = "azure"


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    name: str
    provider: Provider
    context_window: int
    supports_tools: bool = True
    supports_vision: bool = False
    max_output_tokens: int = 4096
    description: str = ""


@dataclass
class ProviderConfig:
    """Base configuration for a provider."""
    provider: Provider
    api_key: str | None = None
    api_base: str | None = None
    enabled: bool = True
    
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        return self.enabled and self.api_key is not None


@dataclass
class GroqConfig(ProviderConfig):
    """Groq Cloud (OpenAI-compatible REST API; LiteLLM ``groq/<model_id>``)."""

    provider: Provider = field(default=Provider.GROQ, init=False)

    @classmethod
    def from_env(cls) -> "GroqConfig":
        return cls(
            api_key=os.environ.get("GROQ_API_KEY"),
            api_base=os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
            enabled=os.environ.get("GROQ_ENABLED", "true").lower() == "true",
        )

    def is_configured(self) -> bool:
        key = (self.api_key or "").strip()
        if not key or key.upper() == "YOUR_KEY_HERE":
            return False
        return self.enabled


@dataclass
class GoogleAIConfig(ProviderConfig):
    """Google AI Studio / Gemini (LiteLLM ``gemini/...`` models)."""

    provider: Provider = field(default=Provider.GOOGLE, init=False)

    @classmethod
    def from_env(cls) -> "GoogleAIConfig":
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            enabled=os.environ.get("GOOGLE_AI_ENABLED", "true").lower() == "true",
        )

    def is_configured(self) -> bool:
        key = (self.api_key or "").strip()
        if not key or key.upper() == "YOUR_KEY_HERE":
            return False
        return self.enabled


@dataclass
class OpenAIConfig(ProviderConfig):
    """OpenAI-specific configuration."""
    provider: Provider = field(default=Provider.OPENAI, init=False)
    organization: str | None = None
    
    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_API_BASE"),
            organization=os.environ.get("OPENAI_ORGANIZATION"),
            enabled=os.environ.get("OPENAI_ENABLED", "true").lower() == "true",
        )


@dataclass
class AnthropicConfig(ProviderConfig):
    """Anthropic-specific configuration."""
    provider: Provider = field(default=Provider.ANTHROPIC, init=False)
    
    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            api_base=os.environ.get("ANTHROPIC_API_BASE"),
            enabled=os.environ.get("ANTHROPIC_ENABLED", "true").lower() == "true",
        )


@dataclass
class OllamaConfig(ProviderConfig):
    """Ollama-specific configuration for local models."""
    provider: Provider = field(default=Provider.OLLAMA, init=False)
    api_key: str | None = field(default="ollama", init=False)
    
    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Load configuration from environment variables."""
        return cls(
            api_base=os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"),
            # Default off: without this, machines without OPENAI_API_KEY would pick Ollama and
            # fail on localhost:11434 when Ollama is not installed/running.
            enabled=os.environ.get("OLLAMA_ENABLED", "false").lower() == "true",
        )
    
    def is_configured(self) -> bool:
        """Ollama doesn't require an API key."""
        return self.enabled and self.api_base is not None


@dataclass
class AzureConfig(ProviderConfig):
    """Azure OpenAI-specific configuration."""
    provider: Provider = field(default=Provider.AZURE, init=False)
    api_version: str = "2024-02-15-preview"
    deployment_name: str | None = None
    
    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.environ.get("AZURE_API_KEY"),
            api_base=os.environ.get("AZURE_API_BASE"),
            api_version=os.environ.get("AZURE_API_VERSION", "2024-02-15-preview"),
            deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME"),
            enabled=os.environ.get("AZURE_ENABLED", "true").lower() == "true",
        )
    
    def is_configured(self) -> bool:
        """Azure requires API key, base URL, and deployment name."""
        return (
            self.enabled 
            and self.api_key is not None 
            and self.api_base is not None
            and self.deployment_name is not None
        )


GROQ_MODELS = [
    ModelInfo(
        id="llama-3.3-70b-versatile",
        name="Llama 3.3 70B Versatile (Groq)",
        provider=Provider.GROQ,
        context_window=131_072,
        supports_tools=True,
        supports_vision=False,
        max_output_tokens=4096,
        description="Strong default for coding/agents; 4k cap helps Groq free-tier TPM (~12k/min)",
    ),
    ModelInfo(
        id="llama-3.1-8b-instant",
        name="Llama 3.1 8B Instant (Groq)",
        provider=Provider.GROQ,
        context_window=131_072,
        supports_tools=True,
        supports_vision=False,
        max_output_tokens=4096,
        description="Very fast on Groq; good for observation / light passes (replaces llama3-8b-8192)",
    ),
    ModelInfo(
        id="mixtral-8x7b-32768",
        name="Mixtral 8x7B (Groq)",
        provider=Provider.GROQ,
        context_window=32_768,
        supports_tools=True,
        supports_vision=False,
        max_output_tokens=4096,
        description="Solid MoE option on Groq free tier",
    ),
]

OPENAI_MODELS = [
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider=Provider.OPENAI,
        context_window=128000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=16384,
        description="Most capable GPT-4 model with vision support",
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider=Provider.OPENAI,
        context_window=128000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=16384,
        description="Smaller, faster, cheaper GPT-4o variant",
    ),
    ModelInfo(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider=Provider.OPENAI,
        context_window=128000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=4096,
        description="GPT-4 Turbo with vision",
    ),
    ModelInfo(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider=Provider.OPENAI,
        context_window=16385,
        supports_tools=True,
        supports_vision=False,
        max_output_tokens=4096,
        description="Fast and cost-effective model",
    ),
]

ANTHROPIC_MODELS = [
    ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider=Provider.ANTHROPIC,
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=8192,
        description="Most intelligent Claude model",
    ),
    ModelInfo(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        provider=Provider.ANTHROPIC,
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=8192,
        description="Fast and cost-effective Claude model",
    ),
    ModelInfo(
        id="claude-3-opus-20240229",
        name="Claude 3 Opus",
        provider=Provider.ANTHROPIC,
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=4096,
        description="Most powerful Claude 3 model",
    ),
    ModelInfo(
        id="claude-3-sonnet-20240229",
        name="Claude 3 Sonnet",
        provider=Provider.ANTHROPIC,
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=4096,
        description="Balanced Claude 3 model",
    ),
]

OLLAMA_MODELS = [
    ModelInfo(
        id="ollama/llama3.2",
        name="Llama 3.2",
        provider=Provider.OLLAMA,
        context_window=128000,
        supports_tools=True,
        supports_vision=False,
        max_output_tokens=4096,
        description="Meta's Llama 3.2 (local)",
    ),
    ModelInfo(
        id="ollama/codellama",
        name="Code Llama",
        provider=Provider.OLLAMA,
        context_window=16384,
        supports_tools=False,
        supports_vision=False,
        max_output_tokens=4096,
        description="Code-specialized Llama (local)",
    ),
    ModelInfo(
        id="ollama/mistral",
        name="Mistral 7B",
        provider=Provider.OLLAMA,
        context_window=32768,
        supports_tools=True,
        supports_vision=False,
        max_output_tokens=4096,
        description="Mistral 7B (local)",
    ),
    ModelInfo(
        id="ollama/mixtral",
        name="Mixtral 8x7B",
        provider=Provider.OLLAMA,
        context_window=32768,
        supports_tools=True,
        supports_vision=False,
        max_output_tokens=4096,
        description="Mixtral MoE model (local)",
    ),
    ModelInfo(
        id="ollama/deepseek-coder",
        name="DeepSeek Coder",
        provider=Provider.OLLAMA,
        context_window=16384,
        supports_tools=False,
        supports_vision=False,
        max_output_tokens=4096,
        description="DeepSeek Coder (local)",
    ),
]

GEMINI_MODELS = [
    ModelInfo(
        id="gemini/gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider=Provider.GOOGLE,
        context_window=1_000_000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=8192,
        description="Fast Gemini 2.0 (Google AI Studio; LiteLLM ``gemini/`` route)",
    ),
]

AZURE_MODELS = [
    ModelInfo(
        id="azure/gpt-4o",
        name="Azure GPT-4o",
        provider=Provider.AZURE,
        context_window=128000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=16384,
        description="GPT-4o on Azure OpenAI",
    ),
    ModelInfo(
        id="azure/gpt-4-turbo",
        name="Azure GPT-4 Turbo",
        provider=Provider.AZURE,
        context_window=128000,
        supports_tools=True,
        supports_vision=True,
        max_output_tokens=4096,
        description="GPT-4 Turbo on Azure OpenAI",
    ),
    ModelInfo(
        id="azure/gpt-35-turbo",
        name="Azure GPT-3.5 Turbo",
        provider=Provider.AZURE,
        context_window=16385,
        supports_tools=True,
        supports_vision=False,
        max_output_tokens=4096,
        description="GPT-3.5 Turbo on Azure OpenAI",
    ),
]


class ModelRegistry:
    """
    Registry of available models across all providers.
    
    Provides methods to:
    - List available models
    - Get model info by ID
    - Filter models by provider or capability
    - Check which providers are configured
    """
    
    def __init__(self):
        self._models: dict[str, ModelInfo] = {}
        self._providers: dict[Provider, ProviderConfig] = {}
        
        for model in (
            GROQ_MODELS
            + GEMINI_MODELS
            + OPENAI_MODELS
            + ANTHROPIC_MODELS
            + OLLAMA_MODELS
            + AZURE_MODELS
        ):
            self._models[model.id] = model
    
    def configure_providers(
        self,
        groq: GroqConfig | None = None,
        google: GoogleAIConfig | None = None,
        openai: OpenAIConfig | None = None,
        anthropic: AnthropicConfig | None = None,
        ollama: OllamaConfig | None = None,
        azure: AzureConfig | None = None,
    ) -> None:
        """Configure providers with their settings."""
        if groq:
            self._providers[Provider.GROQ] = groq
        if google:
            self._providers[Provider.GOOGLE] = google
        if openai:
            self._providers[Provider.OPENAI] = openai
        if anthropic:
            self._providers[Provider.ANTHROPIC] = anthropic
        if ollama:
            self._providers[Provider.OLLAMA] = ollama
        if azure:
            self._providers[Provider.AZURE] = azure
    
    def configure_from_env(self) -> None:
        """Configure all providers from environment variables."""
        self._providers[Provider.GROQ] = GroqConfig.from_env()
        self._providers[Provider.GOOGLE] = GoogleAIConfig.from_env()
        self._providers[Provider.OPENAI] = OpenAIConfig.from_env()
        self._providers[Provider.ANTHROPIC] = AnthropicConfig.from_env()
        self._providers[Provider.OLLAMA] = OllamaConfig.from_env()
        self._providers[Provider.AZURE] = AzureConfig.from_env()
    
    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get model info by ID."""
        return self._models.get(model_id)
    
    def get_provider_config(self, provider: Provider) -> ProviderConfig | None:
        """Get provider configuration."""
        return self._providers.get(provider)
    
    def is_provider_configured(self, provider: Provider) -> bool:
        """Check if a provider is properly configured."""
        config = self._providers.get(provider)
        return config is not None and config.is_configured()
    
    def list_models(
        self,
        provider: Provider | None = None,
        supports_tools: bool | None = None,
        supports_vision: bool | None = None,
        only_configured: bool = True,
    ) -> list[ModelInfo]:
        """
        List available models with optional filtering.
        
        Args:
            provider: Filter by provider
            supports_tools: Filter by tool support
            supports_vision: Filter by vision support
            only_configured: Only return models from configured providers
            
        Returns:
            List of matching models
        """
        models = list(self._models.values())
        
        if provider is not None:
            models = [m for m in models if m.provider == provider]
        
        if supports_tools is not None:
            models = [m for m in models if m.supports_tools == supports_tools]
        
        if supports_vision is not None:
            models = [m for m in models if m.supports_vision == supports_vision]
        
        if only_configured:
            models = [m for m in models if self.is_provider_configured(m.provider)]
        
        return models
    
    def list_configured_providers(self) -> list[Provider]:
        """List all configured providers."""
        return [p for p in Provider if self.is_provider_configured(p)]
    
    def get_default_model(self) -> str:
        """Get the default model ID based on configured providers."""
        if self.is_provider_configured(Provider.GROQ):
            return os.environ.get("DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")
        if self.is_provider_configured(Provider.GOOGLE):
            return os.environ.get("DEFAULT_GEMINI_MODEL", "gemini/gemini-2.0-flash")
        if self.is_provider_configured(Provider.OPENAI):
            return "gpt-4o"
        if self.is_provider_configured(Provider.ANTHROPIC):
            return "claude-3-5-sonnet-20241022"
        if self.is_provider_configured(Provider.AZURE):
            return "azure/gpt-4o"
        if self.is_provider_configured(Provider.OLLAMA):
            return "ollama/llama3.2"
        return "llama-3.3-70b-versatile"
    
    def to_api_format(self, only_configured: bool = True) -> list[dict[str, Any]]:
        """Convert models to API response format."""
        models = self.list_models(only_configured=only_configured)
        return [
            {
                "id": m.id,
                "name": m.name,
                "provider": m.provider.value,
                "context_window": m.context_window,
                "supports_tools": m.supports_tools,
                "supports_vision": m.supports_vision,
                "max_output_tokens": m.max_output_tokens,
                "description": m.description,
            }
            for m in models
        ]


_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        _registry.configure_from_env()
    return _registry


def normalize_groq_legacy_model_id(model_id: str) -> str:
    """
    Map decommissioned GroqCloud IDs to current models.

    See https://console.groq.com/docs/deprecations for official replacement IDs.
    """
    key = (model_id or "").strip()
    if not key:
        return key
    low = key.lower()
    if low in ("llama3-70b-8192", "groq/llama3-70b-8192"):
        return "llama-3.3-70b-versatile"
    if low in ("llama3-8b-8192", "groq/llama3-8b-8192"):
        return "llama-3.1-8b-instant"
    return key


def normalize_gemini_model_id_for_litellm(model_id: str) -> str:
    """
    Map legacy or wrong-prefix Gemini IDs to a model string Google AI Studio + LiteLLM accept.

    LiteLLM uses the ``gemini/`` prefix for API-key (Google AI Studio) access — not ``google/``.
    Unversioned ``gemini-1.5-flash`` often returns HTTP 404 from Google; the default successor is
    ``gemini/gemini-2.0-flash``. Override with ``GEMINI_FLASH_SUCCESSOR_MODEL``.
    """
    key = (model_id or "").strip()
    if not key:
        return key
    low = key.lower()
    successor = (os.environ.get("GEMINI_FLASH_SUCCESSOR_MODEL") or "gemini/gemini-2.0-flash").strip()
    legacy_flash = frozenset(
        {
            "gemini/gemini-1.5-flash",
            "google/gemini-1.5-flash",
            "gemini-1.5-flash",
        }
    )
    if low in legacy_flash:
        return successor
    if low.startswith("google/gemini"):
        return "gemini/" + key.split("/", 1)[1]
    return key


def get_litellm_model_name(model_id: str, registry: ModelRegistry | None = None) -> str:
    """
    Convert a model ID to the format expected by LiteLLM.
    
    Args:
        model_id: The model ID (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        registry: Optional model registry
        
    Returns:
        Model name in LiteLLM format
    """
    model_id = normalize_groq_legacy_model_id(normalize_gemini_model_id_for_litellm(model_id))
    if registry is None:
        registry = get_model_registry()
    
    model = registry.get_model(model_id)
    if model is None:
        return model_id

    if model.provider == Provider.GROQ:
        if model_id.lower().startswith("groq/"):
            return model_id
        return f"groq/{model_id}"

    if model.provider == Provider.GOOGLE:
        return model_id

    if model.provider == Provider.ANTHROPIC:
        return model_id
    
    if model.provider == Provider.OLLAMA:
        return model_id
    
    if model.provider == Provider.AZURE:
        config = registry.get_provider_config(Provider.AZURE)
        if config and isinstance(config, AzureConfig) and config.deployment_name:
            return f"azure/{config.deployment_name}"
        return model_id.replace("azure/", "")
    
    return model_id


def get_provider_env_vars(provider: Provider) -> dict[str, str | None]:
    """Get environment variables for a provider."""
    if provider == Provider.GROQ:
        return {
            "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
            "GROQ_API_BASE": os.environ.get("GROQ_API_BASE"),
        }
    if provider == Provider.GOOGLE:
        return {
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
        }
    if provider == Provider.OPENAI:
        return {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
            "OPENAI_ORGANIZATION": os.environ.get("OPENAI_ORGANIZATION"),
        }
    if provider == Provider.ANTHROPIC:
        return {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        }
    if provider == Provider.OLLAMA:
        return {
            "OLLAMA_API_BASE": os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"),
        }
    if provider == Provider.AZURE:
        return {
            "AZURE_API_KEY": os.environ.get("AZURE_API_KEY"),
            "AZURE_API_BASE": os.environ.get("AZURE_API_BASE"),
            "AZURE_API_VERSION": os.environ.get("AZURE_API_VERSION"),
            "AZURE_DEPLOYMENT_NAME": os.environ.get("AZURE_DEPLOYMENT_NAME"),
        }
    return {}
