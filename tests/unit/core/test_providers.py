"""Unit tests for the providers module."""

import os
import pytest
from unittest.mock import patch

from mini_devin.core.providers import (
    Provider,
    ModelInfo,
    ProviderConfig,
    OpenAIConfig,
    AnthropicConfig,
    OllamaConfig,
    AzureConfig,
    ModelRegistry,
    get_model_registry,
    get_litellm_model_name,
    get_provider_env_vars,
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    OLLAMA_MODELS,
    AZURE_MODELS,
)


class TestProvider:
    """Tests for Provider enum."""

    def test_provider_values(self):
        """Test that all provider values are correct."""
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OLLAMA.value == "ollama"
        assert Provider.AZURE.value == "azure"

    def test_provider_from_string(self):
        """Test creating Provider from string."""
        assert Provider("openai") == Provider.OPENAI
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("ollama") == Provider.OLLAMA
        assert Provider("azure") == Provider.AZURE


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating a ModelInfo instance."""
        model = ModelInfo(
            id="test-model",
            name="Test Model",
            provider=Provider.OPENAI,
            context_window=128000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=4096,
            description="A test model",
        )
        assert model.id == "test-model"
        assert model.name == "Test Model"
        assert model.provider == Provider.OPENAI
        assert model.context_window == 128000
        assert model.supports_tools is True
        assert model.supports_vision is True
        assert model.max_output_tokens == 4096
        assert model.description == "A test model"

    def test_model_info_defaults(self):
        """Test ModelInfo default values."""
        model = ModelInfo(
            id="test",
            name="Test",
            provider=Provider.OPENAI,
            context_window=1000,
        )
        assert model.supports_tools is True
        assert model.supports_vision is False
        assert model.max_output_tokens == 4096
        assert model.description == ""


class TestProviderConfig:
    """Tests for ProviderConfig classes."""

    def test_base_provider_config(self):
        """Test base ProviderConfig."""
        config = ProviderConfig(
            provider=Provider.OPENAI,
            api_key="test-key",
            enabled=True,
        )
        assert config.is_configured() is True

    def test_provider_config_not_configured(self):
        """Test ProviderConfig without API key."""
        config = ProviderConfig(
            provider=Provider.OPENAI,
            api_key=None,
            enabled=True,
        )
        assert config.is_configured() is False

    def test_provider_config_disabled(self):
        """Test disabled ProviderConfig."""
        config = ProviderConfig(
            provider=Provider.OPENAI,
            api_key="test-key",
            enabled=False,
        )
        assert config.is_configured() is False


class TestOpenAIConfig:
    """Tests for OpenAIConfig."""

    def test_openai_config_from_env(self):
        """Test OpenAIConfig.from_env()."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test-key",
            "OPENAI_API_BASE": "https://api.openai.com",
            "OPENAI_ORGANIZATION": "org-123",
            "OPENAI_ENABLED": "true",
        }):
            config = OpenAIConfig.from_env()
            assert config.api_key == "sk-test-key"
            assert config.api_base == "https://api.openai.com"
            assert config.organization == "org-123"
            assert config.enabled is True
            assert config.provider == Provider.OPENAI

    def test_openai_config_disabled(self):
        """Test OpenAIConfig when disabled."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test-key",
            "OPENAI_ENABLED": "false",
        }, clear=True):
            config = OpenAIConfig.from_env()
            assert config.enabled is False


class TestAnthropicConfig:
    """Tests for AnthropicConfig."""

    def test_anthropic_config_from_env(self):
        """Test AnthropicConfig.from_env()."""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
            "ANTHROPIC_ENABLED": "true",
        }):
            config = AnthropicConfig.from_env()
            assert config.api_key == "sk-ant-test-key"
            assert config.enabled is True
            assert config.provider == Provider.ANTHROPIC


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_ollama_config_from_env(self):
        """Test OllamaConfig.from_env()."""
        with patch.dict(os.environ, {
            "OLLAMA_API_BASE": "http://localhost:11434",
            "OLLAMA_ENABLED": "true",
        }):
            config = OllamaConfig.from_env()
            assert config.api_base == "http://localhost:11434"
            assert config.enabled is True
            assert config.provider == Provider.OLLAMA

    def test_ollama_config_is_configured(self):
        """Test OllamaConfig.is_configured() - doesn't require API key."""
        config = OllamaConfig(api_base="http://localhost:11434", enabled=True)
        assert config.is_configured() is True

    def test_ollama_config_default_api_base(self):
        """Test OllamaConfig default API base."""
        with patch.dict(os.environ, {}, clear=True):
            config = OllamaConfig.from_env()
            assert config.api_base == "http://localhost:11434"


class TestAzureConfig:
    """Tests for AzureConfig."""

    def test_azure_config_from_env(self):
        """Test AzureConfig.from_env()."""
        with patch.dict(os.environ, {
            "AZURE_API_KEY": "azure-key",
            "AZURE_API_BASE": "https://myresource.openai.azure.com",
            "AZURE_API_VERSION": "2024-02-15-preview",
            "AZURE_DEPLOYMENT_NAME": "gpt-4o-deployment",
            "AZURE_ENABLED": "true",
        }):
            config = AzureConfig.from_env()
            assert config.api_key == "azure-key"
            assert config.api_base == "https://myresource.openai.azure.com"
            assert config.api_version == "2024-02-15-preview"
            assert config.deployment_name == "gpt-4o-deployment"
            assert config.enabled is True
            assert config.provider == Provider.AZURE

    def test_azure_config_is_configured(self):
        """Test AzureConfig.is_configured() requires all fields."""
        config = AzureConfig(
            api_key="key",
            api_base="https://test.openai.azure.com",
            deployment_name="deployment",
            enabled=True,
        )
        assert config.is_configured() is True

    def test_azure_config_not_configured_missing_deployment(self):
        """Test AzureConfig not configured without deployment name."""
        config = AzureConfig(
            api_key="key",
            api_base="https://test.openai.azure.com",
            deployment_name=None,
            enabled=True,
        )
        assert config.is_configured() is False


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_registry_initialization(self):
        """Test ModelRegistry initializes with all models."""
        registry = ModelRegistry()
        total_models = len(OPENAI_MODELS) + len(ANTHROPIC_MODELS) + len(OLLAMA_MODELS) + len(AZURE_MODELS)
        assert len(registry._models) == total_models

    def test_get_model(self):
        """Test getting a model by ID."""
        registry = ModelRegistry()
        model = registry.get_model("gpt-4o")
        assert model is not None
        assert model.id == "gpt-4o"
        assert model.provider == Provider.OPENAI

    def test_get_model_not_found(self):
        """Test getting a non-existent model."""
        registry = ModelRegistry()
        model = registry.get_model("non-existent-model")
        assert model is None

    def test_configure_providers(self):
        """Test configuring providers."""
        registry = ModelRegistry()
        openai_config = OpenAIConfig(api_key="test-key", enabled=True)
        registry.configure_providers(openai=openai_config)
        assert registry.get_provider_config(Provider.OPENAI) == openai_config

    def test_is_provider_configured(self):
        """Test checking if provider is configured."""
        registry = ModelRegistry()
        openai_config = OpenAIConfig(api_key="test-key", enabled=True)
        registry.configure_providers(openai=openai_config)
        assert registry.is_provider_configured(Provider.OPENAI) is True
        assert registry.is_provider_configured(Provider.ANTHROPIC) is False

    def test_list_models_all(self):
        """Test listing all models."""
        registry = ModelRegistry()
        models = registry.list_models(only_configured=False)
        assert len(models) > 0

    def test_list_models_by_provider(self):
        """Test listing models by provider."""
        registry = ModelRegistry()
        openai_models = registry.list_models(provider=Provider.OPENAI, only_configured=False)
        assert all(m.provider == Provider.OPENAI for m in openai_models)

    def test_list_models_by_capability(self):
        """Test listing models by capability."""
        registry = ModelRegistry()
        tool_models = registry.list_models(supports_tools=True, only_configured=False)
        assert all(m.supports_tools is True for m in tool_models)

    def test_list_configured_providers(self):
        """Test listing configured providers."""
        registry = ModelRegistry()
        openai_config = OpenAIConfig(api_key="test-key", enabled=True)
        registry.configure_providers(openai=openai_config)
        configured = registry.list_configured_providers()
        assert Provider.OPENAI in configured

    def test_get_default_model_openai(self):
        """Test getting default model when OpenAI is configured."""
        registry = ModelRegistry()
        openai_config = OpenAIConfig(api_key="test-key", enabled=True)
        registry.configure_providers(openai=openai_config)
        default = registry.get_default_model()
        assert default == "gpt-4o"

    def test_get_default_model_anthropic(self):
        """Test getting default model when only Anthropic is configured."""
        registry = ModelRegistry()
        anthropic_config = AnthropicConfig(api_key="test-key", enabled=True)
        registry.configure_providers(anthropic=anthropic_config)
        default = registry.get_default_model()
        assert default == "claude-3-5-sonnet-20241022"

    def test_to_api_format(self):
        """Test converting models to API format."""
        registry = ModelRegistry()
        api_models = registry.to_api_format(only_configured=False)
        assert len(api_models) > 0
        assert all("id" in m for m in api_models)
        assert all("name" in m for m in api_models)
        assert all("provider" in m for m in api_models)


class TestGetLiteLLMModelName:
    """Tests for get_litellm_model_name function."""

    def test_openai_model(self):
        """Test OpenAI model name conversion."""
        name = get_litellm_model_name("gpt-4o")
        assert name == "gpt-4o"

    def test_anthropic_model(self):
        """Test Anthropic model name conversion."""
        name = get_litellm_model_name("claude-3-5-sonnet-20241022")
        assert name == "claude-3-5-sonnet-20241022"

    def test_ollama_model(self):
        """Test Ollama model name conversion."""
        name = get_litellm_model_name("ollama/llama3.2")
        assert name == "ollama/llama3.2"

    def test_unknown_model(self):
        """Test unknown model returns as-is."""
        name = get_litellm_model_name("unknown-model")
        assert name == "unknown-model"


class TestGetProviderEnvVars:
    """Tests for get_provider_env_vars function."""

    def test_openai_env_vars(self):
        """Test getting OpenAI environment variables."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            env_vars = get_provider_env_vars(Provider.OPENAI)
            assert "OPENAI_API_KEY" in env_vars
            assert env_vars["OPENAI_API_KEY"] == "test-key"

    def test_anthropic_env_vars(self):
        """Test getting Anthropic environment variables."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            env_vars = get_provider_env_vars(Provider.ANTHROPIC)
            assert "ANTHROPIC_API_KEY" in env_vars

    def test_ollama_env_vars(self):
        """Test getting Ollama environment variables."""
        env_vars = get_provider_env_vars(Provider.OLLAMA)
        assert "OLLAMA_API_BASE" in env_vars

    def test_azure_env_vars(self):
        """Test getting Azure environment variables."""
        env_vars = get_provider_env_vars(Provider.AZURE)
        assert "AZURE_API_KEY" in env_vars
        assert "AZURE_API_BASE" in env_vars
        assert "AZURE_DEPLOYMENT_NAME" in env_vars


class TestModelDefinitions:
    """Tests for model definitions."""

    def test_openai_models_valid(self):
        """Test that all OpenAI models have required fields."""
        for model in OPENAI_MODELS:
            assert model.id is not None
            assert model.name is not None
            assert model.provider == Provider.OPENAI
            assert model.context_window > 0

    def test_anthropic_models_valid(self):
        """Test that all Anthropic models have required fields."""
        for model in ANTHROPIC_MODELS:
            assert model.id is not None
            assert model.name is not None
            assert model.provider == Provider.ANTHROPIC
            assert model.context_window > 0

    def test_ollama_models_valid(self):
        """Test that all Ollama models have required fields."""
        for model in OLLAMA_MODELS:
            assert model.id is not None
            assert model.name is not None
            assert model.provider == Provider.OLLAMA
            assert model.id.startswith("ollama/")

    def test_azure_models_valid(self):
        """Test that all Azure models have required fields."""
        for model in AZURE_MODELS:
            assert model.id is not None
            assert model.name is not None
            assert model.provider == Provider.AZURE
            assert model.id.startswith("azure/")
