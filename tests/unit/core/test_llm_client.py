"""Unit tests for the LLM client module."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mini_devin.core.llm_client import (
    LLMConfig,
    ToolCall,
    LLMMessage,
    LLMResponse,
    LLMClient,
    create_llm_client,
)
from mini_devin.core.providers import Provider


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_config(self):
        """Test default LLMConfig values."""
        config = LLMConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.api_key is None
        assert config.api_base is None
        assert config.timeout == 120
        assert config.max_retries == 3
        assert config.provider is None

    def test_custom_config(self):
        """Test custom LLMConfig values."""
        config = LLMConfig(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=8192,
            api_key="test-key",
            api_base="https://api.example.com",
            timeout=60,
            max_retries=5,
            provider=Provider.ANTHROPIC,
        )
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 0.7
        assert config.max_tokens == 8192
        assert config.api_key == "test-key"
        assert config.api_base == "https://api.example.com"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.provider == Provider.ANTHROPIC


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        tc = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"path": "/test/file.txt"},
        )
        assert tc.id == "call_123"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/test/file.txt"}


class TestLLMMessage:
    """Tests for LLMMessage dataclass."""

    def test_user_message(self):
        """Test creating a user message."""
        msg = LLMMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.tool_calls == []
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_assistant_message_with_content(self):
        """Test creating an assistant message with content."""
        msg = LLMMessage(role="assistant", content="I can help with that.")
        assert msg.role == "assistant"
        assert msg.content == "I can help with that."

    def test_assistant_message_with_tool_calls(self):
        """Test creating an assistant message with tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="read_file", arguments={"path": "/test"}),
        ]
        msg = LLMMessage(role="assistant", tool_calls=tool_calls)
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "read_file"

    def test_tool_message(self):
        """Test creating a tool result message."""
        msg = LLMMessage(
            role="tool",
            content="File contents here",
            tool_call_id="call_1",
            name="read_file",
        )
        assert msg.role == "tool"
        assert msg.content == "File contents here"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "read_file"

    def test_to_dict_user_message(self):
        """Test converting user message to dict."""
        msg = LLMMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_to_dict_assistant_with_tool_calls(self):
        """Test converting assistant message with tool calls to dict."""
        tool_calls = [
            ToolCall(id="call_1", name="test_tool", arguments={"arg": "value"}),
        ]
        msg = LLMMessage(role="assistant", tool_calls=tool_calls)
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert "tool_calls" in d
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["id"] == "call_1"
        assert d["tool_calls"][0]["type"] == "function"
        assert d["tool_calls"][0]["function"]["name"] == "test_tool"
        assert json.loads(d["tool_calls"][0]["function"]["arguments"]) == {"arg": "value"}

    def test_to_dict_tool_message(self):
        """Test converting tool message to dict."""
        msg = LLMMessage(
            role="tool",
            content="result",
            tool_call_id="call_1",
            name="test_tool",
        )
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["content"] == "result"
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "test_tool"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_with_content(self):
        """Test creating a response with content."""
        response = LLMResponse(
            content="Hello!",
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="gpt-4o",
        )
        assert response.content == "Hello!"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 15
        assert response.model == "gpt-4o"

    def test_response_with_tool_calls(self):
        """Test creating a response with tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="test", arguments={}),
        ]
        response = LLMResponse(
            content=None,
            tool_calls=tool_calls,
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="gpt-4o",
        )
        assert response.content is None
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "tool_calls"


class TestLLMClient:
    """Tests for LLMClient class."""

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_client_initialization(self, mock_litellm):
        """Test LLMClient initialization."""
        config = LLMConfig(model="gpt-4o", api_key="test-key")
        client = LLMClient(config)
        assert client.config == config
        assert client.conversation == []
        assert client.total_tokens_used == 0

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_set_system_prompt(self, mock_litellm):
        """Test setting system prompt."""
        client = LLMClient(LLMConfig(api_key="test"))
        client.set_system_prompt("You are a helpful assistant.")
        assert len(client.conversation) == 1
        assert client.conversation[0].role == "system"
        assert client.conversation[0].content == "You are a helpful assistant."

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_set_system_prompt_replaces_existing(self, mock_litellm):
        """Test that setting system prompt replaces existing one."""
        client = LLMClient(LLMConfig(api_key="test"))
        client.set_system_prompt("First prompt")
        client.set_system_prompt("Second prompt")
        system_msgs = [m for m in client.conversation if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "Second prompt"

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_add_user_message(self, mock_litellm):
        """Test adding user message."""
        client = LLMClient(LLMConfig(api_key="test"))
        client.add_user_message("Hello!")
        assert len(client.conversation) == 1
        assert client.conversation[0].role == "user"
        assert client.conversation[0].content == "Hello!"

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_add_assistant_message(self, mock_litellm):
        """Test adding assistant message."""
        client = LLMClient(LLMConfig(api_key="test"))
        client.add_assistant_message(content="I can help!")
        assert len(client.conversation) == 1
        assert client.conversation[0].role == "assistant"
        assert client.conversation[0].content == "I can help!"

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_add_tool_result(self, mock_litellm):
        """Test adding tool result."""
        client = LLMClient(LLMConfig(api_key="test"))
        client.add_tool_result("call_1", "read_file", "file contents")
        assert len(client.conversation) == 1
        assert client.conversation[0].role == "tool"
        assert client.conversation[0].tool_call_id == "call_1"
        assert client.conversation[0].name == "read_file"
        assert client.conversation[0].content == "file contents"

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_clear_conversation(self, mock_litellm):
        """Test clearing conversation keeps system prompt."""
        client = LLMClient(LLMConfig(api_key="test"))
        client.set_system_prompt("System prompt")
        client.add_user_message("User message")
        client.add_assistant_message("Assistant message")
        client.clear_conversation()
        assert len(client.conversation) == 1
        assert client.conversation[0].role == "system"

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_get_conversation_for_api(self, mock_litellm):
        """Test getting conversation in API format."""
        client = LLMClient(LLMConfig(api_key="test"))
        client.set_system_prompt("System")
        client.add_user_message("User")
        api_conv = client.get_conversation_for_api()
        assert len(api_conv) == 2
        assert api_conv[0] == {"role": "system", "content": "System"}
        assert api_conv[1] == {"role": "user", "content": "User"}

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_get_usage_stats(self, mock_litellm):
        """Test getting usage statistics."""
        client = LLMClient(LLMConfig(api_key="test"))
        client.total_tokens_used = 100
        client.total_prompt_tokens = 60
        client.total_completion_tokens = 40
        stats = client.get_usage_stats()
        assert stats["total_tokens"] == 100
        assert stats["prompt_tokens"] == 60
        assert stats["completion_tokens"] == 40


class TestCreateLLMClient:
    """Tests for create_llm_client function."""

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"})
    def test_create_with_defaults(self, mock_litellm):
        """Test creating client with defaults."""
        client = create_llm_client()
        # Default model depends on which providers are configured
        assert client.config.model is not None
        assert client.config.temperature == 0.0

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_create_with_custom_model(self, mock_litellm):
        """Test creating client with custom model."""
        client = create_llm_client(model="gpt-4o-mini", api_key="test-key")
        assert client.config.model == "gpt-4o-mini"

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_create_with_temperature(self, mock_litellm):
        """Test creating client with custom temperature."""
        client = create_llm_client(api_key="test-key", temperature=0.5)
        assert client.config.temperature == 0.5

    @patch("mini_devin.core.llm_client.LITELLM_AVAILABLE", True)
    @patch("mini_devin.core.llm_client.litellm")
    def test_create_with_api_base(self, mock_litellm):
        """Test creating client with custom API base."""
        client = create_llm_client(
            api_key="test-key",
            api_base="http://localhost:11434",
        )
        assert client.config.api_base == "http://localhost:11434"
