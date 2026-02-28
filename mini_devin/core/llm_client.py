"""
LLM Client for Mini-Devin

This module provides a wrapper around LiteLLM for interacting with various
LLM providers (OpenAI, Anthropic, etc.) with support for tool calling.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from mini_devin.core.providers import (
    Provider,
    get_model_registry,
    get_litellm_model_name,
    AzureConfig,
)


# LiteLLM import
try:
    import litellm
    from litellm import acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: str | None = None
    api_base: str | None = None
    timeout: int = 120
    max_retries: int = 3
    provider: Provider | None = None


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMMessage:
    """A message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # For tool messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format for API calls."""
        msg: dict[str, Any] = {"role": self.role}
        
        if self.content is not None:
            msg["content"] = self.content
        
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        
        if self.name:
            msg["name"] = self.name
        
        return msg


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict[str, int]
    model: str


class LLMClient:
    """
    Client for interacting with LLMs via LiteLLM.
    
    Features:
    - Supports multiple providers (OpenAI, Anthropic, etc.)
    - Tool/function calling support
    - Conversation history management
    - Token usage tracking
    - Retry handling
    """
    
    def __init__(self, config: LLMConfig | None = None):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required. Install with: pip install litellm")
        
        self.config = config or LLMConfig()
        self.conversation: list[LLMMessage] = []
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        self._setup_provider()
        
        # Disable litellm logging noise
        litellm.set_verbose = True
    
    def _setup_provider(self) -> None:
        """Set up provider-specific configuration."""
        registry = get_model_registry()
        model_info = registry.get_model(self.config.model)
        
        if model_info:
            self.config.provider = model_info.provider
        
        provider = self.config.provider
        
        if provider == Provider.OPENAI:
            if self.config.api_key:
                os.environ["OPENAI_API_KEY"] = self.config.api_key
        elif provider == Provider.ANTHROPIC:
            if self.config.api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
        elif provider == Provider.OLLAMA:
            if not self.config.api_base:
                self.config.api_base = os.environ.get(
                    "OLLAMA_API_BASE", "http://localhost:11434"
                )
        elif provider == Provider.AZURE:
            if self.config.api_key:
                os.environ["AZURE_API_KEY"] = self.config.api_key
            config = registry.get_provider_config(Provider.AZURE)
            if config and isinstance(config, AzureConfig):
                if config.api_base:
                    self.config.api_base = config.api_base
        else:
            if self.config.api_key:
                os.environ["OPENAI_API_KEY"] = self.config.api_key
        
        # SSL Verification Workaround
        # We must use a custom AsyncOpenAI client for OpenAI to bypass SSL issues
        # and avoid organization header conflicts.
        self._custom_client = None
        if provider == Provider.OPENAI:
            if "OPENAI_ORGANIZATION" in os.environ:
                 del os.environ["OPENAI_ORGANIZATION"]
            if "OPENAI_ORG_ID" in os.environ:
                 del os.environ["OPENAI_ORG_ID"]
            
            # Additional safety for LiteLLM internals
            import litellm
            litellm.organization = ""
            
            try:
                import httpx
                from openai import AsyncOpenAI
                from dotenv import load_dotenv
                load_dotenv()
                self._custom_client = AsyncOpenAI(
                    api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY"),
                    organization=None,
                    http_client=httpx.AsyncClient(verify=False)
                )
            except ImportError:
                 pass
        else:
             # Fallback for others (though only OpenAI uses this path typically)
             try:
                import httpx
                litellm.aclient_session = httpx.AsyncClient(verify=False)
                litellm.client_session = httpx.Client(verify=False)
             except ImportError:
                pass
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt."""
        # Remove existing system message if any
        self.conversation = [m for m in self.conversation if m.role != "system"]
        # Add new system message at the beginning
        self.conversation.insert(0, LLMMessage(role="system", content=prompt))
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.conversation.append(LLMMessage(role="user", content=content))
    
    def add_assistant_message(
        self,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        """Add an assistant message to the conversation."""
        self.conversation.append(LLMMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls or [],
        ))
    
    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> None:
        """Add a tool result to the conversation."""
        self.conversation.append(LLMMessage(
            role="tool",
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name,
        ))
    
    def clear_conversation(self) -> None:
        """Clear the conversation history (keeps system prompt)."""
        system_msg = next((m for m in self.conversation if m.role == "system"), None)
        self.conversation = []
        if system_msg:
            self.conversation.append(system_msg)
    
    def get_conversation_for_api(self) -> list[dict[str, Any]]:
        """Get conversation in API format."""
        return [msg.to_dict() for msg in self.conversation]
    
    async def complete(
        self,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        stream: bool = False,
        on_token: Callable[[str], Any] | None = None,
    ) -> LLMResponse:
        """
        Get a completion from the LLM.
        
        Args:
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or {"type": "function", "function": {"name": "..."}}
            stream: Whether to stream the response
            on_token: Optional callback for streaming tokens
            
        Returns:
            LLMResponse with content and/or tool calls
        """
        messages = self.get_conversation_for_api()
        
        model_name = get_litellm_model_name(self.config.model)
        
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "stream": stream,
        }
        
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": tool,
                }
                for tool in tools
            ]
            kwargs["tool_choice"] = tool_choice
        
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        
        if self._custom_client and self.config.provider == Provider.OPENAI:
             kwargs["client"] = self._custom_client

        # Make the API call
        try:
            print(f"[LLM] Requesting completion: model={model_name}, tools={len(tools) if tools else 0}, stream={stream}")
            response = await acompletion(**kwargs)
        except Exception as e:
            print(f"[LLM] Error in acompletion: {str(e)}")
            # Raise a more descriptive error
            error_msg = str(e)
            if "AuthenticationError" in error_msg or "401" in error_msg:
                raise RuntimeError(f"LLM Authentication Failed: API Key might be invalid or missing for {model_name}")
            elif "RateLimitError" in error_msg or "429" in error_msg:
                raise RuntimeError(f"LLM Rate Limit Reached: {error_msg}")
            elif "NotFoundError" in error_msg or "404" in error_msg:
                raise RuntimeError(f"LLM Model Not Found: {model_name}. Check if the model ID is correct.")
            else:
                raise RuntimeError(f"LLM API Error: {error_msg}")
        
        content = ""
        tool_calls_dict = {}
        finish_reason = "stop"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_returned = model_name
        
        if stream:
            async for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue
                    
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue
                
                # Handle text chunks
                if getattr(delta, "content", None):
                    chunk_text = delta.content
                    content += chunk_text
                    # DEBUG LOG: Trace token string duplication issue from upstream
                    print(f"[TOKEN STREAM CHUNK]: {repr(chunk_text)}")
                    if on_token:
                        import asyncio
                        if asyncio.iscoroutinefunction(on_token):
                            await on_token(chunk_text)
                        else:
                            on_token(chunk_text)
                
                # Handle tool calls in streaming mode
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        index = tc.index
                        if index not in tool_calls_dict:
                            tool_calls_dict[index] = {
                                "id": getattr(tc, "id", "") or "",
                                "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else "",
                                "arguments": getattr(tc.function, "arguments", "") if hasattr(tc, "function") else ""
                            }
                        else:
                            if hasattr(tc, "id") and getattr(tc, "id", None):
                                tool_calls_dict[index]["id"] = tc.id
                            if hasattr(tc, "function"):
                                if getattr(tc.function, "name", None):
                                    tool_calls_dict[index]["name"] += tc.function.name
                                if getattr(tc.function, "arguments", None):
                                    tool_calls_dict[index]["arguments"] += tc.function.arguments
                
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                    
            # Parse streaming tool calls
            tool_calls = []
            for _, tc_data in sorted(tool_calls_dict.items()):
                try:
                    arguments = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    arguments=arguments
                ))
                
        else:
            # Parse non-streaming response
            choice = response.choices[0]
            message = choice.message
            content = message.content
            finish_reason = choice.finish_reason
            if hasattr(response, "model"):
                model_returned = response.model
                
            # Extract tool calls
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    ))
            
            # Track usage
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

        self.total_prompt_tokens += usage["prompt_tokens"]
        self.total_completion_tokens += usage["completion_tokens"]
        self.total_tokens_used += usage["total_tokens"]
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model_returned,
        )
    
    async def chat(
        self,
        user_message: str,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 10,
    ) -> str:
        """
        High-level chat interface with automatic tool execution.
        
        Args:
            user_message: The user's message
            tools: List of tool definitions
            tool_executor: Async function to execute tools: (name, args) -> result
            max_tool_rounds: Maximum rounds of tool calling
            
        Returns:
            The final assistant response text
        """
        self.add_user_message(user_message)
        
        for _ in range(max_tool_rounds):
            response = await self.complete(tools=tools)
            
            # If no tool calls, we're done
            if not response.tool_calls:
                self.add_assistant_message(content=response.content)
                return response.content or ""
            
            # Add assistant message with tool calls
            self.add_assistant_message(tool_calls=response.tool_calls)
            
            # Execute tools and add results
            if tool_executor:
                for tc in response.tool_calls:
                    try:
                        result = await tool_executor(tc.name, tc.arguments)
                        result_str = json.dumps(result) if not isinstance(result, str) else result
                    except Exception as e:
                        result_str = f"Error executing tool: {str(e)}"
                    
                    self.add_tool_result(tc.id, tc.name, result_str)
            else:
                # No executor, just note the tool calls
                for tc in response.tool_calls:
                    self.add_tool_result(
                        tc.id,
                        tc.name,
                        f"Tool {tc.name} called with args: {tc.arguments}",
                    )
        
        # Max rounds reached, get final response without tools
        response = await self.complete(tools=None)
        self.add_assistant_message(content=response.content)
        return response.content or ""
    
    def get_usage_stats(self) -> dict[str, int]:
        """Get token usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }


def create_llm_client(
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    api_base: str | None = None,
) -> LLMClient:
    """
    Create an LLM client with common defaults.
    
    Args:
        model: Model ID (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "ollama/llama3.2")
               If None, uses the default model based on configured providers.
        api_key: API key for the provider. If None, uses environment variable.
        temperature: Temperature for generation (0.0 = deterministic)
        api_base: Base URL for the API (mainly for Ollama or custom endpoints)
        
    Returns:
        Configured LLMClient instance
    """
    registry = get_model_registry()
    
    if model is None:
        model = registry.get_default_model()
    
    model_info = registry.get_model(model)
    provider = model_info.provider if model_info else None
    
    if api_key is None and provider:
        if provider == Provider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == Provider.ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == Provider.AZURE:
            api_key = os.environ.get("AZURE_API_KEY")
        elif provider == Provider.OLLAMA:
            api_key = "ollama"
    
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    config = LLMConfig(
        model=model,
        api_key=api_key,
        temperature=temperature,
        api_base=api_base,
        provider=provider,
    )
    return LLMClient(config)
