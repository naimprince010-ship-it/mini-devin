"""
LLM Client for Plodder

This module provides a wrapper around LiteLLM for interacting with various
LLM providers (OpenAI, Anthropic, etc.) with support for tool calling.
"""

import json
import os
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from mini_devin.core.providers import (
    Provider,
    get_model_registry,
    get_litellm_model_name,
    AzureConfig,
)


def _is_gemini_litellm_model(model_id: str) -> bool:
    """True for Google AI Studio (``gemini/``) and mistaken ``google/gemini`` prefixes (normalized at call time)."""
    m = (model_id or "").strip().lower()
    return m.startswith("gemini/") or m.startswith("vertex_ai/") or m.startswith("google/gemini")


def _gemini_safety_settings_block_none() -> list[dict[str, str]]:
    """Core categories at BLOCK_NONE for code/terminal-heavy agent workloads."""
    categories = (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
    return [{"category": c, "threshold": "BLOCK_NONE"} for c in categories]


def _apply_gemini_safety_override(kwargs: dict[str, Any], model_name: str) -> None:
    """LiteLLM safety_settings BLOCK_NONE for Gemini / Vertex when enabled via env."""
    if not _is_gemini_litellm_model(model_name):
        return
    if os.environ.get("LLM_GEMINI_SAFETY_BLOCK_NONE", "true").lower() in ("0", "false", "no"):
        return
    kwargs["safety_settings"] = _gemini_safety_settings_block_none()


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
    max_tokens: int = 16384
    api_key: str | None = None
    api_base: str | None = None
    timeout: int = 300
    max_retries: int = 3
    provider: Provider | None = None
    # Max messages sent to the API (system preserved; non-system tail-trimmed). None = use env/heuristic.
    max_history_messages: int | None = None


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


def _tool_call_ids_from_api_assistant_dict(msg: dict[str, Any]) -> set[str]:
    """Collect OpenAI ``tool_calls[].id`` values from an assistant message dict."""
    raw = msg.get("tool_calls")
    if not raw:
        return set()
    out: set[str] = set()
    for tc in raw:
        if isinstance(tc, dict):
            tid = tc.get("id")
            if tid:
                out.add(str(tid))
    return out


def _openai_non_system_window_valid(msgs: list[dict[str, Any]]) -> bool:
    """
    True if ``msgs`` can be sent as a contiguous non-system suffix to OpenAI-style APIs.

    Rejects orphan ``tool`` messages and assistant rows with ``tool_calls`` that are not
    immediately followed by ``tool`` messages covering every ``tool_call_id``.
    """
    if not msgs:
        return True
    if msgs[0].get("role") == "tool":
        return False
    i = 0
    n = len(msgs)
    while i < n:
        m = msgs[i]
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            need = _tool_call_ids_from_api_assistant_dict(m)
            if not need:
                i += 1
                continue
            seen: set[str] = set()
            i += 1
            while i < n and msgs[i].get("role") == "tool":
                tid = msgs[i].get("tool_call_id")
                if tid is not None and str(tid) in need:
                    seen.add(str(tid))
                i += 1
            if seen != need:
                return False
            continue
        if role == "tool":
            return False
        i += 1
    return True


def _tail_trim_non_system_openai_safe(
    full_rest: list[dict[str, Any]],
    cap: int,
) -> list[dict[str, Any]]:
    """
    Keep at most ``cap`` messages from the tail of ``full_rest``, adjusting the left edge
    so the slice never starts with an orphan ``tool`` row or cuts through an assistant
    ``tool_calls`` block (which would omit required ``tool`` replies and break the API).
    """
    if cap <= 0:
        return []
    if len(full_rest) <= cap:
        return list(full_rest)
    start = len(full_rest) - cap
    while start < len(full_rest):
        window = full_rest[start:]
        if _openai_non_system_window_valid(window):
            return window
        start += 1
    return []


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
            raise ImportError(
                "litellm is required. From the project root run: poetry install "
                "(recommended), or: pip install litellm — then restart the API."
            )
        
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
        elif isinstance(self.config.model, str) and _is_gemini_litellm_model(self.config.model):
            self.config.provider = Provider.GOOGLE

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
        elif provider == Provider.GOOGLE:
            if self.config.api_key:
                os.environ["GEMINI_API_KEY"] = self.config.api_key
                if not (os.environ.get("GOOGLE_API_KEY") or "").strip():
                    os.environ["GOOGLE_API_KEY"] = self.config.api_key
        else:
            if self.config.api_key and provider is None:
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
            litellm.organization = ""
            
            try:
                import httpx
                from openai import AsyncOpenAI
                from dotenv import load_dotenv

                load_dotenv(override=True)
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

    def replace_conversation_from_api_messages(self, messages: list[dict[str, Any]]) -> None:
        """Restore conversation from OpenAI-style dicts (e.g. DB ``conversation_json``)."""
        out: list[LLMMessage] = []
        for m in messages:
            role = str(m.get("role") or "user")
            raw_content = m.get("content")
            if isinstance(raw_content, list):
                content: str | None = json.dumps(raw_content, default=str)
            elif raw_content is None:
                content = None
            else:
                content = str(raw_content)

            if role == "assistant" and m.get("tool_calls"):
                tcs: list[ToolCall] = []
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict) and "function" in tc:
                        fn = tc["function"]
                        name = str(fn.get("name") or "")
                        raw_args = fn.get("arguments", "{}")
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args) if raw_args.strip() else {}
                            except json.JSONDecodeError:
                                args = {"_raw": raw_args}
                        elif isinstance(raw_args, dict):
                            args = raw_args
                        else:
                            args = {}
                        tcs.append(ToolCall(id=str(tc.get("id") or ""), name=name, arguments=args))
                out.append(LLMMessage(role="assistant", content=content, tool_calls=tcs))
            elif role == "tool":
                out.append(
                    LLMMessage(
                        role="tool",
                        content=content or "",
                        tool_call_id=m.get("tool_call_id"),
                        name=m.get("name"),
                    )
                )
            else:
                out.append(LLMMessage(role=role, content=content))
        self.conversation = out

    def _conversation_message_limit(self) -> int | None:
        """How many messages to send (including system). None = no trimming."""
        if self.config.max_history_messages is not None:
            return self.config.max_history_messages if self.config.max_history_messages > 0 else None
        raw = (os.environ.get("LLM_MAX_HISTORY_MESSAGES") or "").strip()
        if raw.isdigit():
            n = int(raw)
            return n if n > 0 else None
        if _is_gemini_litellm_model(self.config.model):
            return 200
        return 80

    def get_conversation_for_api(self) -> list[dict[str, Any]]:
        """Get conversation in API format (optional tail trim via LLM_MAX_HISTORY_MESSAGES)."""
        msgs = [msg.to_dict() for msg in self.conversation]
        limit = self._conversation_message_limit()
        if limit is None or len(msgs) <= limit:
            return msgs
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        rest = [m for m in msgs if m.get("role") != "system"]
        cap_rest = max(0, limit - len(system_msgs))
        if cap_rest <= 0:
            return system_msgs
        if len(rest) <= cap_rest:
            return msgs
        rest = _tail_trim_non_system_openai_safe(rest, cap_rest)
        return system_msgs + rest

    async def completion_ephemeral(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """
        One-shot completion without mutating ``self.conversation`` (summaries, condenser, etc.).
        Applies the same Gemini safety override as ``complete`` when applicable.
        """
        model_name = get_litellm_model_name(self.config.model)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": min(max_tokens, self.config.max_tokens),
            "timeout": self.config.timeout,
            "stream": False,
        }
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        _apply_gemini_safety_override(kwargs, model_name)
        if self._custom_client and self.config.provider == Provider.OPENAI:
            kwargs["client"] = self._custom_client
        if self.config.provider == Provider.GOOGLE and self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        try:
            response = await acompletion(**kwargs)
        except Exception as e:
            raise RuntimeError(f"LLM ephemeral completion failed: {e}") from e
        choice = response.choices[0]
        message = choice.message
        text = getattr(message, "content", None) or ""
        if hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens += getattr(response.usage, "prompt_tokens", 0)
            self.total_completion_tokens += getattr(response.usage, "completion_tokens", 0)
            self.total_tokens_used += getattr(response.usage, "total_tokens", 0)
        return str(text).strip()

    async def complete(
        self,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        stream: bool = False,
        on_token: Callable[[str], Any] | None = None,
        messages_for_api: list[dict[str, Any]] | None = None,
        ephemeral_user_messages: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """
        Get a completion from the LLM.
        
        Args:
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or {"type": "function", "function": {"name": "..."}}
            stream: Whether to stream the response
            on_token: Optional callback for streaming tokens
            messages_for_api: When set, use this message list instead of ``get_conversation_for_api()``.
            ephemeral_user_messages: Extra user messages appended (e.g. PLAN.md + file context).

        Returns:
            LLMResponse with content and/or tool calls
        """
        if messages_for_api is not None:
            messages = [dict(m) for m in messages_for_api]
        else:
            messages = self.get_conversation_for_api()
        if ephemeral_user_messages:
            messages = messages + [dict(m) for m in ephemeral_user_messages]

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

        _apply_gemini_safety_override(kwargs, model_name)

        if self._custom_client and self.config.provider == Provider.OPENAI:
             kwargs["client"] = self._custom_client

        if self.config.provider == Provider.GOOGLE and self.config.api_key:
            kwargs["api_key"] = self.config.api_key

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
                raise RuntimeError(
                    f"LLM Model Not Found: {model_name}. "
                    "For Google AI Studio, LiteLLM expects ``gemini/<model>`` (not ``google/...``). "
                    "Legacy ``gemini/gemini-1.5-flash`` is remapped to ``gemini/gemini-2.0-flash`` by default; "
                    "set GEMINI_FLASH_SUCCESSOR_MODEL to override. Raw error: "
                    + error_msg[:500]
                )
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
                    # DEBUG LOG (Windows cp1252 consoles crash on arrows etc. in npm output)
                    try:
                        print(f"[TOKEN STREAM CHUNK]: {repr(chunk_text)}")
                    except UnicodeEncodeError:
                        safe = repr(chunk_text).encode("ascii", "backslashreplace").decode("ascii")
                        print(f"[TOKEN STREAM CHUNK]: {safe}")
                    if on_token:
                        import asyncio
                        if inspect.iscoroutinefunction(on_token):
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
                    
            # Parse streaming tool calls (drop incomplete chunks — empty id breaks OpenAI tool ordering)
            tool_calls = []
            for _, tc_data in sorted(tool_calls_dict.items()):
                if not (tc_data.get("id") or "").strip() or not (tc_data.get("name") or "").strip():
                    continue
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

    if model is None or (
        isinstance(model, str) and model.strip().lower() in ("auto", "default", "")
    ):
        model = os.environ.get("LLM_MODEL") or registry.get_default_model()
    
    model_info = registry.get_model(model)
    if model_info:
        provider = model_info.provider
    elif isinstance(model, str) and _is_gemini_litellm_model(model):
        provider = Provider.GOOGLE
    else:
        provider = None

    if api_key is None and provider:
        if provider == Provider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == Provider.ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == Provider.AZURE:
            api_key = os.environ.get("AZURE_API_KEY")
        elif provider == Provider.GOOGLE:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif provider == Provider.OLLAMA:
            api_key = "ollama"

    if api_key is None and provider != Provider.GOOGLE:
        api_key = os.environ.get("OPENAI_API_KEY")

    if isinstance(api_key, str):
        api_key = api_key.strip() or None
    if isinstance(api_key, str) and api_key.upper() == "YOUR_KEY_HERE":
        api_key = None

    if not api_key:
        if provider == Provider.GOOGLE or (
            isinstance(model, str) and _is_gemini_litellm_model(model)
        ):
            raise ValueError(
                "No Gemini API key is set. Add GEMINI_API_KEY=... (or GOOGLE_API_KEY) to the "
                ".env file in the project root (next to pyproject.toml), save, and restart the API."
            )
        raise ValueError(
            "No LLM API key is set. For OpenAI models, add OPENAI_API_KEY=sk-... to the "
            ".env file in the project root (next to pyproject.toml), save, and restart the API."
        )

    max_out_raw = (os.environ.get("LLM_MAX_OUTPUT_TOKENS") or "").strip()
    if max_out_raw.isdigit():
        max_tokens = int(max_out_raw)
    elif model_info is not None:
        max_tokens = model_info.max_output_tokens
    elif provider == Provider.GOOGLE:
        max_tokens = 8192
    else:
        max_tokens = 16384

    timeout_raw = (os.environ.get("LLM_TIMEOUT_SEC") or "").strip()
    if timeout_raw.isdigit():
        timeout = int(timeout_raw)
    else:
        timeout = 300

    config = LLMConfig(
        model=model,
        api_key=api_key,
        temperature=temperature,
        api_base=api_base,
        provider=provider,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return LLMClient(config)
