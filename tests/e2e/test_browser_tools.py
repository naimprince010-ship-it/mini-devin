"""
End-to-End Tests for Browser Tools (Phase 10).

Tests that the browser tools (search, fetch, interactive) work correctly
for web-based tasks.
"""

import pytest

from mini_devin.orchestrator.agent import Agent
from mini_devin.tools.browser.base import ToolResult



class TestBrowserSearchTool:
    """Tests for browser search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_search_tool_registered(self, mock_llm_client, temp_workspace):
        """Test that browser_search tool is registered."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,

        )
        
        tool_schemas = agent._get_tool_schemas()
        tool_names = [t["name"] for t in tool_schemas]
        
        assert "browser_search" in tool_names
    
    @pytest.mark.asyncio
    async def test_search_tool_schema(self, mock_llm_client, temp_workspace):
        """Test that browser_search tool has correct schema."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,

        )
        
        tool_schemas = agent._get_tool_schemas()
        search_tool = next(
            (t for t in tool_schemas if t["name"] == "browser_search"),
            None,
        )
        
        assert search_tool is not None
        assert "query" in search_tool["parameters"]["properties"]


class TestBrowserFetchTool:
    """Tests for browser fetch tool functionality."""
    
    @pytest.mark.asyncio
    async def test_fetch_tool_registered(self, mock_llm_client, temp_workspace):
        """Test that browser_fetch tool is registered."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,

        )
        
        tool_schemas = agent._get_tool_schemas()
        tool_names = [t["name"] for t in tool_schemas]
        
        assert "browser_fetch" in tool_names
    
    @pytest.mark.asyncio
    async def test_fetch_tool_schema(self, mock_llm_client, temp_workspace):
        """Test that browser_fetch tool has correct schema."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,

        )
        
        tool_schemas = agent._get_tool_schemas()
        fetch_tool = next(
            (t for t in tool_schemas if t["name"] == "browser_fetch"),
            None,
        )
        
        assert fetch_tool is not None
        assert "url" in fetch_tool["parameters"]["properties"]


class TestBrowserInteractiveTool:
    """Tests for browser interactive tool functionality."""
    
    @pytest.mark.asyncio
    async def test_interactive_tool_registered(self, mock_llm_client, temp_workspace):
        """Test that browser_interactive tool is registered."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,

        )
        
        tool_schemas = agent._get_tool_schemas()
        tool_names = [t["name"] for t in tool_schemas]
        
        assert "browser_interactive" in tool_names
    
    @pytest.mark.asyncio
    async def test_interactive_tool_schema(self, mock_llm_client, temp_workspace):
        """Test that browser_interactive tool has correct schema."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,

        )
        
        tool_schemas = agent._get_tool_schemas()
        interactive_tool = next(
            (t for t in tool_schemas if t["name"] == "browser_interactive"),
            None,
        )
        
        assert interactive_tool is not None
        assert "url" in interactive_tool["parameters"]["properties"]
        assert "action" in interactive_tool["parameters"]["properties"]


class TestBrowserToolsIntegration:
    """Integration tests for browser tools working together."""
    
    @pytest.mark.asyncio
    async def test_all_browser_tools_available(self, mock_llm_client, temp_workspace):
        """Test that all browser tools are available."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,

        )
        
        tool_schemas = agent._get_tool_schemas()
        tool_names = [t["name"] for t in tool_schemas]
        
        browser_tools = [
            "browser_search",
            "browser_fetch",
            "browser_interactive",
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_scroll",
            "browser_screenshot",
        ]
        for tool in browser_tools:
            assert tool in tool_names, f"Browser tool {tool} not found"
    
    @pytest.mark.asyncio
    async def test_browser_tools_have_descriptions(self, mock_llm_client, temp_workspace):
        """Test that all browser tools have descriptions."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,

        )
        
        tool_schemas = agent._get_tool_schemas()
        browser_tools = [
            "browser_search",
            "browser_fetch",
            "browser_interactive",
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_scroll",
            "browser_screenshot",
        ]
        
        for tool_name in browser_tools:
            tool = next(
                (t for t in tool_schemas if t["name"] == tool_name),
                None,
            )
            assert tool is not None
            assert "description" in tool
            assert len(tool["description"]) > 0

    @pytest.mark.asyncio
    async def test_advanced_browser_tool_schemas(self, mock_llm_client, temp_workspace):
        """Test that the advanced browser tools expose the expected parameters."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
        )

        tool_schemas = {tool["name"]: tool for tool in agent._get_tool_schemas()}

        assert "url" in tool_schemas["browser_navigate"]["parameters"]["properties"]
        assert "selector" in tool_schemas["browser_click"]["parameters"]["properties"]
        assert "x" in tool_schemas["browser_click"]["parameters"]["properties"]
        assert "text" in tool_schemas["browser_type"]["parameters"]["properties"]
        assert "submit" in tool_schemas["browser_type"]["parameters"]["properties"]
        assert "direction" in tool_schemas["browser_scroll"]["parameters"]["properties"]
        assert "full_page" in tool_schemas["browser_screenshot"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_advanced_browser_tools_emit_observations(self, mock_llm_client, temp_workspace):
        """Test that advanced browser tool results are recorded as OBSERVATION events."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
        )
        tool = agent.registry.get("browser_screenshot")
        assert tool is not None

        async def fake_execute(arguments):
            assert arguments == {"full_page": True}
            return ToolResult(
                success=True,
                data={
                    "action": "browser_screenshot",
                    "url": "https://example.com",
                    "title": "Example",
                    "detail": "full_page",
                    "screenshot_base64": "abc123",
                    "interactive_elements": [{"id": "b1"}],
                    "viewport_width": 1280,
                    "viewport_height": 720,
                    "action_time_ms": 42,
                },
                message="browser screenshot observation",
            )

        events = []
        tool.execute = fake_execute  # type: ignore[method-assign]
        agent._append_session_event = events.append  # type: ignore[method-assign]

        result = await agent._execute_tool("browser_screenshot", {"full_page": True})

        assert result == "browser screenshot observation"
        assert len(events) == 2
        event = events[-1]
        assert event.kind.value == "observation"
        assert event.tool_name == "browser_screenshot"
        assert event.output == "browser screenshot observation"
        assert event.meta["browser"]["interactive_element_count"] == 1
        assert event.meta["browser"]["has_screenshot"] is True
