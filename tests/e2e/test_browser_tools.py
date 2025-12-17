"""
End-to-End Tests for Browser Tools (Phase 10).

Tests that the browser tools (search, fetch, interactive) work correctly
for web-based tasks.
"""

import pytest

from mini_devin.orchestrator.agent import Agent
from mini_devin.config.settings import AgentGatesSettings


class TestBrowserSearchTool:
    """Tests for browser search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_search_tool_registered(self, mock_llm_client, temp_workspace):
        """Test that browser_search tool is registered."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        tool_schemas = agent._get_tool_schemas()
        tool_names = [t["function"]["name"] for t in tool_schemas]
        
        assert "browser_search" in tool_names
    
    @pytest.mark.asyncio
    async def test_search_tool_schema(self, mock_llm_client, temp_workspace):
        """Test that browser_search tool has correct schema."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        tool_schemas = agent._get_tool_schemas()
        search_tool = next(
            (t for t in tool_schemas if t["function"]["name"] == "browser_search"),
            None,
        )
        
        assert search_tool is not None
        assert "query" in search_tool["function"]["parameters"]["properties"]


class TestBrowserFetchTool:
    """Tests for browser fetch tool functionality."""
    
    @pytest.mark.asyncio
    async def test_fetch_tool_registered(self, mock_llm_client, temp_workspace):
        """Test that browser_fetch tool is registered."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        tool_schemas = agent._get_tool_schemas()
        tool_names = [t["function"]["name"] for t in tool_schemas]
        
        assert "browser_fetch" in tool_names
    
    @pytest.mark.asyncio
    async def test_fetch_tool_schema(self, mock_llm_client, temp_workspace):
        """Test that browser_fetch tool has correct schema."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        tool_schemas = agent._get_tool_schemas()
        fetch_tool = next(
            (t for t in tool_schemas if t["function"]["name"] == "browser_fetch"),
            None,
        )
        
        assert fetch_tool is not None
        assert "url" in fetch_tool["function"]["parameters"]["properties"]


class TestBrowserInteractiveTool:
    """Tests for browser interactive tool functionality."""
    
    @pytest.mark.asyncio
    async def test_interactive_tool_registered(self, mock_llm_client, temp_workspace):
        """Test that browser_interactive tool is registered."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        tool_schemas = agent._get_tool_schemas()
        tool_names = [t["function"]["name"] for t in tool_schemas]
        
        assert "browser_interactive" in tool_names
    
    @pytest.mark.asyncio
    async def test_interactive_tool_schema(self, mock_llm_client, temp_workspace):
        """Test that browser_interactive tool has correct schema."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        tool_schemas = agent._get_tool_schemas()
        interactive_tool = next(
            (t for t in tool_schemas if t["function"]["name"] == "browser_interactive"),
            None,
        )
        
        assert interactive_tool is not None
        assert "url" in interactive_tool["function"]["parameters"]["properties"]
        assert "actions" in interactive_tool["function"]["parameters"]["properties"]


class TestBrowserToolsIntegration:
    """Integration tests for browser tools working together."""
    
    @pytest.mark.asyncio
    async def test_all_browser_tools_available(self, mock_llm_client, temp_workspace):
        """Test that all browser tools are available."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        tool_schemas = agent._get_tool_schemas()
        tool_names = [t["function"]["name"] for t in tool_schemas]
        
        browser_tools = ["browser_search", "browser_fetch", "browser_interactive"]
        for tool in browser_tools:
            assert tool in tool_names, f"Browser tool {tool} not found"
    
    @pytest.mark.asyncio
    async def test_browser_tools_have_descriptions(self, mock_llm_client, temp_workspace):
        """Test that all browser tools have descriptions."""
        agent = Agent(
            llm_client=mock_llm_client,
            working_directory=temp_workspace,
            gates_settings=AgentGatesSettings(
                planning_required=False,
                review_required=False,
            ),
        )
        
        tool_schemas = agent._get_tool_schemas()
        browser_tools = ["browser_search", "browser_fetch", "browser_interactive"]
        
        for tool_name in browser_tools:
            tool = next(
                (t for t in tool_schemas if t["function"]["name"] == tool_name),
                None,
            )
            assert tool is not None
            assert "description" in tool["function"]
            assert len(tool["function"]["description"]) > 0
