"""
Browser Search Tool for Mini-Devin

This module implements API-based web search using various providers:
- Tavily API (primary)
- SerpAPI (fallback)
- DuckDuckGo (free fallback)
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from .base import BaseBrowserTool, ToolResult


class SearchProvider(str, Enum):
    """Supported search providers."""
    TAVILY = "tavily"
    SERPAPI = "serpapi"
    DUCKDUCKGO = "duckduckgo"


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    score: float = 0.0
    published_date: str | None = None
    source: str | None = None


@dataclass
class SearchResponse:
    """Response from a search query."""
    query: str
    results: list[SearchResult]
    total_results: int
    provider: SearchProvider
    search_time_ms: int = 0
    error: str | None = None


class BrowserSearchTool(BaseBrowserTool):
    """
    API-based web search tool.
    
    Supports multiple providers:
    - Tavily: Best for AI-focused search with relevance scoring
    - SerpAPI: Google search results
    - DuckDuckGo: Free fallback option
    """
    
    def __init__(
        self,
        provider: SearchProvider = SearchProvider.TAVILY,
        api_key: str | None = None,
        max_results: int = 10,
        timeout: int = 30,
    ):
        super().__init__(
            name="browser_search",
            description="Search the web for information using API-based search",
        )
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
        self.max_results = max_results
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    def _get_api_key(self) -> str | None:
        """Get API key from environment."""
        if self.provider == SearchProvider.TAVILY:
            return os.environ.get("TAVILY_API_KEY")
        elif self.provider == SearchProvider.SERPAPI:
            return os.environ.get("SERPAPI_API_KEY")
        return None
    
    async def execute(self, input_data: Any) -> ToolResult:
        """Execute a search query."""
        query = input_data.query if hasattr(input_data, "query") else str(input_data)
        max_results = getattr(input_data, "max_results", self.max_results)
        
        start_time = datetime.utcnow()
        
        try:
            if self.provider == SearchProvider.TAVILY:
                response = await self._search_tavily(query, max_results)
            elif self.provider == SearchProvider.SERPAPI:
                response = await self._search_serpapi(query, max_results)
            else:
                response = await self._search_duckduckgo(query, max_results)
            
            end_time = datetime.utcnow()
            response.search_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return ToolResult(
                success=True,
                data=response,
                message=f"Found {len(response.results)} results for '{query}'",
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Search failed: {str(e)}",
                error=str(e),
            )
    
    async def _search_tavily(self, query: str, max_results: int) -> SearchResponse:
        """Search using Tavily API."""
        if not self.api_key:
            raise ValueError("Tavily API key not configured")
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                score=item.get("score", 0.0),
                published_date=item.get("published_date"),
                source=item.get("source"),
            ))
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            provider=SearchProvider.TAVILY,
        )
    
    async def _search_serpapi(self, query: str, max_results: int) -> SearchResponse:
        """Search using SerpAPI."""
        if not self.api_key:
            raise ValueError("SerpAPI key not configured")
        
        url = "https://serpapi.com/search"
        params = {
            "api_key": self.api_key,
            "q": query,
            "num": max_results,
            "engine": "google",
        }
        
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("organic_results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                score=item.get("position", 0) / 10.0,  # Convert position to score
                source="google",
            ))
        
        return SearchResponse(
            query=query,
            results=results[:max_results],
            total_results=len(results),
            provider=SearchProvider.SERPAPI,
        )
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> SearchResponse:
        """Search using DuckDuckGo (free, no API key required)."""
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Abstract (main result)
        if data.get("Abstract"):
            results.append(SearchResult(
                title=data.get("Heading", ""),
                url=data.get("AbstractURL", ""),
                snippet=data.get("Abstract", ""),
                score=1.0,
                source="duckduckgo",
            ))
        
        # Related topics
        for item in data.get("RelatedTopics", [])[:max_results - 1]:
            if isinstance(item, dict) and "Text" in item:
                results.append(SearchResult(
                    title=item.get("Text", "")[:100],
                    url=item.get("FirstURL", ""),
                    snippet=item.get("Text", ""),
                    score=0.5,
                    source="duckduckgo",
                ))
        
        return SearchResponse(
            query=query,
            results=results[:max_results],
            total_results=len(results),
            provider=SearchProvider.DUCKDUCKGO,
        )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def create_search_tool(
    provider: str = "tavily",
    api_key: str | None = None,
    max_results: int = 10,
) -> BrowserSearchTool:
    """Create a browser search tool."""
    provider_enum = SearchProvider(provider.lower())
    return BrowserSearchTool(
        provider=provider_enum,
        api_key=api_key,
        max_results=max_results,
    )
