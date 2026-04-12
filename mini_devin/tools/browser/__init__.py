"""
Browser Tools for Plodder

This module provides browser capabilities:
- Search: API-based web search (Tavily/SerpAPI)
- Fetch: Headless HTTP fetch with readability extraction
- Interactive: Selenium-based interactive browser (legacy)
- Playwright: Full visual automation with Playwright (recommended)
- Citations: Storage for fetched page references
"""

from .search import BrowserSearchTool, create_search_tool
from .fetch import BrowserFetchTool, create_fetch_tool
from .interactive import BrowserInteractiveTool, create_interactive_tool
from .playwright_tool import PlaywrightBrowserTool, create_playwright_tool
from .citations import CitationStore, Citation, create_citation_store

__all__ = [
    "BrowserSearchTool",
    "create_search_tool",
    "BrowserFetchTool",
    "create_fetch_tool",
    "BrowserInteractiveTool",
    "create_interactive_tool",
    "PlaywrightBrowserTool",
    "create_playwright_tool",
    "CitationStore",
    "Citation",
    "create_citation_store",
]

