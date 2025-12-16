"""
Browser Tools for Mini-Devin

This module provides browser capabilities:
- Search: API-based web search (Tavily/SerpAPI)
- Fetch: Headless HTTP fetch with readability extraction
- Interactive: Playwright for JS-heavy pages and interactions
- Citations: Storage for fetched page references
"""

from .search import BrowserSearchTool, create_search_tool
from .fetch import BrowserFetchTool, create_fetch_tool
from .interactive import BrowserInteractiveTool, create_interactive_tool
from .citations import CitationStore, Citation, create_citation_store

__all__ = [
    "BrowserSearchTool",
    "create_search_tool",
    "BrowserFetchTool",
    "create_fetch_tool",
    "BrowserInteractiveTool",
    "create_interactive_tool",
    "CitationStore",
    "Citation",
    "create_citation_store",
]
