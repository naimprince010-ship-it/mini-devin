"""
Browser Fetch Tool for Mini-Devin

This module implements headless HTTP fetch with content extraction:
- Fetch web pages via HTTP
- Extract clean text content using readability algorithms
- Handle various content types (HTML, JSON, text)
- Cache fetched pages for efficiency
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from .base import BaseBrowserTool, ToolResult


@dataclass
class FetchedPage:
    """A fetched web page with extracted content."""
    url: str
    title: str
    content: str
    raw_html: str | None = None
    content_type: str = "text/html"
    status_code: int = 200
    fetch_time: datetime = field(default_factory=datetime.utcnow)
    word_count: int = 0
    links: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class FetchResponse:
    """Response from a fetch operation."""
    success: bool
    page: FetchedPage | None
    error: str | None = None
    cached: bool = False
    fetch_time_ms: int = 0


class ContentExtractor:
    """
    Extract clean text content from HTML.
    
    Uses simple heuristics to extract main content:
    - Remove scripts, styles, navigation
    - Extract text from article/main/content elements
    - Clean up whitespace
    """
    
    # Tags to remove completely
    REMOVE_TAGS = [
        "script", "style", "nav", "header", "footer", "aside",
        "noscript", "iframe", "form", "button", "input",
    ]
    
    # Tags that likely contain main content
    CONTENT_TAGS = [
        "article", "main", "content", "post", "entry",
        "story", "text", "body-content",
    ]
    
    def extract(self, html: str, url: str) -> tuple[str, str, list[str], dict[str, str]]:
        """
        Extract content from HTML.
        
        Returns:
            Tuple of (title, content, links, metadata)
        """
        # Extract title
        title = self._extract_title(html)
        
        # Extract metadata
        metadata = self._extract_metadata(html)
        
        # Extract links
        links = self._extract_links(html, url)
        
        # Remove unwanted tags
        cleaned_html = self._remove_tags(html)
        
        # Extract text content
        content = self._extract_text(cleaned_html)
        
        # Clean up whitespace
        content = self._clean_whitespace(content)
        
        return title, content, links, metadata
    
    def _extract_title(self, html: str) -> str:
        """Extract page title."""
        # Try <title> tag
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try <h1> tag
        match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return "Untitled"
    
    def _extract_metadata(self, html: str) -> dict[str, str]:
        """Extract metadata from meta tags."""
        metadata = {}
        
        # Extract meta description
        match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            metadata["description"] = match.group(1)
        
        # Extract meta keywords
        match = re.search(
            r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            metadata["keywords"] = match.group(1)
        
        # Extract og:title
        match = re.search(
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            metadata["og_title"] = match.group(1)
        
        # Extract og:description
        match = re.search(
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            metadata["og_description"] = match.group(1)
        
        return metadata
    
    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract links from HTML."""
        links = []
        parsed_base = urlparse(base_url)
        
        for match in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\']', html, re.IGNORECASE):
            href = match.group(1)
            
            # Skip anchors and javascript
            if href.startswith("#") or href.startswith("javascript:"):
                continue
            
            # Make relative URLs absolute
            if href.startswith("/"):
                href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
            elif not href.startswith("http"):
                continue
            
            if href not in links:
                links.append(href)
        
        return links[:50]  # Limit to 50 links
    
    def _remove_tags(self, html: str) -> str:
        """Remove unwanted tags from HTML."""
        for tag in self.REMOVE_TAGS:
            # Remove opening and closing tags with content
            html = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>",
                "",
                html,
                flags=re.IGNORECASE | re.DOTALL
            )
            # Remove self-closing tags
            html = re.sub(rf"<{tag}[^>]*/?>", "", html, flags=re.IGNORECASE)
        
        return html
    
    def _extract_text(self, html: str) -> str:
        """Extract text from HTML."""
        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        
        # Decode HTML entities
        text = self._decode_entities(text)
        
        return text
    
    def _decode_entities(self, text: str) -> str:
        """Decode common HTML entities."""
        entities = {
            "&nbsp;": " ",
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&apos;": "'",
            "&#39;": "'",
            "&mdash;": "—",
            "&ndash;": "–",
            "&hellip;": "...",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r"\n\s*\n", "\n\n", text)
        
        # Strip leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        
        return text.strip()


class BrowserFetchTool(BaseBrowserTool):
    """
    Headless HTTP fetch tool with content extraction.
    
    Features:
    - Fetch web pages via HTTP
    - Extract clean text content
    - Handle various content types
    - Cache fetched pages
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_content_length: int = 1_000_000,  # 1MB
        cache_enabled: bool = True,
        user_agent: str | None = None,
    ):
        super().__init__(
            name="browser_fetch",
            description="Fetch web pages and extract clean text content",
        )
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.cache_enabled = cache_enabled
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; MiniDevin/1.0; +https://github.com/mini-devin)"
        )
        
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        )
        self._extractor = ContentExtractor()
        self._cache: dict[str, FetchedPage] = {}
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    async def execute(self, input_data: Any) -> ToolResult:
        """Fetch a web page."""
        url = input_data.url if hasattr(input_data, "url") else str(input_data)
        use_cache = getattr(input_data, "use_cache", self.cache_enabled)
        extract_content = getattr(input_data, "extract_content", True)
        
        start_time = datetime.utcnow()
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(url)
            if cache_key in self._cache:
                page = self._cache[cache_key]
                return ToolResult(
                    success=True,
                    data=FetchResponse(
                        success=True,
                        page=page,
                        cached=True,
                        fetch_time_ms=0,
                    ),
                    message=f"Retrieved cached page: {page.title}",
                )
        
        try:
            response = await self._client.get(url)
            
            content_type = response.headers.get("content-type", "text/html")
            
            # Check content length
            content_length = int(response.headers.get("content-length", 0))
            if content_length > self.max_content_length:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Content too large: {content_length} bytes",
                    error="Content exceeds maximum size",
                )
            
            raw_content = response.text
            
            # Extract content based on type
            if "text/html" in content_type and extract_content:
                title, content, links, metadata = self._extractor.extract(raw_content, url)
            elif "application/json" in content_type:
                title = "JSON Response"
                content = raw_content
                links = []
                metadata = {}
            else:
                title = "Text Content"
                content = raw_content
                links = []
                metadata = {}
            
            page = FetchedPage(
                url=url,
                title=title,
                content=content,
                raw_html=raw_content if "text/html" in content_type else None,
                content_type=content_type,
                status_code=response.status_code,
                word_count=len(content.split()),
                links=links,
                metadata=metadata,
            )
            
            # Cache the page
            if use_cache:
                cache_key = self._get_cache_key(url)
                self._cache[cache_key] = page
            
            end_time = datetime.utcnow()
            fetch_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return ToolResult(
                success=True,
                data=FetchResponse(
                    success=True,
                    page=page,
                    cached=False,
                    fetch_time_ms=fetch_time_ms,
                ),
                message=f"Fetched: {title} ({page.word_count} words)",
            )
            
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                data=FetchResponse(
                    success=False,
                    page=None,
                    error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                ),
                message=f"HTTP error: {e.response.status_code}",
                error=str(e),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=FetchResponse(
                    success=False,
                    page=None,
                    error=str(e),
                ),
                message=f"Fetch failed: {str(e)}",
                error=str(e),
            )
    
    def clear_cache(self) -> None:
        """Clear the page cache."""
        self._cache.clear()
    
    def get_cached_urls(self) -> list[str]:
        """Get list of cached URLs."""
        return [page.url for page in self._cache.values()]
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def create_fetch_tool(
    timeout: int = 30,
    cache_enabled: bool = True,
) -> BrowserFetchTool:
    """Create a browser fetch tool."""
    return BrowserFetchTool(
        timeout=timeout,
        cache_enabled=cache_enabled,
    )
