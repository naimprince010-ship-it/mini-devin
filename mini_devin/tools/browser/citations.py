"""
Citation Storage for Mini-Devin

This module provides citation management for fetched web pages:
- Store citations with metadata
- Retrieve citations by ID or URL
- Export citations in various formats
- Track citation usage in agent responses
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Citation:
    """A citation for a web page or document."""
    citation_id: str
    url: str
    title: str
    snippet: str
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    author: str | None = None
    published_date: str | None = None
    source_type: str = "web"  # web, api, document
    metadata: dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            "citation_id": self.citation_id,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "accessed_at": self.accessed_at.isoformat(),
            "author": self.author,
            "published_date": self.published_date,
            "source_type": self.source_type,
            "metadata": self.metadata,
            "usage_count": self.usage_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Citation":
        """Create citation from dictionary."""
        accessed_at = data.get("accessed_at")
        if isinstance(accessed_at, str):
            accessed_at = datetime.fromisoformat(accessed_at)
        elif accessed_at is None:
            accessed_at = datetime.utcnow()
        
        return cls(
            citation_id=data["citation_id"],
            url=data["url"],
            title=data["title"],
            snippet=data["snippet"],
            accessed_at=accessed_at,
            author=data.get("author"),
            published_date=data.get("published_date"),
            source_type=data.get("source_type", "web"),
            metadata=data.get("metadata", {}),
            usage_count=data.get("usage_count", 0),
        )
    
    def format_inline(self) -> str:
        """Format citation for inline reference."""
        return f"[{self.citation_id}]"
    
    def format_full(self) -> str:
        """Format citation for bibliography."""
        parts = [f"[{self.citation_id}]"]
        
        if self.author:
            parts.append(self.author)
        
        parts.append(f'"{self.title}"')
        
        if self.published_date:
            parts.append(f"({self.published_date})")
        
        parts.append(self.url)
        parts.append(f"Accessed: {self.accessed_at.strftime('%Y-%m-%d')}")
        
        return " ".join(parts)
    
    def format_markdown(self) -> str:
        """Format citation as markdown."""
        return f"[{self.citation_id}]: {self.url} \"{self.title}\""


class CitationStore:
    """
    Storage and management for citations.
    
    Features:
    - Add citations from fetched pages
    - Retrieve by ID or URL
    - Export to JSON, markdown, BibTeX
    - Track usage statistics
    - Persist to disk
    """
    
    def __init__(self, storage_path: str | None = None):
        self._citations: dict[str, Citation] = {}
        self._url_index: dict[str, str] = {}  # url -> citation_id
        self._storage_path = Path(storage_path) if storage_path else None
        self._next_id = 1
        
        # Load existing citations if storage path exists
        if self._storage_path and self._storage_path.exists():
            self._load()
    
    def _generate_id(self) -> str:
        """Generate a unique citation ID."""
        citation_id = f"cite{self._next_id}"
        self._next_id += 1
        return citation_id
    
    def _generate_id_from_url(self, url: str) -> str:
        """Generate a deterministic ID from URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"cite_{url_hash}"
    
    def add(
        self,
        url: str,
        title: str,
        snippet: str,
        author: str | None = None,
        published_date: str | None = None,
        source_type: str = "web",
        metadata: dict[str, Any] | None = None,
    ) -> Citation:
        """Add a new citation."""
        # Check if URL already exists
        if url in self._url_index:
            existing_id = self._url_index[url]
            return self._citations[existing_id]
        
        citation_id = self._generate_id()
        
        citation = Citation(
            citation_id=citation_id,
            url=url,
            title=title,
            snippet=snippet[:500] if len(snippet) > 500 else snippet,
            author=author,
            published_date=published_date,
            source_type=source_type,
            metadata=metadata or {},
        )
        
        self._citations[citation_id] = citation
        self._url_index[url] = citation_id
        
        # Auto-save if storage path configured
        if self._storage_path:
            self._save()
        
        return citation
    
    def add_from_page(self, page: Any) -> Citation:
        """Add citation from a FetchedPage object."""
        return self.add(
            url=page.url,
            title=page.title,
            snippet=page.content[:500] if page.content else "",
            source_type="web",
            metadata=page.metadata if hasattr(page, "metadata") else {},
        )
    
    def add_from_search_result(self, result: Any) -> Citation:
        """Add citation from a SearchResult object."""
        return self.add(
            url=result.url,
            title=result.title,
            snippet=result.snippet,
            published_date=result.published_date if hasattr(result, "published_date") else None,
            source_type="search",
        )
    
    def get(self, citation_id: str) -> Citation | None:
        """Get citation by ID."""
        return self._citations.get(citation_id)
    
    def get_by_url(self, url: str) -> Citation | None:
        """Get citation by URL."""
        citation_id = self._url_index.get(url)
        if citation_id:
            return self._citations.get(citation_id)
        return None
    
    def mark_used(self, citation_id: str) -> None:
        """Mark a citation as used (increment usage count)."""
        if citation_id in self._citations:
            self._citations[citation_id].usage_count += 1
    
    def get_all(self) -> list[Citation]:
        """Get all citations."""
        return list(self._citations.values())
    
    def get_used(self) -> list[Citation]:
        """Get all citations that have been used."""
        return [c for c in self._citations.values() if c.usage_count > 0]
    
    def search(self, query: str) -> list[Citation]:
        """Search citations by title or snippet."""
        query_lower = query.lower()
        results = []
        
        for citation in self._citations.values():
            if (query_lower in citation.title.lower() or 
                query_lower in citation.snippet.lower()):
                results.append(citation)
        
        return results
    
    def remove(self, citation_id: str) -> bool:
        """Remove a citation."""
        if citation_id in self._citations:
            citation = self._citations[citation_id]
            del self._url_index[citation.url]
            del self._citations[citation_id]
            
            if self._storage_path:
                self._save()
            
            return True
        return False
    
    def clear(self) -> None:
        """Clear all citations."""
        self._citations.clear()
        self._url_index.clear()
        self._next_id = 1
        
        if self._storage_path:
            self._save()
    
    def export_json(self) -> str:
        """Export citations as JSON."""
        data = {
            "citations": [c.to_dict() for c in self._citations.values()],
            "exported_at": datetime.utcnow().isoformat(),
        }
        return json.dumps(data, indent=2)
    
    def export_markdown(self) -> str:
        """Export citations as markdown bibliography."""
        lines = ["# References", ""]
        
        for citation in sorted(self._citations.values(), key=lambda c: c.citation_id):
            lines.append(citation.format_markdown())
        
        return "\n".join(lines)
    
    def export_bibtex(self) -> str:
        """Export citations as BibTeX."""
        entries = []
        
        for citation in self._citations.values():
            entry_type = "misc"
            entry_id = citation.citation_id.replace("cite", "ref")
            
            fields = [
                f"  title = {{{citation.title}}}",
                f"  url = {{{citation.url}}}",
                f"  note = {{Accessed: {citation.accessed_at.strftime('%Y-%m-%d')}}}",
            ]
            
            if citation.author:
                fields.insert(0, f"  author = {{{citation.author}}}")
            
            if citation.published_date:
                fields.append(f"  year = {{{citation.published_date[:4]}}}")
            
            entry = f"@{entry_type}{{{entry_id},\n" + ",\n".join(fields) + "\n}"
            entries.append(entry)
        
        return "\n\n".join(entries)
    
    def _save(self) -> None:
        """Save citations to disk."""
        if not self._storage_path:
            return
        
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "citations": [c.to_dict() for c in self._citations.values()],
            "next_id": self._next_id,
            "saved_at": datetime.utcnow().isoformat(),
        }
        
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        """Load citations from disk."""
        if not self._storage_path or not self._storage_path.exists():
            return
        
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
            
            self._next_id = data.get("next_id", 1)
            
            for citation_data in data.get("citations", []):
                citation = Citation.from_dict(citation_data)
                self._citations[citation.citation_id] = citation
                self._url_index[citation.url] = citation.citation_id
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load citations: {e}")
    
    def __len__(self) -> int:
        """Get number of citations."""
        return len(self._citations)
    
    def __contains__(self, citation_id: str) -> bool:
        """Check if citation exists."""
        return citation_id in self._citations


def create_citation_store(storage_path: str | None = None) -> CitationStore:
    """Create a citation store."""
    return CitationStore(storage_path=storage_path)
