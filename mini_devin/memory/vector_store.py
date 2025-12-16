"""
Vector Store for Semantic Retrieval

This module provides vector-based semantic search using embeddings.
Supports multiple embedding providers and storage backends.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class Document:
    """A document to be stored in the vector store."""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_text(cls, text: str, metadata: Optional[dict] = None) -> "Document":
        """Create a document from text, generating an ID from content hash."""
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        return cls(id=doc_id, content=text, metadata=metadata or {})
    
    @classmethod
    def from_file(cls, file_path: str, chunk_size: int = 1000) -> list["Document"]:
        """Create documents from a file, optionally chunking large files."""
        path = Path(file_path)
        if not path.exists():
            return []
        
        content = path.read_text(errors="ignore")
        
        if len(content) <= chunk_size:
            return [cls.from_text(content, {"file_path": file_path, "chunk": 0})]
        
        # Chunk the content
        documents = []
        chunks = _chunk_text(content, chunk_size)
        for i, chunk in enumerate(chunks):
            doc = cls.from_text(chunk, {"file_path": file_path, "chunk": i, "total_chunks": len(chunks)})
            documents.append(doc)
        
        return documents
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
        )


@dataclass
class SearchResult:
    """A search result from the vector store."""
    document: Document
    score: float
    rank: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank,
        }


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        raise NotImplementedError
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        raise NotImplementedError


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    Simple embedding provider using TF-IDF-like features.
    
    This is a fallback when no external embedding API is available.
    Uses character n-grams and word frequency features.
    """
    
    def __init__(self, dimension: int = 256):
        self._dimension = dimension
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, text: str) -> list[float]:
        """Generate a simple embedding using character n-grams."""
        # Normalize text
        text = text.lower()
        
        # Generate character n-grams (2-4 grams)
        ngrams = []
        for n in range(2, 5):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        
        # Also add words
        words = text.split()
        ngrams.extend(words)
        
        # Create embedding using hash-based feature extraction
        embedding = [0.0] * self._dimension
        
        for ngram in ngrams:
            # Hash the ngram to get an index
            idx = hash(ngram) % self._dimension
            # Use a simple count-based feature
            embedding[idx] += 1.0
        
        # Normalize the embedding
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._dimension = 1536 if "3-small" in model else 3072
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception:
            # Fall back to simple embedding
            fallback = SimpleEmbeddingProvider(self._dimension)
            return fallback.embed(text)
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except Exception:
            return [self.embed(text) for text in texts]


class VectorStore:
    """
    Vector store for semantic search.
    
    Stores documents with embeddings and supports similarity search.
    Uses in-memory storage with optional persistence to disk.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        persist_path: Optional[str] = None,
    ):
        self.embedding_provider = embedding_provider or SimpleEmbeddingProvider()
        self.persist_path = Path(persist_path) if persist_path else None
        self.documents: dict[str, Document] = {}
        
        # Load existing data if persist path exists
        if self.persist_path and self.persist_path.exists():
            self._load()
    
    def add(self, document: Document) -> None:
        """Add a document to the store."""
        # Generate embedding if not present
        if document.embedding is None:
            document.embedding = self.embedding_provider.embed(document.content)
        
        self.documents[document.id] = document
        
        # Persist if configured
        if self.persist_path:
            self._save()
    
    def add_batch(self, documents: list[Document]) -> None:
        """Add multiple documents to the store."""
        # Generate embeddings for documents without them
        texts_to_embed = []
        indices_to_embed = []
        
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                indices_to_embed.append(i)
        
        if texts_to_embed:
            embeddings = self.embedding_provider.embed_batch(texts_to_embed)
            for idx, embedding in zip(indices_to_embed, embeddings):
                documents[idx].embedding = embedding
        
        # Add all documents
        for doc in documents:
            self.documents[doc.id] = doc
        
        # Persist if configured
        if self.persist_path:
            self._save()
    
    def remove(self, doc_id: str) -> bool:
        """Remove a document from the store."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if self.persist_path:
                self._save()
            return True
        return False
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            filter_metadata: Filter results by metadata fields
            
        Returns:
            List of search results sorted by similarity
        """
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed(query)
        
        # Calculate similarities
        results = []
        for doc in self.documents.values():
            # Apply metadata filter
            if filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            # Calculate cosine similarity
            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                if score >= min_score:
                    results.append(SearchResult(document=doc, score=score))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Add ranks and limit
        for i, result in enumerate(results[:limit]):
            result.rank = i + 1
        
        return results[:limit]
    
    def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search using a pre-computed embedding."""
        results = []
        for doc in self.documents.values():
            if doc.embedding:
                score = self._cosine_similarity(embedding, doc.embedding)
                if score >= min_score:
                    results.append(SearchResult(document=doc, score=score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        for i, result in enumerate(results[:limit]):
            result.rank = i + 1
        
        return results[:limit]
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _save(self) -> None:
        """Save the store to disk."""
        if not self.persist_path:
            return
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "documents": [doc.to_dict() for doc in self.documents.values()],
        }
        
        self.persist_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load the store from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            for doc_data in data.get("documents", []):
                doc = Document.from_dict(doc_data)
                self.documents[doc.id] = doc
        except Exception:
            pass
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self.documents.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
    
    def get_statistics(self) -> dict:
        """Get statistics about the store."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_provider.dimension,
            "persist_path": str(self.persist_path) if self.persist_path else None,
        }


def _chunk_text(text: str, chunk_size: int, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                for punct in [". ", "! ", "? ", "\n"]:
                    sent_break = text.rfind(punct, start, end)
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + len(punct)
                        break
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return [c for c in chunks if c]


def create_vector_store(
    persist_path: Optional[str] = None,
    use_openai: bool = False,
    api_key: Optional[str] = None,
) -> VectorStore:
    """
    Create a new vector store.
    
    Args:
        persist_path: Path to persist the store (optional)
        use_openai: Whether to use OpenAI embeddings
        api_key: OpenAI API key (optional, uses env var if not provided)
        
    Returns:
        Configured VectorStore instance
    """
    if use_openai:
        provider = OpenAIEmbeddingProvider(api_key=api_key)
    else:
        provider = SimpleEmbeddingProvider()
    
    return VectorStore(embedding_provider=provider, persist_path=persist_path)
