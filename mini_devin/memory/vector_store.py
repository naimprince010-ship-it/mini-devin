"""
Vector Store for Semantic Retrieval

This module provides vector-based semantic search using embeddings.
Supports multiple embedding providers and storage backends.
"""

import hashlib
import json
import math
import os
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
            "backend": "local",
        }


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed store with the same public interface as ``VectorStore``."""

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        persist_path: Optional[str] = None,
        collection_name: str = "plodder_memory",
    ):
        super().__init__(embedding_provider=embedding_provider, persist_path=None)
        self.persist_path = Path(persist_path) if persist_path else (Path.home() / ".mini-devin" / "chroma")
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_error: str | None = None
        try:
            import chromadb

            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_path))
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            self._init_error = str(e)

    @property
    def _available(self) -> bool:
        return self._collection is not None

    def add(self, document: Document) -> None:
        self.add_batch([document])

    def add_batch(self, documents: list[Document]) -> None:
        if not documents:
            return
        if not self._available:
            for doc in documents:
                super().add(doc)
            return

        texts_to_embed: list[str] = []
        idxs: list[int] = []
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                idxs.append(i)
        if texts_to_embed:
            embeddings = self.embedding_provider.embed_batch(texts_to_embed)
            for i, emb in zip(idxs, embeddings):
                documents[i].embedding = emb

        ids = [doc.id for doc in documents]
        vectors = [doc.embedding or self.embedding_provider.embed(doc.content) for doc in documents]
        docs = [doc.content for doc in documents]
        metas = []
        for doc in documents:
            meta = dict(doc.metadata or {})
            if not meta:
                meta = {"_plodder": "1"}
            metas.append(meta)
        self._collection.upsert(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)

    def remove(self, doc_id: str) -> bool:
        if not self._available:
            return super().remove(doc_id)
        self._collection.delete(ids=[doc_id])
        return True

    def get(self, doc_id: str) -> Optional[Document]:
        if not self._available:
            return super().get(doc_id)
        row = self._collection.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
        ids = row.get("ids") or []
        if not ids:
            return None
        docs = row.get("documents") or [[]]
        metas = row.get("metadatas") or [[]]
        embs = row.get("embeddings") or [[]]
        content = docs[0] if isinstance(docs, list) and docs else ""
        if isinstance(content, list):
            content = content[0] if content else ""
        metadata = metas[0] if isinstance(metas, list) and metas else {}
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else {}
        emb = embs[0] if isinstance(embs, list) and embs else None
        if isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
            emb = emb[0]
        return Document(id=doc_id, content=str(content or ""), metadata=dict(metadata or {}), embedding=emb)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        if not self._available:
            return super().search(query, limit=limit, min_score=min_score, filter_metadata=filter_metadata)

        query_embedding = self.embedding_provider.embed(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": max(1, int(limit)),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata
        row = self._collection.query(**kwargs)

        ids = (row.get("ids") or [[]])[0]
        docs = (row.get("documents") or [[]])[0]
        metas = (row.get("metadatas") or [[]])[0]
        dists = (row.get("distances") or [[]])[0]
        embs = (row.get("embeddings") or [[]])[0]

        out: list[SearchResult] = []
        for i, doc_id in enumerate(ids):
            dist = float(dists[i]) if i < len(dists) else 1.0
            score = max(0.0, min(1.0, 1.0 - dist))
            if score < min_score:
                continue
            metadata = dict(metas[i] or {}) if i < len(metas) else {}
            metadata.pop("_plodder", None)
            doc = Document(
                id=str(doc_id),
                content=str(docs[i]) if i < len(docs) else "",
                metadata=metadata,
                embedding=embs[i] if i < len(embs) else None,
            )
            out.append(SearchResult(document=doc, score=score, rank=len(out) + 1))
        return out[:limit]

    def clear(self) -> None:
        if not self._available:
            super().clear()
            return
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def get_statistics(self) -> dict:
        if not self._available:
            stats = super().get_statistics()
            stats["backend"] = "local_fallback"
            stats["backend_error"] = self._init_error
            return stats
        return {
            "total_documents": int(self._collection.count()),
            "embedding_dimension": self.embedding_provider.dimension,
            "persist_path": str(self.persist_path),
            "backend": "chroma",
            "collection_name": self.collection_name,
        }


class PineconeVectorStore(VectorStore):
    """Pinecone-backed store with local fallback when the backend is unavailable."""

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        persist_path: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        super().__init__(embedding_provider=embedding_provider, persist_path=persist_path)
        self.index_name = index_name or os.environ.get("PINECONE_INDEX", "")
        self.namespace = namespace or os.environ.get("PINECONE_NAMESPACE", "default")
        self._pc = None
        self._index = None
        self._init_error: str | None = None
        try:
            from pinecone import Pinecone

            api_key = os.environ.get("PINECONE_API_KEY", "")
            if not api_key or not self.index_name:
                raise RuntimeError("PINECONE_API_KEY or PINECONE_INDEX is missing")
            self._pc = Pinecone(api_key=api_key)
            self._index = self._pc.Index(self.index_name)
        except Exception as e:
            self._init_error = str(e)

    @property
    def _available(self) -> bool:
        return self._index is not None

    def add(self, document: Document) -> None:
        self.add_batch([document])

    def add_batch(self, documents: list[Document]) -> None:
        if not documents:
            return
        if not self._available:
            super().add_batch(documents)
            return

        texts_to_embed: list[str] = []
        idxs: list[int] = []
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                idxs.append(i)
        if texts_to_embed:
            embeddings = self.embedding_provider.embed_batch(texts_to_embed)
            for i, emb in zip(idxs, embeddings):
                documents[i].embedding = emb

        vectors = []
        for doc in documents:
            emb = doc.embedding or self.embedding_provider.embed(doc.content)
            metadata = dict(doc.metadata or {})
            metadata["content"] = doc.content
            vectors.append({"id": doc.id, "values": emb, "metadata": metadata})
        self._index.upsert(vectors=vectors, namespace=self.namespace)

    def remove(self, doc_id: str) -> bool:
        if not self._available:
            return super().remove(doc_id)
        self._index.delete(ids=[doc_id], namespace=self.namespace)
        return True

    def get(self, doc_id: str) -> Optional[Document]:
        if not self._available:
            return super().get(doc_id)
        row = self._index.fetch(ids=[doc_id], namespace=self.namespace)
        vectors = (row.get("vectors") or {}) if isinstance(row, dict) else {}
        item = vectors.get(doc_id)
        if not item:
            return None
        meta = dict(item.get("metadata") or {})
        content = str(meta.pop("content", ""))
        emb = item.get("values")
        return Document(id=doc_id, content=content, metadata=meta, embedding=emb)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        if not self._available:
            return super().search(query, limit=limit, min_score=min_score, filter_metadata=filter_metadata)

        query_embedding = self.embedding_provider.embed(query)
        row = self._index.query(
            vector=query_embedding,
            top_k=max(1, int(limit)),
            namespace=self.namespace,
            filter=filter_metadata,
            include_metadata=True,
            include_values=True,
        )
        matches = row.get("matches") if isinstance(row, dict) else getattr(row, "matches", [])
        out: list[SearchResult] = []
        for i, m in enumerate(matches or []):
            score = float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0) or 0.0)
            if score < min_score:
                continue
            metadata = dict(m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {})
            content = str(metadata.pop("content", ""))
            doc_id = str(m.get("id") if isinstance(m, dict) else getattr(m, "id", ""))
            values = m.get("values") if isinstance(m, dict) else getattr(m, "values", None)
            out.append(
                SearchResult(
                    document=Document(id=doc_id, content=content, metadata=metadata, embedding=values),
                    score=score,
                    rank=i + 1,
                )
            )
        return out

    def clear(self) -> None:
        if not self._available:
            super().clear()
            return
        self._index.delete(delete_all=True, namespace=self.namespace)

    def get_statistics(self) -> dict:
        if not self._available:
            stats = super().get_statistics()
            stats["backend"] = "local_fallback"
            stats["backend_error"] = self._init_error
            return stats
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_provider.dimension,
            "persist_path": str(self.persist_path) if self.persist_path else None,
            "backend": "pinecone",
            "index_name": self.index_name,
            "namespace": self.namespace,
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
    backend: Optional[str] = None,
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

    selected = (backend or os.environ.get("PLODDER_VECTOR_BACKEND") or "local").strip().lower()
    if selected == "chroma":
        return ChromaVectorStore(
            embedding_provider=provider,
            persist_path=persist_path,
            collection_name=os.environ.get("PLODDER_CHROMA_COLLECTION", "plodder_memory"),
        )
    if selected == "pinecone":
        return PineconeVectorStore(
            embedding_provider=provider,
            persist_path=persist_path,
            index_name=os.environ.get("PINECONE_INDEX", ""),
            namespace=os.environ.get("PINECONE_NAMESPACE", "default"),
        )
    return VectorStore(embedding_provider=provider, persist_path=persist_path)
