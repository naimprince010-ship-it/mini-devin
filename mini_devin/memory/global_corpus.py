"""
Optional global corpus (RAG) over many repositories.

Stores embeddings in a separate persisted VectorStore from the per-workspace index.
Enable with GLOBAL_RAG_ENABLED=true and build the index using scripts/ingest_global_corpus.py.

Legal: only index material you have rights to use; see docs/GLOBAL_CORPUS.md.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .vector_store import SearchResult, VectorStore, create_vector_store

_store: VectorStore | None = None
_store_path: Optional[str] = None


def _default_corpus_path() -> Path:
    override = os.environ.get("MINI_DEVIN_GLOBAL_CORPUS_PATH", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".mini-devin" / "global_corpus" / "vector_store.json"


def get_global_corpus_store() -> VectorStore:
    """Singleton VectorStore for the global corpus (lazy)."""
    global _store, _store_path
    path = str(_default_corpus_path())
    if _store is not None and _store_path == path:
        return _store
    _default_corpus_path().parent.mkdir(parents=True, exist_ok=True)
    use_openai = os.environ.get("GLOBAL_RAG_OPENAI_EMBEDDINGS", "").lower() in ("1", "true", "yes")
    _store = create_vector_store(
        persist_path=path,
        use_openai=use_openai,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    _store_path = path
    return _store


def search_global_corpus(query: str, *, limit: int = 8) -> list[SearchResult]:
    """Semantic search over the global corpus (empty if not built)."""
    if not query.strip():
        return []
    store = get_global_corpus_store()
    return store.search(query.strip(), limit=limit, min_score=0.0)


def format_global_corpus_block(
    query: str,
    *,
    limit: int = 8,
    max_chars: int = 8000,
) -> str:
    """Format top hits as a markdown block for injection into the agent chat."""
    hits = search_global_corpus(query, limit=limit)
    if not hits:
        return ""
    parts: list[str] = [
        "## Global reference corpus (retrieved snippets)\n",
        "Use as patterns only; prefer the current workspace as source of truth.\n",
    ]
    used = 0
    for h in hits:
        meta = h.document.metadata or {}
        src = meta.get("repo", meta.get("file_path", "unknown"))
        chunk = f"### ({h.score:.3f}) {src}\n```\n{h.document.content.strip()}\n```\n"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n".join(parts).strip()
