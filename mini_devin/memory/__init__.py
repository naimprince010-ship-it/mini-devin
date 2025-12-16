"""
Memory & Indexing Module for Mini-Devin

This module provides memory and context management for Phase 4:
- Code symbol index with tree-sitter for structural indexing
- Vector store for semantic retrieval (embeddings)
- Retrieval policies (when to search vs when to use embeddings)
- Working memory for current task context
- Long-term memory for decisions and lessons learned
"""

from .symbol_index import (
    Symbol,
    SymbolType,
    SymbolIndex,
    create_symbol_index,
)

from .vector_store import (
    Document,
    SearchResult,
    VectorStore,
    create_vector_store,
)

from .retrieval import (
    RetrievalPolicy,
    RetrievalResult,
    RetrievalManager,
    create_retrieval_manager,
)

from .working_memory import (
    WorkingMemory,
    MemoryItem,
    create_working_memory,
)

__all__ = [
    # Symbol index
    "Symbol",
    "SymbolType",
    "SymbolIndex",
    "create_symbol_index",
    # Vector store
    "Document",
    "SearchResult",
    "VectorStore",
    "create_vector_store",
    # Retrieval
    "RetrievalPolicy",
    "RetrievalResult",
    "RetrievalManager",
    "create_retrieval_manager",
    # Working memory
    "WorkingMemory",
    "MemoryItem",
    "create_working_memory",
]
