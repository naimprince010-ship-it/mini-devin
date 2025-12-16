"""
Retrieval Policies for Memory System

This module implements retrieval policies that determine when to use
lexical search (ripgrep), structural search (symbol index), or
semantic search (vector embeddings).
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .symbol_index import Symbol, SymbolIndex, SymbolType
from .vector_store import Document, SearchResult, VectorStore


class RetrievalMethod(str, Enum):
    """Methods for retrieving information."""
    LEXICAL = "lexical"  # Text/regex search (ripgrep)
    STRUCTURAL = "structural"  # Symbol-based search (tree-sitter)
    SEMANTIC = "semantic"  # Embedding-based search (vector store)
    HYBRID = "hybrid"  # Combination of methods


class QueryType(str, Enum):
    """Types of queries that influence retrieval strategy."""
    EXACT_MATCH = "exact_match"  # Looking for exact string/identifier
    DEFINITION = "definition"  # Looking for where something is defined
    USAGE = "usage"  # Looking for where something is used
    SIMILAR = "similar"  # Looking for similar code/concepts
    EXPLANATION = "explanation"  # Looking for documentation/comments
    ERROR = "error"  # Looking for error-related code
    GENERAL = "general"  # General search


@dataclass
class RetrievalPolicy:
    """Policy for how to retrieve information."""
    primary_method: RetrievalMethod
    fallback_methods: list[RetrievalMethod] = field(default_factory=list)
    query_type: QueryType = QueryType.GENERAL
    confidence: float = 0.8
    max_results: int = 10
    min_score: float = 0.3
    
    def to_dict(self) -> dict:
        return {
            "primary_method": self.primary_method.value,
            "fallback_methods": [m.value for m in self.fallback_methods],
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "max_results": self.max_results,
            "min_score": self.min_score,
        }


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    query: str
    method_used: RetrievalMethod
    results: list[Any]
    total_found: int
    policy: RetrievalPolicy
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "method_used": self.method_used.value,
            "total_found": self.total_found,
            "policy": self.policy.to_dict(),
            "metadata": self.metadata,
        }


class PolicySelector:
    """Selects the appropriate retrieval policy based on query analysis."""
    
    # Patterns that suggest specific query types
    DEFINITION_PATTERNS = [
        r"where\s+is\s+(\w+)\s+defined",
        r"definition\s+of\s+(\w+)",
        r"find\s+(\w+)\s+class",
        r"find\s+(\w+)\s+function",
        r"locate\s+(\w+)",
    ]
    
    USAGE_PATTERNS = [
        r"where\s+is\s+(\w+)\s+used",
        r"usage\s+of\s+(\w+)",
        r"references\s+to\s+(\w+)",
        r"calls\s+to\s+(\w+)",
        r"who\s+calls\s+(\w+)",
    ]
    
    SIMILAR_PATTERNS = [
        r"similar\s+to",
        r"like\s+(\w+)",
        r"related\s+to",
        r"examples\s+of",
    ]
    
    ERROR_PATTERNS = [
        r"error",
        r"exception",
        r"traceback",
        r"failed",
        r"bug",
        r"fix",
    ]
    
    def select_policy(self, query: str) -> RetrievalPolicy:
        """
        Select the best retrieval policy for a query.
        
        Args:
            query: The search query
            
        Returns:
            Appropriate RetrievalPolicy
        """
        query_lower = query.lower()
        
        # Check for exact identifier (looks like code)
        if self._is_identifier(query):
            return RetrievalPolicy(
                primary_method=RetrievalMethod.STRUCTURAL,
                fallback_methods=[RetrievalMethod.LEXICAL],
                query_type=QueryType.EXACT_MATCH,
                confidence=0.9,
            )
        
        # Check for definition queries
        for pattern in self.DEFINITION_PATTERNS:
            if re.search(pattern, query_lower):
                return RetrievalPolicy(
                    primary_method=RetrievalMethod.STRUCTURAL,
                    fallback_methods=[RetrievalMethod.LEXICAL],
                    query_type=QueryType.DEFINITION,
                    confidence=0.85,
                )
        
        # Check for usage queries
        for pattern in self.USAGE_PATTERNS:
            if re.search(pattern, query_lower):
                return RetrievalPolicy(
                    primary_method=RetrievalMethod.LEXICAL,
                    fallback_methods=[RetrievalMethod.STRUCTURAL],
                    query_type=QueryType.USAGE,
                    confidence=0.8,
                )
        
        # Check for similarity queries
        for pattern in self.SIMILAR_PATTERNS:
            if re.search(pattern, query_lower):
                return RetrievalPolicy(
                    primary_method=RetrievalMethod.SEMANTIC,
                    fallback_methods=[RetrievalMethod.LEXICAL],
                    query_type=QueryType.SIMILAR,
                    confidence=0.7,
                )
        
        # Check for error-related queries
        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern, query_lower):
                return RetrievalPolicy(
                    primary_method=RetrievalMethod.HYBRID,
                    fallback_methods=[RetrievalMethod.LEXICAL],
                    query_type=QueryType.ERROR,
                    confidence=0.75,
                )
        
        # Default to hybrid for general queries
        return RetrievalPolicy(
            primary_method=RetrievalMethod.HYBRID,
            fallback_methods=[RetrievalMethod.SEMANTIC, RetrievalMethod.LEXICAL],
            query_type=QueryType.GENERAL,
            confidence=0.6,
        )
    
    def _is_identifier(self, query: str) -> bool:
        """Check if query looks like a code identifier."""
        # Single word that looks like an identifier
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", query):
            return True
        # Qualified name (e.g., ClassName.method_name)
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$", query):
            return True
        # Snake_case or camelCase
        if "_" in query or (query[0].islower() and any(c.isupper() for c in query)):
            return True
        return False


class RetrievalManager:
    """
    Manages retrieval across multiple sources.
    
    Coordinates between symbol index, vector store, and lexical search
    based on retrieval policies.
    """
    
    def __init__(
        self,
        workspace_path: str,
        symbol_index: Optional[SymbolIndex] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.workspace_path = Path(workspace_path)
        self.symbol_index = symbol_index or SymbolIndex(workspace_path)
        self.vector_store = vector_store or VectorStore()
        self.policy_selector = PolicySelector()
        self._indexed = False
    
    def index_workspace(self, force: bool = False) -> dict:
        """
        Index the workspace for retrieval.
        
        Args:
            force: Force re-indexing even if already indexed
            
        Returns:
            Statistics about the indexing
        """
        if self._indexed and not force:
            return self.get_statistics()
        
        # Index symbols
        symbol_count = self.symbol_index.index_directory()
        
        # Index files for vector store
        doc_count = 0
        for ext in [".py", ".js", ".ts", ".go", ".rs", ".md", ".txt"]:
            for file_path in self.workspace_path.rglob(f"*{ext}"):
                # Skip non-source directories
                if any(part.startswith(".") or part in ["node_modules", "__pycache__", "venv", ".venv", "dist", "build"]
                       for part in file_path.parts):
                    continue
                
                try:
                    documents = Document.from_file(str(file_path))
                    self.vector_store.add_batch(documents)
                    doc_count += len(documents)
                except Exception:
                    pass
        
        self._indexed = True
        
        return {
            "symbols_indexed": symbol_count,
            "documents_indexed": doc_count,
        }
    
    def retrieve(
        self,
        query: str,
        policy: Optional[RetrievalPolicy] = None,
        max_results: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve information based on query and policy.
        
        Args:
            query: Search query
            policy: Retrieval policy (auto-selected if not provided)
            max_results: Maximum number of results
            
        Returns:
            RetrievalResult with found items
        """
        # Auto-select policy if not provided
        if policy is None:
            policy = self.policy_selector.select_policy(query)
        
        policy.max_results = max_results
        
        # Execute retrieval based on primary method
        results = self._execute_retrieval(query, policy.primary_method, policy)
        
        # Try fallback methods if primary didn't find enough results
        if len(results) < max_results // 2 and policy.fallback_methods:
            for fallback_method in policy.fallback_methods:
                fallback_results = self._execute_retrieval(query, fallback_method, policy)
                results.extend(fallback_results)
                if len(results) >= max_results:
                    break
        
        # Deduplicate and limit results
        seen = set()
        unique_results = []
        for result in results:
            key = self._get_result_key(result)
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return RetrievalResult(
            query=query,
            method_used=policy.primary_method,
            results=unique_results[:max_results],
            total_found=len(unique_results),
            policy=policy,
        )
    
    def _execute_retrieval(
        self,
        query: str,
        method: RetrievalMethod,
        policy: RetrievalPolicy,
    ) -> list[Any]:
        """Execute retrieval using a specific method."""
        if method == RetrievalMethod.STRUCTURAL:
            return self._structural_search(query, policy)
        elif method == RetrievalMethod.SEMANTIC:
            return self._semantic_search(query, policy)
        elif method == RetrievalMethod.LEXICAL:
            return self._lexical_search(query, policy)
        elif method == RetrievalMethod.HYBRID:
            return self._hybrid_search(query, policy)
        return []
    
    def _structural_search(self, query: str, policy: RetrievalPolicy) -> list[Symbol]:
        """Search using the symbol index."""
        # Determine symbol types to search based on query type
        symbol_types = None
        if policy.query_type == QueryType.DEFINITION:
            symbol_types = [SymbolType.CLASS, SymbolType.FUNCTION, SymbolType.METHOD]
        
        return self.symbol_index.search(query, symbol_types=symbol_types, limit=policy.max_results)
    
    def _semantic_search(self, query: str, policy: RetrievalPolicy) -> list[SearchResult]:
        """Search using the vector store."""
        return self.vector_store.search(
            query,
            limit=policy.max_results,
            min_score=policy.min_score,
        )
    
    def _lexical_search(self, query: str, policy: RetrievalPolicy) -> list[dict]:
        """Search using ripgrep (lexical/text search)."""
        import subprocess
        
        results = []
        
        try:
            # Use ripgrep for fast text search
            cmd = [
                "rg",
                "--json",
                "--max-count", str(policy.max_results * 2),
                "--type-add", "code:*.{py,js,ts,go,rs,java,c,cpp,h}",
                "--type", "code",
                query,
                str(self.workspace_path),
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    import json
                    data = json.loads(line)
                    if data.get("type") == "match":
                        match_data = data.get("data", {})
                        results.append({
                            "type": "lexical_match",
                            "file": match_data.get("path", {}).get("text", ""),
                            "line_number": match_data.get("line_number", 0),
                            "content": match_data.get("lines", {}).get("text", "").strip(),
                        })
                except Exception:
                    pass
        except Exception:
            # Fallback to simple grep if ripgrep fails
            pass
        
        return results[:policy.max_results]
    
    def _hybrid_search(self, query: str, policy: RetrievalPolicy) -> list[Any]:
        """Combine multiple search methods."""
        results = []
        
        # Get results from each method
        structural_results = self._structural_search(query, policy)
        semantic_results = self._semantic_search(query, policy)
        lexical_results = self._lexical_search(query, policy)
        
        # Interleave results (prioritize structural for code queries)
        max_per_method = policy.max_results // 3 + 1
        
        results.extend(structural_results[:max_per_method])
        results.extend(semantic_results[:max_per_method])
        results.extend(lexical_results[:max_per_method])
        
        return results
    
    def _get_result_key(self, result: Any) -> str:
        """Get a unique key for a result to enable deduplication."""
        if isinstance(result, Symbol):
            return f"symbol:{result.location.file_path}:{result.name}"
        elif isinstance(result, SearchResult):
            return f"doc:{result.document.id}"
        elif isinstance(result, dict):
            return f"lexical:{result.get('file', '')}:{result.get('line_number', 0)}"
        return str(id(result))
    
    def find_definition(self, name: str) -> Optional[Symbol]:
        """Find the definition of a symbol by name."""
        return self.symbol_index.get_symbol(name)
    
    def find_usages(self, name: str, max_results: int = 20) -> list[dict]:
        """Find usages of a symbol."""
        policy = RetrievalPolicy(
            primary_method=RetrievalMethod.LEXICAL,
            query_type=QueryType.USAGE,
            max_results=max_results,
        )
        return self._lexical_search(name, policy)
    
    def find_similar(self, text: str, max_results: int = 10) -> list[SearchResult]:
        """Find code/text similar to the given text."""
        return self.vector_store.search(text, limit=max_results)
    
    def get_context_for_file(self, file_path: str) -> dict:
        """Get context information for a file."""
        symbols = self.symbol_index.get_file_symbols(file_path)
        
        return {
            "file_path": file_path,
            "symbols": [s.to_dict() for s in symbols],
            "symbol_count": len(symbols),
            "classes": [s for s in symbols if s.symbol_type == SymbolType.CLASS],
            "functions": [s for s in symbols if s.symbol_type in [SymbolType.FUNCTION, SymbolType.METHOD]],
        }
    
    def get_statistics(self) -> dict:
        """Get statistics about the retrieval system."""
        return {
            "workspace": str(self.workspace_path),
            "indexed": self._indexed,
            "symbol_index": self.symbol_index.get_statistics(),
            "vector_store": self.vector_store.get_statistics(),
        }


def create_retrieval_manager(
    workspace_path: str,
    symbol_index: Optional[SymbolIndex] = None,
    vector_store: Optional[VectorStore] = None,
) -> RetrievalManager:
    """Create a new retrieval manager for a workspace."""
    return RetrievalManager(
        workspace_path=workspace_path,
        symbol_index=symbol_index,
        vector_store=vector_store,
    )
