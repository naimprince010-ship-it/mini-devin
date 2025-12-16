"""
Conversation Memory for Mini-Devin Agent

This module provides long-term memory across sessions for learning from past tasks.
It stores:
- Task summaries and outcomes
- Lessons learned from successes and failures
- Reusable patterns and solutions
- User preferences and context
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import uuid


class ConversationEntryType(str, Enum):
    """Types of conversation memory entries."""
    TASK_SUMMARY = "task_summary"
    LESSON_LEARNED = "lesson_learned"
    ERROR_PATTERN = "error_pattern"
    SOLUTION_PATTERN = "solution_pattern"
    USER_PREFERENCE = "user_preference"
    CODE_PATTERN = "code_pattern"
    TOOL_USAGE = "tool_usage"
    FEEDBACK = "feedback"


class Importance(str, Enum):
    """Importance levels for memory entries."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConversationEntry:
    """An entry in conversation memory."""
    id: str
    entry_type: ConversationEntryType
    content: str
    summary: str
    importance: Importance = Importance.MEDIUM
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    embedding: Optional[list[float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "entry_type": self.entry_type.value,
            "content": self.content,
            "summary": self.summary,
            "importance": self.importance.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            entry_type=ConversationEntryType(data["entry_type"]),
            content=data["content"],
            summary=data["summary"],
            importance=Importance(data.get("importance", "medium")),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id"),
            task_id=data.get("task_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )
    
    def content_hash(self) -> str:
        """Get a hash of the content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class TaskSummary:
    """Summary of a completed task."""
    task_id: str
    session_id: str
    description: str
    outcome: str
    success: bool
    duration_seconds: int
    tools_used: list[str]
    files_modified: list[str]
    errors_encountered: list[str]
    lessons: list[str]
    
    def to_entry(self) -> ConversationEntry:
        """Convert to a ConversationEntry."""
        content = f"""Task: {self.description}
Outcome: {self.outcome}
Success: {self.success}
Duration: {self.duration_seconds}s
Tools used: {', '.join(self.tools_used)}
Files modified: {', '.join(self.files_modified)}
Errors: {', '.join(self.errors_encountered) if self.errors_encountered else 'None'}
Lessons: {'; '.join(self.lessons) if self.lessons else 'None'}"""
        
        return ConversationEntry(
            id=f"task_{self.task_id}",
            entry_type=ConversationEntryType.TASK_SUMMARY,
            content=content,
            summary=f"{'Success' if self.success else 'Failed'}: {self.description[:100]}",
            importance=Importance.HIGH if self.success else Importance.CRITICAL,
            tags=["task", "success" if self.success else "failure"] + self.tools_used[:3],
            metadata={
                "task_id": self.task_id,
                "session_id": self.session_id,
                "success": self.success,
                "duration_seconds": self.duration_seconds,
                "tools_used": self.tools_used,
                "files_modified": self.files_modified,
            },
            session_id=self.session_id,
            task_id=self.task_id,
        )


class ConversationMemory:
    """
    Long-term conversation memory for the agent.
    
    Provides:
    - Persistent storage of task summaries and lessons
    - Semantic search for relevant past experiences
    - Pattern recognition for common errors and solutions
    - User preference tracking
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_entries: int = 10000,
        embedding_fn: Optional[callable] = None,
    ):
        """
        Initialize conversation memory.
        
        Args:
            storage_path: Path to store memory data (JSON file)
            max_entries: Maximum number of entries to keep
            embedding_fn: Optional function to generate embeddings
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_entries = max_entries
        self.embedding_fn = embedding_fn
        
        self.entries: dict[str, ConversationEntry] = {}
        self._content_hashes: set[str] = set()
        
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def add(self, entry: ConversationEntry, deduplicate: bool = True) -> Optional[str]:
        """
        Add an entry to conversation memory.
        
        Args:
            entry: The entry to add
            deduplicate: If True, skip duplicate content
            
        Returns:
            The entry ID if added, None if deduplicated
        """
        if deduplicate:
            content_hash = entry.content_hash()
            if content_hash in self._content_hashes:
                return None
            self._content_hashes.add(content_hash)
        
        if not entry.id:
            entry.id = str(uuid.uuid4())[:8]
        
        if self.embedding_fn and entry.embedding is None:
            try:
                entry.embedding = self.embedding_fn(entry.content)
            except Exception:
                pass
        
        self.entries[entry.id] = entry
        
        self._enforce_limit()
        
        if self.storage_path:
            self._save()
        
        return entry.id
    
    def add_task_summary(self, summary: TaskSummary) -> str:
        """Add a task summary to memory."""
        entry = summary.to_entry()
        return self.add(entry, deduplicate=False)
    
    def add_lesson(
        self,
        lesson: str,
        context: str,
        tags: list[str] | None = None,
        importance: Importance = Importance.HIGH,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> Optional[str]:
        """Add a lesson learned to memory."""
        entry = ConversationEntry(
            id=f"lesson_{uuid.uuid4().hex[:8]}",
            entry_type=ConversationEntryType.LESSON_LEARNED,
            content=f"Lesson: {lesson}\nContext: {context}",
            summary=lesson[:100],
            importance=importance,
            tags=["lesson"] + (tags or []),
            session_id=session_id,
            task_id=task_id,
        )
        return self.add(entry)
    
    def add_error_pattern(
        self,
        error: str,
        cause: str,
        solution: str,
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> Optional[str]:
        """Add an error pattern to memory."""
        entry = ConversationEntry(
            id=f"error_{uuid.uuid4().hex[:8]}",
            entry_type=ConversationEntryType.ERROR_PATTERN,
            content=f"Error: {error}\nCause: {cause}\nSolution: {solution}",
            summary=f"Error: {error[:50]} -> {solution[:50]}",
            importance=Importance.HIGH,
            tags=["error", "pattern"] + (tags or []),
            metadata={"error": error, "cause": cause, "solution": solution},
            session_id=session_id,
        )
        return self.add(entry)
    
    def add_solution_pattern(
        self,
        problem: str,
        solution: str,
        code_example: str | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> Optional[str]:
        """Add a solution pattern to memory."""
        content = f"Problem: {problem}\nSolution: {solution}"
        if code_example:
            content += f"\nExample:\n```\n{code_example}\n```"
        
        entry = ConversationEntry(
            id=f"solution_{uuid.uuid4().hex[:8]}",
            entry_type=ConversationEntryType.SOLUTION_PATTERN,
            content=content,
            summary=f"Solution: {problem[:50]} -> {solution[:50]}",
            importance=Importance.HIGH,
            tags=["solution", "pattern"] + (tags or []),
            metadata={"problem": problem, "solution": solution},
            session_id=session_id,
        )
        return self.add(entry)
    
    def add_user_preference(
        self,
        preference: str,
        value: str,
        context: str | None = None,
    ) -> Optional[str]:
        """Add a user preference to memory."""
        content = f"Preference: {preference}\nValue: {value}"
        if context:
            content += f"\nContext: {context}"
        
        entry = ConversationEntry(
            id=f"pref_{hashlib.sha256(preference.encode()).hexdigest()[:8]}",
            entry_type=ConversationEntryType.USER_PREFERENCE,
            content=content,
            summary=f"Preference: {preference}",
            importance=Importance.CRITICAL,
            tags=["preference", "user"],
            metadata={"preference": preference, "value": value},
        )
        
        existing = self.entries.get(entry.id)
        if existing:
            existing.content = content
            existing.metadata = entry.metadata
            if self.storage_path:
                self._save()
            return existing.id
        
        return self.add(entry, deduplicate=False)
    
    def add_feedback(
        self,
        feedback: str,
        rating: int,
        context: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> Optional[str]:
        """Add user feedback to memory."""
        content = f"Feedback: {feedback}\nRating: {rating}/5"
        if context:
            content += f"\nContext: {context}"
        
        entry = ConversationEntry(
            id=f"feedback_{uuid.uuid4().hex[:8]}",
            entry_type=ConversationEntryType.FEEDBACK,
            content=content,
            summary=f"Feedback ({rating}/5): {feedback[:50]}",
            importance=Importance.HIGH if rating <= 2 else Importance.MEDIUM,
            tags=["feedback", f"rating_{rating}"],
            metadata={"rating": rating},
            session_id=session_id,
            task_id=task_id,
        )
        return self.add(entry)
    
    def get(self, entry_id: str) -> Optional[ConversationEntry]:
        """Get an entry by ID."""
        entry = self.entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
        return entry
    
    def search(
        self,
        query: str,
        entry_types: list[ConversationEntryType] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_importance: Importance | None = None,
    ) -> list[ConversationEntry]:
        """
        Search conversation memory.
        
        Args:
            query: Search query (keyword-based)
            entry_types: Filter by entry types
            tags: Filter by tags
            limit: Maximum results to return
            min_importance: Minimum importance level
            
        Returns:
            List of matching entries
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        importance_order = [Importance.CRITICAL, Importance.HIGH, Importance.MEDIUM, Importance.LOW]
        min_importance_idx = importance_order.index(min_importance) if min_importance else len(importance_order)
        
        for entry in self.entries.values():
            if entry_types and entry.entry_type not in entry_types:
                continue
            
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            entry_importance_idx = importance_order.index(entry.importance)
            if entry_importance_idx > min_importance_idx:
                continue
            
            content_lower = entry.content.lower()
            summary_lower = entry.summary.lower()
            
            score = 0
            for word in query_words:
                if word in content_lower:
                    score += 1
                if word in summary_lower:
                    score += 2
                if any(word in tag.lower() for tag in entry.tags):
                    score += 3
            
            if score > 0:
                results.append((score, entry))
        
        results.sort(key=lambda x: (-x[0], -x[1].access_count))
        
        matched_entries = [entry for _, entry in results[:limit]]
        for entry in matched_entries:
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
        
        return matched_entries
    
    def search_similar(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ConversationEntry]:
        """
        Search for similar entries using embeddings.
        
        Falls back to keyword search if embeddings are not available.
        """
        if not self.embedding_fn:
            return self.search(query, limit=limit)
        
        try:
            query_embedding = self.embedding_fn(query)
        except Exception:
            return self.search(query, limit=limit)
        
        results = []
        for entry in self.entries.values():
            if entry.embedding is None:
                continue
            
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            results.append((similarity, entry))
        
        results.sort(key=lambda x: -x[0])
        
        return [entry for _, entry in results[:limit]]
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_by_type(self, entry_type: ConversationEntryType) -> list[ConversationEntry]:
        """Get all entries of a specific type."""
        return [e for e in self.entries.values() if e.entry_type == entry_type]
    
    def get_by_session(self, session_id: str) -> list[ConversationEntry]:
        """Get all entries from a specific session."""
        return [e for e in self.entries.values() if e.session_id == session_id]
    
    def get_by_task(self, task_id: str) -> list[ConversationEntry]:
        """Get all entries from a specific task."""
        return [e for e in self.entries.values() if e.task_id == task_id]
    
    def get_user_preferences(self) -> dict[str, str]:
        """Get all user preferences as a dictionary."""
        preferences = {}
        for entry in self.get_by_type(ConversationEntryType.USER_PREFERENCE):
            pref = entry.metadata.get("preference")
            value = entry.metadata.get("value")
            if pref and value:
                preferences[pref] = value
        return preferences
    
    def get_error_solutions(self, error: str) -> list[str]:
        """Get solutions for similar errors."""
        results = self.search(
            error,
            entry_types=[ConversationEntryType.ERROR_PATTERN],
            limit=5,
        )
        return [e.metadata.get("solution", "") for e in results if e.metadata.get("solution")]
    
    def get_recent_lessons(self, limit: int = 10) -> list[ConversationEntry]:
        """Get recent lessons learned."""
        lessons = self.get_by_type(ConversationEntryType.LESSON_LEARNED)
        lessons.sort(key=lambda x: x.created_at, reverse=True)
        return lessons[:limit]
    
    def get_context_for_task(self, task_description: str, max_entries: int = 5) -> str:
        """
        Get relevant context from memory for a new task.
        
        Args:
            task_description: Description of the new task
            max_entries: Maximum entries to include
            
        Returns:
            Formatted context string
        """
        relevant = self.search(task_description, limit=max_entries)
        
        if not relevant:
            return ""
        
        context_parts = ["## Relevant Past Experience\n"]
        
        for entry in relevant:
            if entry.entry_type == ConversationEntryType.TASK_SUMMARY:
                context_parts.append(f"### Previous Task\n{entry.summary}\n")
            elif entry.entry_type == ConversationEntryType.LESSON_LEARNED:
                context_parts.append(f"### Lesson\n{entry.content}\n")
            elif entry.entry_type == ConversationEntryType.ERROR_PATTERN:
                context_parts.append(f"### Known Error Pattern\n{entry.content}\n")
            elif entry.entry_type == ConversationEntryType.SOLUTION_PATTERN:
                context_parts.append(f"### Solution Pattern\n{entry.content}\n")
        
        return "\n".join(context_parts)
    
    def remove(self, entry_id: str) -> bool:
        """Remove an entry from memory."""
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            self._content_hashes.discard(entry.content_hash())
            del self.entries[entry_id]
            if self.storage_path:
                self._save()
            return True
        return False
    
    def clear(self, keep_preferences: bool = True) -> None:
        """Clear conversation memory."""
        if keep_preferences:
            preferences = {
                k: v for k, v in self.entries.items()
                if v.entry_type == ConversationEntryType.USER_PREFERENCE
            }
            self.entries = preferences
        else:
            self.entries.clear()
        
        self._content_hashes = {e.content_hash() for e in self.entries.values()}
        
        if self.storage_path:
            self._save()
    
    def _enforce_limit(self) -> None:
        """Remove old entries to stay within limit."""
        if len(self.entries) <= self.max_entries:
            return
        
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda x: (
                x.entry_type == ConversationEntryType.USER_PREFERENCE,
                list(Importance).index(x.importance),
                x.access_count,
                x.created_at.timestamp() if x.created_at else 0,
            ),
        )
        
        to_remove = len(self.entries) - self.max_entries
        for entry in sorted_entries[:to_remove]:
            self._content_hashes.discard(entry.content_hash())
            del self.entries[entry.id]
    
    def get_statistics(self) -> dict:
        """Get statistics about conversation memory."""
        by_type = {}
        for entry_type in ConversationEntryType:
            by_type[entry_type.value] = len(self.get_by_type(entry_type))
        
        by_importance = {}
        for importance in Importance:
            by_importance[importance.value] = len([
                e for e in self.entries.values() if e.importance == importance
            ])
        
        total_accesses = sum(e.access_count for e in self.entries.values())
        
        return {
            "total_entries": len(self.entries),
            "max_entries": self.max_entries,
            "by_type": by_type,
            "by_importance": by_importance,
            "total_accesses": total_accesses,
            "has_embeddings": self.embedding_fn is not None,
        }
    
    def _save(self) -> None:
        """Save memory to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "max_entries": self.max_entries,
            "entries": [e.to_dict() for e in self.entries.values()],
        }
        
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load memory from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            self.max_entries = data.get("max_entries", self.max_entries)
            
            for entry_data in data.get("entries", []):
                entry = ConversationEntry.from_dict(entry_data)
                self.entries[entry.id] = entry
                self._content_hashes.add(entry.content_hash())
        except Exception:
            pass
    
    def export(self) -> dict:
        """Export memory to dictionary."""
        return {
            "statistics": self.get_statistics(),
            "entries": [e.to_dict() for e in self.entries.values()],
        }
    
    def import_entries(self, entries: list[dict]) -> int:
        """Import entries from a list of dictionaries."""
        imported = 0
        for entry_data in entries:
            try:
                entry = ConversationEntry.from_dict(entry_data)
                if self.add(entry):
                    imported += 1
            except Exception:
                continue
        return imported


def create_conversation_memory(
    storage_path: Optional[str] = None,
    max_entries: int = 10000,
    embedding_fn: Optional[callable] = None,
) -> ConversationMemory:
    """Create a new conversation memory instance."""
    return ConversationMemory(
        storage_path=storage_path,
        max_entries=max_entries,
        embedding_fn=embedding_fn,
    )
