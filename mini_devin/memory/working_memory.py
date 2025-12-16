"""
Working Memory for Mini-Devin Agent

This module provides working memory management for the agent's current task context.
It maintains a structured view of:
- Current plan and constraints
- Active file excerpts
- Latest tool outputs
- Decisions made and lessons learned
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class MemoryItemType(str, Enum):
    """Types of items in working memory."""
    PLAN = "plan"
    CONSTRAINT = "constraint"
    FILE_EXCERPT = "file_excerpt"
    TOOL_OUTPUT = "tool_output"
    DECISION = "decision"
    LESSON = "lesson"
    ERROR = "error"
    CONTEXT = "context"
    GOAL = "goal"


class MemoryPriority(str, Enum):
    """Priority levels for memory items."""
    CRITICAL = "critical"  # Must always be in context
    HIGH = "high"  # Should be in context if possible
    MEDIUM = "medium"  # Include if space allows
    LOW = "low"  # Can be dropped if needed


@dataclass
class MemoryItem:
    """An item in working memory."""
    id: str
    item_type: MemoryItemType
    content: str
    priority: MemoryPriority = MemoryPriority.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    token_estimate: int = 0
    
    def __post_init__(self):
        # Estimate tokens (rough approximation: 4 chars per token)
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4
    
    def is_expired(self) -> bool:
        """Check if the item has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "item_type": self.item_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "token_estimate": self.token_estimate,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            item_type=MemoryItemType(data["item_type"]),
            content=data["content"],
            priority=MemoryPriority(data.get("priority", "medium")),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            token_estimate=data.get("token_estimate", 0),
        )


@dataclass
class FileExcerpt:
    """An excerpt from a file in working memory."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str = ""
    relevance: str = ""
    
    def to_memory_item(self, item_id: str, priority: MemoryPriority = MemoryPriority.MEDIUM) -> MemoryItem:
        """Convert to a MemoryItem."""
        return MemoryItem(
            id=item_id,
            item_type=MemoryItemType.FILE_EXCERPT,
            content=self.content,
            priority=priority,
            metadata={
                "file_path": self.file_path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "language": self.language,
                "relevance": self.relevance,
            },
        )


@dataclass
class ToolOutput:
    """Output from a tool execution."""
    tool_name: str
    input_summary: str
    output: str
    success: bool
    duration_ms: int = 0
    
    def to_memory_item(self, item_id: str, priority: MemoryPriority = MemoryPriority.MEDIUM) -> MemoryItem:
        """Convert to a MemoryItem."""
        return MemoryItem(
            id=item_id,
            item_type=MemoryItemType.TOOL_OUTPUT,
            content=self.output,
            priority=priority,
            metadata={
                "tool_name": self.tool_name,
                "input_summary": self.input_summary,
                "success": self.success,
                "duration_ms": self.duration_ms,
            },
        )


class WorkingMemory:
    """
    Working memory for the agent's current task.
    
    Manages a structured set of memory items with:
    - Priority-based retention
    - Token budget management
    - Automatic expiration
    - Context window optimization
    """
    
    def __init__(
        self,
        max_tokens: int = 8000,
        persist_path: Optional[str] = None,
    ):
        self.max_tokens = max_tokens
        self.persist_path = Path(persist_path) if persist_path else None
        self.items: dict[str, MemoryItem] = {}
        self._item_counter = 0
        
        # Load existing data if persist path exists
        if self.persist_path and self.persist_path.exists():
            self._load()
    
    def _generate_id(self, prefix: str = "mem") -> str:
        """Generate a unique ID for a memory item."""
        self._item_counter += 1
        return f"{prefix}_{self._item_counter}"
    
    def add(self, item: MemoryItem) -> str:
        """
        Add an item to working memory.
        
        Args:
            item: The memory item to add
            
        Returns:
            The item ID
        """
        # Generate ID if not set
        if not item.id:
            item.id = self._generate_id(item.item_type.value)
        
        self.items[item.id] = item
        
        # Enforce token budget
        self._enforce_budget()
        
        # Persist if configured
        if self.persist_path:
            self._save()
        
        return item.id
    
    def add_plan(self, plan: str, priority: MemoryPriority = MemoryPriority.HIGH) -> str:
        """Add a plan to working memory."""
        item = MemoryItem(
            id=self._generate_id("plan"),
            item_type=MemoryItemType.PLAN,
            content=plan,
            priority=priority,
        )
        return self.add(item)
    
    def add_constraint(self, constraint: str, priority: MemoryPriority = MemoryPriority.CRITICAL) -> str:
        """Add a constraint to working memory."""
        item = MemoryItem(
            id=self._generate_id("constraint"),
            item_type=MemoryItemType.CONSTRAINT,
            content=constraint,
            priority=priority,
        )
        return self.add(item)
    
    def add_file_excerpt(
        self,
        file_path: str,
        content: str,
        start_line: int = 1,
        end_line: int = 0,
        language: str = "",
        relevance: str = "",
        priority: MemoryPriority = MemoryPriority.MEDIUM,
    ) -> str:
        """Add a file excerpt to working memory."""
        if end_line == 0:
            end_line = start_line + content.count("\n")
        
        excerpt = FileExcerpt(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            language=language,
            relevance=relevance,
        )
        
        item = excerpt.to_memory_item(self._generate_id("file"), priority)
        return self.add(item)
    
    def add_tool_output(
        self,
        tool_name: str,
        input_summary: str,
        output: str,
        success: bool,
        duration_ms: int = 0,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
    ) -> str:
        """Add a tool output to working memory."""
        tool_output = ToolOutput(
            tool_name=tool_name,
            input_summary=input_summary,
            output=output,
            success=success,
            duration_ms=duration_ms,
        )
        
        item = tool_output.to_memory_item(self._generate_id("tool"), priority)
        return self.add(item)
    
    def add_decision(self, decision: str, reasoning: str = "", priority: MemoryPriority = MemoryPriority.HIGH) -> str:
        """Add a decision to working memory."""
        item = MemoryItem(
            id=self._generate_id("decision"),
            item_type=MemoryItemType.DECISION,
            content=decision,
            priority=priority,
            metadata={"reasoning": reasoning},
        )
        return self.add(item)
    
    def add_lesson(self, lesson: str, context: str = "", priority: MemoryPriority = MemoryPriority.HIGH) -> str:
        """Add a lesson learned to working memory."""
        item = MemoryItem(
            id=self._generate_id("lesson"),
            item_type=MemoryItemType.LESSON,
            content=lesson,
            priority=priority,
            metadata={"context": context},
        )
        return self.add(item)
    
    def add_error(self, error: str, context: str = "", priority: MemoryPriority = MemoryPriority.HIGH) -> str:
        """Add an error to working memory."""
        item = MemoryItem(
            id=self._generate_id("error"),
            item_type=MemoryItemType.ERROR,
            content=error,
            priority=priority,
            metadata={"context": context},
        )
        return self.add(item)
    
    def add_goal(self, goal: str, priority: MemoryPriority = MemoryPriority.CRITICAL) -> str:
        """Add a goal to working memory."""
        item = MemoryItem(
            id=self._generate_id("goal"),
            item_type=MemoryItemType.GOAL,
            content=goal,
            priority=priority,
        )
        return self.add(item)
    
    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get an item by ID."""
        return self.items.get(item_id)
    
    def remove(self, item_id: str) -> bool:
        """Remove an item from working memory."""
        if item_id in self.items:
            del self.items[item_id]
            if self.persist_path:
                self._save()
            return True
        return False
    
    def get_by_type(self, item_type: MemoryItemType) -> list[MemoryItem]:
        """Get all items of a specific type."""
        return [item for item in self.items.values() if item.item_type == item_type]
    
    def get_by_priority(self, priority: MemoryPriority) -> list[MemoryItem]:
        """Get all items with a specific priority."""
        return [item for item in self.items.values() if item.priority == priority]
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get the current context as a formatted string.
        
        Args:
            max_tokens: Maximum tokens to include (uses self.max_tokens if not specified)
            
        Returns:
            Formatted context string
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # Sort items by priority and recency
        sorted_items = sorted(
            self.items.values(),
            key=lambda x: (
                -list(MemoryPriority).index(x.priority),  # Higher priority first
                -x.created_at.timestamp(),  # More recent first
            ),
        )
        
        # Build context within token budget
        context_parts = []
        current_tokens = 0
        
        for item in sorted_items:
            if item.is_expired():
                continue
            
            if current_tokens + item.token_estimate > max_tokens:
                # Only skip if not critical
                if item.priority != MemoryPriority.CRITICAL:
                    continue
            
            # Format the item
            formatted = self._format_item(item)
            context_parts.append(formatted)
            current_tokens += item.token_estimate
        
        return "\n\n".join(context_parts)
    
    def _format_item(self, item: MemoryItem) -> str:
        """Format a memory item for context."""
        type_labels = {
            MemoryItemType.PLAN: "Current Plan",
            MemoryItemType.CONSTRAINT: "Constraint",
            MemoryItemType.FILE_EXCERPT: "File",
            MemoryItemType.TOOL_OUTPUT: "Tool Output",
            MemoryItemType.DECISION: "Decision",
            MemoryItemType.LESSON: "Lesson Learned",
            MemoryItemType.ERROR: "Error",
            MemoryItemType.CONTEXT: "Context",
            MemoryItemType.GOAL: "Goal",
        }
        
        label = type_labels.get(item.item_type, item.item_type.value)
        
        if item.item_type == MemoryItemType.FILE_EXCERPT:
            file_path = item.metadata.get("file_path", "unknown")
            start_line = item.metadata.get("start_line", 1)
            end_line = item.metadata.get("end_line", start_line)
            return f"[{label}: {file_path}:{start_line}-{end_line}]\n{item.content}"
        
        elif item.item_type == MemoryItemType.TOOL_OUTPUT:
            tool_name = item.metadata.get("tool_name", "unknown")
            success = "Success" if item.metadata.get("success", True) else "Failed"
            return f"[{label}: {tool_name} ({success})]\n{item.content}"
        
        elif item.item_type == MemoryItemType.DECISION:
            reasoning = item.metadata.get("reasoning", "")
            if reasoning:
                return f"[{label}]\n{item.content}\nReasoning: {reasoning}"
            return f"[{label}]\n{item.content}"
        
        else:
            return f"[{label}]\n{item.content}"
    
    def _enforce_budget(self) -> None:
        """Remove low-priority items to stay within token budget."""
        total_tokens = sum(item.token_estimate for item in self.items.values())
        
        if total_tokens <= self.max_tokens:
            return
        
        # Sort by priority (lowest first) and age (oldest first)
        sorted_items = sorted(
            self.items.values(),
            key=lambda x: (
                list(MemoryPriority).index(x.priority),  # Lower priority first
                x.created_at.timestamp(),  # Older first
            ),
        )
        
        # Remove items until within budget
        for item in sorted_items:
            if item.priority == MemoryPriority.CRITICAL:
                continue  # Never remove critical items
            
            if total_tokens <= self.max_tokens:
                break
            
            total_tokens -= item.token_estimate
            del self.items[item.id]
    
    def cleanup_expired(self) -> int:
        """Remove expired items. Returns count of removed items."""
        expired_ids = [
            item_id for item_id, item in self.items.items()
            if item.is_expired()
        ]
        
        for item_id in expired_ids:
            del self.items[item_id]
        
        if expired_ids and self.persist_path:
            self._save()
        
        return len(expired_ids)
    
    def clear(self, keep_critical: bool = True) -> None:
        """Clear working memory."""
        if keep_critical:
            self.items = {
                item_id: item for item_id, item in self.items.items()
                if item.priority == MemoryPriority.CRITICAL
            }
        else:
            self.items.clear()
        
        if self.persist_path:
            self._save()
    
    def get_statistics(self) -> dict:
        """Get statistics about working memory."""
        total_tokens = sum(item.token_estimate for item in self.items.values())
        
        by_type = {}
        for item_type in MemoryItemType:
            items = self.get_by_type(item_type)
            by_type[item_type.value] = len(items)
        
        by_priority = {}
        for priority in MemoryPriority:
            items = self.get_by_priority(priority)
            by_priority[priority.value] = len(items)
        
        return {
            "total_items": len(self.items),
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": total_tokens / self.max_tokens if self.max_tokens > 0 else 0,
            "by_type": by_type,
            "by_priority": by_priority,
        }
    
    def _save(self) -> None:
        """Save working memory to disk."""
        if not self.persist_path:
            return
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "max_tokens": self.max_tokens,
            "item_counter": self._item_counter,
            "items": [item.to_dict() for item in self.items.values()],
        }
        
        self.persist_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load working memory from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            self.max_tokens = data.get("max_tokens", self.max_tokens)
            self._item_counter = data.get("item_counter", 0)
            
            for item_data in data.get("items", []):
                item = MemoryItem.from_dict(item_data)
                self.items[item.id] = item
        except Exception:
            pass
    
    def to_dict(self) -> dict:
        """Export working memory to dictionary."""
        return {
            "statistics": self.get_statistics(),
            "items": [item.to_dict() for item in self.items.values()],
        }


def create_working_memory(
    max_tokens: int = 8000,
    persist_path: Optional[str] = None,
) -> WorkingMemory:
    """Create a new working memory instance."""
    return WorkingMemory(max_tokens=max_tokens, persist_path=persist_path)
