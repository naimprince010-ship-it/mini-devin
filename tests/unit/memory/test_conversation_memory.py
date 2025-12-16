"""Unit tests for the conversation memory module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from mini_devin.memory.conversation_memory import (
    ConversationEntryType,
    Importance,
    ConversationEntry,
    TaskSummary,
    ConversationMemory,
    create_conversation_memory,
)


class TestConversationEntry:
    """Tests for ConversationEntry dataclass."""
    
    def test_create_entry(self):
        """Test creating a conversation entry."""
        entry = ConversationEntry(
            id="test1",
            entry_type=ConversationEntryType.LESSON_LEARNED,
            content="Always run tests before committing",
            summary="Run tests first",
        )
        assert entry.id == "test1"
        assert entry.entry_type == ConversationEntryType.LESSON_LEARNED
        assert entry.content == "Always run tests before committing"
        assert entry.summary == "Run tests first"
        assert entry.importance == Importance.MEDIUM
        assert entry.tags == []
        assert entry.access_count == 0
    
    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = ConversationEntry(
            id="test1",
            entry_type=ConversationEntryType.LESSON_LEARNED,
            content="Test content",
            summary="Test summary",
            tags=["test", "unit"],
        )
        data = entry.to_dict()
        assert data["id"] == "test1"
        assert data["entry_type"] == "lesson_learned"
        assert data["content"] == "Test content"
        assert data["tags"] == ["test", "unit"]
    
    def test_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            "id": "test1",
            "entry_type": "lesson_learned",
            "content": "Test content",
            "summary": "Test summary",
            "importance": "high",
            "tags": ["test"],
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        entry = ConversationEntry.from_dict(data)
        assert entry.id == "test1"
        assert entry.entry_type == ConversationEntryType.LESSON_LEARNED
        assert entry.importance == Importance.HIGH
        assert entry.tags == ["test"]
    
    def test_content_hash(self):
        """Test content hash for deduplication."""
        entry1 = ConversationEntry(
            id="1", entry_type=ConversationEntryType.LESSON_LEARNED,
            content="Same content", summary="s1"
        )
        entry2 = ConversationEntry(
            id="2", entry_type=ConversationEntryType.LESSON_LEARNED,
            content="Same content", summary="s2"
        )
        entry3 = ConversationEntry(
            id="3", entry_type=ConversationEntryType.LESSON_LEARNED,
            content="Different content", summary="s3"
        )
        assert entry1.content_hash() == entry2.content_hash()
        assert entry1.content_hash() != entry3.content_hash()


class TestTaskSummary:
    """Tests for TaskSummary dataclass."""
    
    def test_create_task_summary(self):
        """Test creating a task summary."""
        summary = TaskSummary(
            task_id="task1",
            session_id="session1",
            description="Fix bug in login",
            outcome="Bug fixed successfully",
            success=True,
            duration_seconds=300,
            tools_used=["terminal", "editor"],
            files_modified=["auth.py"],
            errors_encountered=[],
            lessons=["Check error logs first"],
        )
        assert summary.task_id == "task1"
        assert summary.success
        assert summary.duration_seconds == 300
    
    def test_to_entry(self):
        """Test converting task summary to entry."""
        summary = TaskSummary(
            task_id="task1",
            session_id="session1",
            description="Fix bug",
            outcome="Fixed",
            success=True,
            duration_seconds=60,
            tools_used=["terminal"],
            files_modified=["test.py"],
            errors_encountered=[],
            lessons=["Test first"],
        )
        entry = summary.to_entry()
        assert entry.entry_type == ConversationEntryType.TASK_SUMMARY
        assert entry.importance == Importance.HIGH
        assert "task" in entry.tags
        assert "success" in entry.tags


class TestConversationMemory:
    """Tests for ConversationMemory class."""
    
    @pytest.fixture
    def memory(self):
        """Create a conversation memory instance."""
        return ConversationMemory(max_entries=100)
    
    @pytest.fixture
    def memory_with_storage(self, tmp_path):
        """Create a conversation memory with file storage."""
        storage_path = tmp_path / "memory.json"
        return ConversationMemory(storage_path=str(storage_path), max_entries=100)
    
    def test_add_entry(self, memory):
        """Test adding an entry."""
        entry = ConversationEntry(
            id="test1",
            entry_type=ConversationEntryType.LESSON_LEARNED,
            content="Test lesson",
            summary="Test",
        )
        entry_id = memory.add(entry)
        assert entry_id == "test1"
        assert len(memory.entries) == 1
    
    def test_add_deduplication(self, memory):
        """Test that duplicate content is deduplicated."""
        entry1 = ConversationEntry(
            id="1", entry_type=ConversationEntryType.LESSON_LEARNED,
            content="Same content", summary="s1"
        )
        entry2 = ConversationEntry(
            id="2", entry_type=ConversationEntryType.LESSON_LEARNED,
            content="Same content", summary="s2"
        )
        memory.add(entry1)
        result = memory.add(entry2)
        assert result is None
        assert len(memory.entries) == 1
    
    def test_add_lesson(self, memory):
        """Test adding a lesson."""
        entry_id = memory.add_lesson(
            lesson="Always test first",
            context="After a bug was introduced",
            tags=["testing"],
        )
        assert entry_id is not None
        entry = memory.get(entry_id)
        assert entry.entry_type == ConversationEntryType.LESSON_LEARNED
        assert "lesson" in entry.tags
    
    def test_add_error_pattern(self, memory):
        """Test adding an error pattern."""
        entry_id = memory.add_error_pattern(
            error="ModuleNotFoundError",
            cause="Missing dependency",
            solution="Run pip install",
            tags=["python"],
        )
        assert entry_id is not None
        entry = memory.get(entry_id)
        assert entry.entry_type == ConversationEntryType.ERROR_PATTERN
        assert entry.metadata["solution"] == "Run pip install"
    
    def test_add_solution_pattern(self, memory):
        """Test adding a solution pattern."""
        entry_id = memory.add_solution_pattern(
            problem="Slow database queries",
            solution="Add index to column",
            code_example="CREATE INDEX idx_name ON table(column);",
            tags=["database"],
        )
        assert entry_id is not None
        entry = memory.get(entry_id)
        assert entry.entry_type == ConversationEntryType.SOLUTION_PATTERN
        assert "CREATE INDEX" in entry.content
    
    def test_add_user_preference(self, memory):
        """Test adding a user preference."""
        entry_id = memory.add_user_preference(
            preference="code_style",
            value="black",
            context="Python formatting",
        )
        assert entry_id is not None
        entry = memory.get(entry_id)
        assert entry.entry_type == ConversationEntryType.USER_PREFERENCE
        assert entry.importance == Importance.CRITICAL
    
    def test_user_preference_update(self, memory):
        """Test that user preferences are updated, not duplicated."""
        memory.add_user_preference("style", "black")
        memory.add_user_preference("style", "ruff")
        
        prefs = memory.get_user_preferences()
        assert prefs["style"] == "ruff"
        assert len(memory.get_by_type(ConversationEntryType.USER_PREFERENCE)) == 1
    
    def test_add_feedback(self, memory):
        """Test adding feedback."""
        entry_id = memory.add_feedback(
            feedback="Great job on the refactoring",
            rating=5,
            context="Code review",
        )
        assert entry_id is not None
        entry = memory.get(entry_id)
        assert entry.entry_type == ConversationEntryType.FEEDBACK
        assert entry.metadata["rating"] == 5
    
    def test_add_task_summary(self, memory):
        """Test adding a task summary."""
        summary = TaskSummary(
            task_id="task1",
            session_id="session1",
            description="Fix bug",
            outcome="Fixed",
            success=True,
            duration_seconds=60,
            tools_used=["terminal"],
            files_modified=["test.py"],
            errors_encountered=[],
            lessons=[],
        )
        entry_id = memory.add_task_summary(summary)
        assert entry_id is not None
        assert "task_task1" in entry_id
    
    def test_search(self, memory):
        """Test searching memory."""
        memory.add_lesson("Always run tests", "Testing context", tags=["testing"])
        memory.add_lesson("Use type hints", "Python context", tags=["python"])
        memory.add_lesson("Write documentation", "Docs context", tags=["docs"])
        
        results = memory.search("tests testing")
        assert len(results) >= 1
        assert any("test" in r.content.lower() for r in results)
    
    def test_search_by_type(self, memory):
        """Test searching by entry type."""
        memory.add_lesson("Lesson 1", "context")
        memory.add_error_pattern("Error", "Cause", "Solution")
        
        results = memory.search(
            "lesson error",
            entry_types=[ConversationEntryType.LESSON_LEARNED],
        )
        assert all(r.entry_type == ConversationEntryType.LESSON_LEARNED for r in results)
    
    def test_search_by_tags(self, memory):
        """Test searching by tags."""
        memory.add_lesson("Python lesson", "context", tags=["python"])
        memory.add_lesson("JavaScript lesson", "context", tags=["javascript"])
        
        results = memory.search("lesson", tags=["python"])
        assert all("python" in r.tags for r in results)
    
    def test_get_by_type(self, memory):
        """Test getting entries by type."""
        memory.add_lesson("Lesson 1", "context")
        memory.add_lesson("Lesson 2", "context")
        memory.add_error_pattern("Error", "Cause", "Solution")
        
        lessons = memory.get_by_type(ConversationEntryType.LESSON_LEARNED)
        assert len(lessons) == 2
    
    def test_get_by_session(self, memory):
        """Test getting entries by session."""
        memory.add_lesson("Lesson 1", "context", session_id="s1")
        memory.add_lesson("Lesson 2", "context", session_id="s2")
        
        results = memory.get_by_session("s1")
        assert len(results) == 1
    
    def test_get_error_solutions(self, memory):
        """Test getting solutions for errors."""
        memory.add_error_pattern(
            "ModuleNotFoundError: numpy",
            "Missing package",
            "pip install numpy",
        )
        memory.add_error_pattern(
            "ModuleNotFoundError: pandas",
            "Missing package",
            "pip install pandas",
        )
        
        solutions = memory.get_error_solutions("ModuleNotFoundError")
        assert len(solutions) >= 1
        assert any("pip install" in s for s in solutions)
    
    def test_get_recent_lessons(self, memory):
        """Test getting recent lessons."""
        memory.add_lesson("Lesson 1", "context")
        memory.add_lesson("Lesson 2", "context")
        memory.add_lesson("Lesson 3", "context")
        
        lessons = memory.get_recent_lessons(limit=2)
        assert len(lessons) == 2
    
    def test_get_context_for_task(self, memory):
        """Test getting context for a new task."""
        memory.add_lesson("Always check imports", "Python debugging")
        memory.add_error_pattern(
            "ImportError",
            "Missing module",
            "Install the package",
        )
        
        context = memory.get_context_for_task("Fix import error in Python")
        assert len(context) > 0
        assert "Relevant Past Experience" in context
    
    def test_remove(self, memory):
        """Test removing an entry."""
        entry_id = memory.add_lesson("Test lesson", "context")
        assert memory.remove(entry_id)
        assert memory.get(entry_id) is None
    
    def test_clear(self, memory):
        """Test clearing memory."""
        memory.add_lesson("Lesson", "context")
        memory.add_user_preference("pref", "value")
        
        memory.clear(keep_preferences=True)
        
        assert len(memory.get_by_type(ConversationEntryType.LESSON_LEARNED)) == 0
        assert len(memory.get_by_type(ConversationEntryType.USER_PREFERENCE)) == 1
    
    def test_enforce_limit(self, memory):
        """Test that entry limit is enforced."""
        memory.max_entries = 5
        
        for i in range(10):
            memory.add_lesson(f"Lesson {i}", "context", importance=Importance.LOW)
        
        assert len(memory.entries) <= 5
    
    def test_statistics(self, memory):
        """Test getting statistics."""
        memory.add_lesson("Lesson", "context")
        memory.add_error_pattern("Error", "Cause", "Solution")
        
        stats = memory.get_statistics()
        assert stats["total_entries"] == 2
        assert stats["by_type"]["lesson_learned"] == 1
        assert stats["by_type"]["error_pattern"] == 1
    
    def test_persistence(self, memory_with_storage):
        """Test saving and loading from disk."""
        memory_with_storage.add_lesson("Persistent lesson", "context")
        storage_path = memory_with_storage.storage_path
        
        new_memory = ConversationMemory(storage_path=str(storage_path))
        
        assert len(new_memory.entries) == 1
        lessons = new_memory.get_by_type(ConversationEntryType.LESSON_LEARNED)
        assert len(lessons) == 1
        assert "Persistent" in lessons[0].content
    
    def test_export_import(self, memory):
        """Test exporting and importing entries."""
        memory.add_lesson("Lesson 1", "context")
        memory.add_lesson("Lesson 2", "context")
        
        exported = memory.export()
        assert len(exported["entries"]) == 2
        
        new_memory = ConversationMemory()
        imported = new_memory.import_entries(exported["entries"])
        assert imported == 2
        assert len(new_memory.entries) == 2


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_conversation_memory(self):
        """Test create_conversation_memory factory."""
        memory = create_conversation_memory(max_entries=500)
        assert isinstance(memory, ConversationMemory)
        assert memory.max_entries == 500
    
    def test_create_with_storage(self, tmp_path):
        """Test creating with storage path."""
        storage_path = tmp_path / "memory.json"
        memory = create_conversation_memory(storage_path=str(storage_path))
        assert memory.storage_path == storage_path
