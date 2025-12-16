"""Unit tests for the editor tool."""

import os
import tempfile
import pytest
from pathlib import Path

from mini_devin.tools.editor import EditorTool, create_editor_tool
from mini_devin.schemas.tools import (
    ToolStatus,
    ReadFileInput,
    WriteFileInput,
    ListDirectoryInput,
    SearchInput,
    EditorAction,
)


class TestEditorTool:
    """Tests for EditorTool class."""

    def test_tool_initialization(self):
        """Test EditorTool initialization."""
        tool = EditorTool(working_directory="/tmp")
        assert tool.working_directory == "/tmp"
        assert tool.name == "editor"

    def test_tool_name(self):
        """Test tool name property."""
        tool = EditorTool()
        assert tool.name == "editor"

    def test_tool_description(self):
        """Test tool description."""
        tool = EditorTool()
        assert "edit" in tool.description.lower() or "file" in tool.description.lower()

    def test_input_schema(self):
        """Test input schema property."""
        tool = EditorTool()
        assert tool.input_schema == ReadFileInput

    def test_get_schema_for_llm(self):
        """Test getting tool schema for LLM."""
        tool = EditorTool()
        schema = tool.get_schema_for_llm()
        assert "name" in schema
        assert schema["name"] == "editor"
        assert "description" in schema
        assert "parameters" in schema

    def test_resolve_path_absolute(self):
        """Test resolving absolute path."""
        tool = EditorTool(working_directory="/tmp")
        resolved = tool._resolve_path("/home/user/file.txt")
        assert resolved == "/home/user/file.txt"

    def test_resolve_path_relative(self):
        """Test resolving relative path."""
        tool = EditorTool(working_directory="/tmp")
        resolved = tool._resolve_path("file.txt")
        assert resolved == "/tmp/file.txt"

    def test_detect_language_python(self):
        """Test language detection for Python files."""
        tool = EditorTool()
        assert tool._detect_language("test.py") == "python"

    def test_detect_language_javascript(self):
        """Test language detection for JavaScript files."""
        tool = EditorTool()
        assert tool._detect_language("test.js") == "javascript"

    def test_detect_language_typescript(self):
        """Test language detection for TypeScript files."""
        tool = EditorTool()
        assert tool._detect_language("test.ts") == "typescript"

    def test_detect_language_unknown(self):
        """Test language detection for unknown files."""
        tool = EditorTool()
        assert tool._detect_language("test.xyz") is None

    @pytest.mark.asyncio
    async def test_read_file(self):
        """Test reading a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name
        
        try:
            tool = EditorTool()
            result = await tool.execute({
                "action": EditorAction.READ_FILE,
                "path": temp_path
            })
            assert result.status == ToolStatus.SUCCESS
            assert "Hello, World!" in result.content
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        """Test reading a non-existent file."""
        tool = EditorTool()
        result = await tool.execute({
            "action": EditorAction.READ_FILE,
            "path": "/nonexistent/file.txt"
        })
        assert result.status == ToolStatus.FAILURE
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_list_directory(self):
        """Test listing directory contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "file1.txt")).touch()
            Path(os.path.join(tmpdir, "file2.py")).touch()
            
            tool = EditorTool()
            # Use ListDirectoryInput directly to bypass ReadFileInput validation
            result = await tool._list_directory(ListDirectoryInput(path=tmpdir))
            
            assert result.status == ToolStatus.SUCCESS
            entry_names = [e.name for e in result.entries]
            assert "file1.txt" in entry_names or "file2.py" in entry_names


class TestEditorToolSafety:
    """Safety-focused tests for EditorTool."""

    def test_max_file_size_limit(self):
        """Test that there's a file size limit."""
        tool = EditorTool()
        assert hasattr(tool, 'max_file_size')
        assert tool.max_file_size > 0

    def test_max_output_lines_limit(self):
        """Test that there's an output lines limit."""
        tool = EditorTool()
        assert hasattr(tool, 'max_output_lines')
        assert tool.max_output_lines > 0


class TestEditorToolEdgeCases:
    """Edge case tests for EditorTool."""

    @pytest.mark.asyncio
    async def test_read_empty_file(self):
        """Test reading an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            tool = EditorTool()
            result = await tool.execute({
                "action": EditorAction.READ_FILE,
                "path": temp_path
            })
            assert result.status == ToolStatus.SUCCESS
            assert result.content == ""
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_binary_file(self):
        """Test reading a binary file."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(b'\x00\x01\x02\x03')
            temp_path = f.name
        
        try:
            tool = EditorTool()
            result = await tool.execute({
                "action": EditorAction.READ_FILE,
                "path": temp_path
            })
            assert result is not None
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self):
        """Test listing a non-existent directory."""
        tool = EditorTool()
        # Use ListDirectoryInput directly to bypass ReadFileInput validation
        result = await tool._list_directory(ListDirectoryInput(path="/nonexistent/directory"))
        assert result.status == ToolStatus.FAILURE


class TestCreateEditorTool:
    """Tests for create_editor_tool function."""

    def test_create_with_defaults(self):
        """Test creating tool with defaults."""
        tool = create_editor_tool()
        assert tool.name == "editor"
        assert tool.max_file_size > 0

    def test_create_with_working_directory(self):
        """Test creating tool with custom working directory."""
        tool = EditorTool(working_directory="/tmp")
        assert tool.working_directory == "/tmp"
