"""
Editor Tool for Mini-Devin

This module implements the code editor tool that allows the agent to
read, write, search, and modify files in the codebase.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Any

from ..core.tool_interface import BaseTool, ToolPolicy
from ..schemas.tools import (
    EditorAction,
    ReadFileInput,
    ReadFileOutput,
    WriteFileInput,
    WriteFileOutput,
    ApplyPatchInput,
    ApplyPatchOutput,
    SearchInput,
    SearchMatch,
    SearchOutput,
    ListDirectoryInput,
    FileInfo,
    ListDirectoryOutput,
    ToolStatus,
)


class EditorTool(BaseTool[ReadFileInput, ReadFileOutput]):
    """
    Editor tool for file operations.
    
    This is a multi-action tool that supports:
    - read_file: Read file contents with optional line range
    - write_file: Write content to a file
    - apply_patch: Apply a unified diff patch to a file
    - search: Search for patterns in files using ripgrep
    - list_directory: List directory contents
    
    The tool dispatches to the appropriate handler based on the action field.
    """
    
    def __init__(
        self,
        policy: ToolPolicy | None = None,
        working_directory: str | None = None,
        max_file_size: int = 1_000_000,  # 1MB
        max_output_lines: int = 2000,
    ):
        super().__init__(policy)
        self.working_directory = working_directory or os.getcwd()
        self.max_file_size = max_file_size
        self.max_output_lines = max_output_lines
    
    @property
    def name(self) -> str:
        return "editor"
    
    @property
    def description(self) -> str:
        return """Read, write, search, and modify files in the codebase.

Actions:
- read_file: Read file contents (optionally specify line range)
- write_file: Write content to a file (creates directories if needed)
- apply_patch: Apply a unified diff patch to modify a file
- search: Search for patterns in files using regex
- list_directory: List directory contents

Use this tool for all file operations during development."""
    
    @property
    def input_schema(self) -> type[ReadFileInput]:
        # Return the base input type; actual dispatch happens in _execute
        return ReadFileInput
    
    @property
    def output_schema(self) -> type[ReadFileOutput]:
        return ReadFileOutput
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to working directory."""
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.working_directory, path))
    
    def _detect_language(self, path: str) -> str | None:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "zsh",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",
            ".md": "markdown",
            ".toml": "toml",
        }
        ext = os.path.splitext(path)[1].lower()
        return ext_map.get(ext)
    
    async def _read_file(self, input_data: ReadFileInput) -> ReadFileOutput:
        """Read file contents."""
        start_time = time.time()
        path = self._resolve_path(input_data.path)
        
        if not os.path.exists(path):
            return ReadFileOutput(
                status=ToolStatus.FAILURE,
                error_message=f"File not found: {path}",
                content="",
                total_lines=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        if not os.path.isfile(path):
            return ReadFileOutput(
                status=ToolStatus.FAILURE,
                error_message=f"Path is not a file: {path}",
                content="",
                total_lines=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        # Check file size
        file_size = os.path.getsize(path)
        if file_size > self.max_file_size:
            return ReadFileOutput(
                status=ToolStatus.FAILURE,
                error_message=f"File too large: {file_size} bytes (max {self.max_file_size})",
                content="",
                total_lines=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Apply line range if specified
            if input_data.line_range:
                start = input_data.line_range.start_line - 1  # Convert to 0-indexed
                end = input_data.line_range.end_line or total_lines
                lines = lines[start:end]
            
            # Truncate if too many lines
            if len(lines) > self.max_output_lines:
                lines = lines[:self.max_output_lines]
                lines.append(f"\n... [TRUNCATED: showing {self.max_output_lines} of {total_lines} lines]\n")
            
            content = "".join(lines)
            language = self._detect_language(path)
            
            return ReadFileOutput(
                status=ToolStatus.SUCCESS,
                content=content,
                total_lines=total_lines,
                language=language,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
            
        except Exception as e:
            return ReadFileOutput(
                status=ToolStatus.FAILURE,
                error_message=str(e),
                content="",
                total_lines=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
    
    async def _write_file(self, input_data: WriteFileInput) -> WriteFileOutput:
        """Write content to a file."""
        start_time = time.time()
        path = self._resolve_path(input_data.path)
        
        try:
            # Create directories if needed
            if input_data.create_directories:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(input_data.content)
            
            bytes_written = len(input_data.content.encode("utf-8"))
            
            return WriteFileOutput(
                status=ToolStatus.SUCCESS,
                bytes_written=bytes_written,
                path=path,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
            
        except Exception as e:
            return WriteFileOutput(
                status=ToolStatus.FAILURE,
                error_message=str(e),
                bytes_written=0,
                path=path,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
    
    async def _apply_patch(self, input_data: ApplyPatchInput) -> ApplyPatchOutput:
        """Apply a unified diff patch to a file."""
        start_time = time.time()
        path = self._resolve_path(input_data.path)
        
        if not os.path.exists(path):
            return ApplyPatchOutput(
                status=ToolStatus.FAILURE,
                error_message=f"File not found: {path}",
                hunks_applied=0,
                hunks_failed=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        try:
            # Read current content
            with open(path, "r", encoding="utf-8") as f:
                original_content = f.read()
            
            # Parse and apply the patch
            # This is a simplified patch application - for production, use a proper diff library
            new_content, hunks_applied, hunks_failed = self._apply_unified_diff(
                original_content, input_data.patch
            )
            
            if hunks_failed > 0 and hunks_applied == 0:
                return ApplyPatchOutput(
                    status=ToolStatus.FAILURE,
                    error_message=f"Failed to apply patch: {hunks_failed} hunks failed",
                    hunks_applied=hunks_applied,
                    hunks_failed=hunks_failed,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )
            
            if input_data.dry_run:
                return ApplyPatchOutput(
                    status=ToolStatus.SUCCESS,
                    hunks_applied=hunks_applied,
                    hunks_failed=hunks_failed,
                    resulting_content=new_content,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )
            
            # Write the patched content
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return ApplyPatchOutput(
                status=ToolStatus.SUCCESS,
                hunks_applied=hunks_applied,
                hunks_failed=hunks_failed,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
            
        except Exception as e:
            return ApplyPatchOutput(
                status=ToolStatus.FAILURE,
                error_message=str(e),
                hunks_applied=0,
                hunks_failed=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
    
    def _apply_unified_diff(self, content: str, patch: str) -> tuple[str, int, int]:
        """
        Apply a unified diff patch to content.
        
        Returns: (new_content, hunks_applied, hunks_failed)
        """
        lines = content.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        
        hunks_applied = 0
        hunks_failed = 0
        
        # Parse patch into hunks
        hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        
        patch_lines = patch.splitlines(keepends=True)
        i = 0
        offset = 0  # Track line number offset from applied changes
        
        while i < len(patch_lines):
            line = patch_lines[i]
            
            # Skip header lines
            if line.startswith("---") or line.startswith("+++") or line.startswith("diff"):
                i += 1
                continue
            
            # Look for hunk header
            match = hunk_pattern.match(line)
            if match:
                old_start = int(match.group(1)) - 1 + offset  # Convert to 0-indexed
                _old_count = int(match.group(2)) if match.group(2) else 1  # noqa: F841
                _new_start = int(match.group(3)) - 1  # noqa: F841
                _new_count = int(match.group(4)) if match.group(4) else 1  # noqa: F841
                
                i += 1
                
                # Collect hunk lines
                old_lines = []
                new_lines = []
                
                while i < len(patch_lines):
                    pline = patch_lines[i]
                    if pline.startswith("@@") or pline.startswith("diff"):
                        break
                    elif pline.startswith("-"):
                        old_lines.append(pline[1:])
                        i += 1
                    elif pline.startswith("+"):
                        new_lines.append(pline[1:])
                        i += 1
                    elif pline.startswith(" ") or pline == "\n":
                        context = pline[1:] if pline.startswith(" ") else pline
                        old_lines.append(context)
                        new_lines.append(context)
                        i += 1
                    else:
                        i += 1
                
                # Try to apply the hunk
                try:
                    # Remove old lines and insert new lines
                    del lines[old_start:old_start + len(old_lines)]
                    for j, new_line in enumerate(new_lines):
                        lines.insert(old_start + j, new_line)
                    
                    offset += len(new_lines) - len(old_lines)
                    hunks_applied += 1
                except Exception:
                    hunks_failed += 1
            else:
                i += 1
        
        return "".join(lines), hunks_applied, hunks_failed
    
    async def _search(self, input_data: SearchInput) -> SearchOutput:
        """Search for patterns in files using ripgrep or fallback to Python."""
        start_time = time.time()
        path = self._resolve_path(input_data.path)
        
        if not os.path.exists(path):
            return SearchOutput(
                status=ToolStatus.FAILURE,
                error_message=f"Path not found: {path}",
                matches=[],
                total_matches=0,
                files_searched=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        try:
            # Try to use ripgrep first (faster)
            matches, files_searched = await self._search_with_ripgrep(
                input_data.pattern,
                path,
                input_data.file_pattern,
                input_data.case_sensitive,
                input_data.max_results,
            )
        except Exception:
            # Fallback to Python-based search
            matches, files_searched = self._search_with_python(
                input_data.pattern,
                path,
                input_data.file_pattern,
                input_data.case_sensitive,
                input_data.max_results,
            )
        
        truncated = len(matches) >= input_data.max_results
        
        return SearchOutput(
            status=ToolStatus.SUCCESS,
            matches=matches,
            total_matches=len(matches),
            files_searched=files_searched,
            truncated=truncated,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )
    
    async def _search_with_ripgrep(
        self,
        pattern: str,
        path: str,
        file_pattern: str | None,
        case_sensitive: bool,
        max_results: int,
    ) -> tuple[list[SearchMatch], int]:
        """Search using ripgrep."""
        cmd = ["rg", "--json", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        if file_pattern:
            cmd.extend(["-g", file_pattern])
        
        cmd.extend(["-m", str(max_results), pattern, path])
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, _ = await process.communicate()
        
        matches = []
        files_seen = set()
        
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            try:
                data = json.loads(line)
                if data.get("type") == "match":
                    match_data = data["data"]
                    file_path = match_data["path"]["text"]
                    files_seen.add(file_path)
                    
                    for submatch in match_data.get("submatches", []):
                        matches.append(SearchMatch(
                            file_path=file_path,
                            line_number=match_data["line_number"],
                            line_content=match_data["lines"]["text"].rstrip("\n"),
                            match_start=submatch["start"],
                            match_end=submatch["end"],
                        ))
                        
                        if len(matches) >= max_results:
                            break
                            
            except json.JSONDecodeError:
                continue
            
            if len(matches) >= max_results:
                break
        
        return matches, len(files_seen)
    
    def _search_with_python(
        self,
        pattern: str,
        path: str,
        file_pattern: str | None,
        case_sensitive: bool,
        max_results: int,
    ) -> tuple[list[SearchMatch], int]:
        """Fallback Python-based search."""
        import fnmatch
        
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        matches = []
        files_searched = 0
        
        def search_file(filepath: str) -> None:
            nonlocal files_searched
            files_searched += 1
            
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        for match in regex.finditer(line):
                            matches.append(SearchMatch(
                                file_path=filepath,
                                line_number=line_num,
                                line_content=line.rstrip("\n"),
                                match_start=match.start(),
                                match_end=match.end(),
                            ))
                            if len(matches) >= max_results:
                                return
            except Exception:
                pass
        
        if os.path.isfile(path):
            search_file(path)
        else:
            for root, _, files in os.walk(path):
                if any(skip in root for skip in [".git", "node_modules", "__pycache__"]):
                    continue
                for filename in files:
                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue
                    filepath = os.path.join(root, filename)
                    search_file(filepath)
                    if len(matches) >= max_results:
                        break
                if len(matches) >= max_results:
                    break
        
        return matches, files_searched
    
    async def _list_directory(self, input_data: ListDirectoryInput) -> ListDirectoryOutput:
        """List directory contents."""
        start_time = time.time()
        path = self._resolve_path(input_data.path)
        
        if not os.path.exists(path):
            return ListDirectoryOutput(
                status=ToolStatus.FAILURE,
                error_message=f"Path not found: {path}",
                entries=[],
                total_files=0,
                total_directories=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        if not os.path.isdir(path):
            return ListDirectoryOutput(
                status=ToolStatus.FAILURE,
                error_message=f"Path is not a directory: {path}",
                entries=[],
                total_files=0,
                total_directories=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        entries = []
        total_files = 0
        total_directories = 0
        
        def scan_dir(dir_path: str, depth: int) -> None:
            nonlocal total_files, total_directories
            
            if depth > input_data.max_depth:
                return
            
            try:
                for entry in os.scandir(dir_path):
                    # Skip hidden files if not requested
                    if not input_data.include_hidden and entry.name.startswith("."):
                        continue
                    
                    try:
                        stat = entry.stat()
                        is_dir = entry.is_dir()
                        
                        if is_dir:
                            total_directories += 1
                        else:
                            total_files += 1
                        
                        entries.append(FileInfo(
                            name=entry.name,
                            path=entry.path,
                            is_directory=is_dir,
                            size_bytes=stat.st_size if not is_dir else None,
                            modified_time=datetime.fromtimestamp(stat.st_mtime),
                        ))
                        
                        # Recurse into directories
                        if is_dir and input_data.recursive:
                            scan_dir(entry.path, depth + 1)
                            
                    except OSError:
                        continue
                        
            except OSError:
                pass
        
        scan_dir(path, 1)
        
        # Sort entries: directories first, then by name
        entries.sort(key=lambda e: (not e.is_directory, e.name.lower()))
        
        return ListDirectoryOutput(
            status=ToolStatus.SUCCESS,
            entries=entries,
            total_files=total_files,
            total_directories=total_directories,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )
    
    async def _execute(self, input_data: Any) -> Any:
        """Dispatch to the appropriate handler based on action."""
        # Handle different input types
        if isinstance(input_data, ReadFileInput):
            return await self._read_file(input_data)
        elif isinstance(input_data, WriteFileInput):
            return await self._write_file(input_data)
        elif isinstance(input_data, ApplyPatchInput):
            return await self._apply_patch(input_data)
        elif isinstance(input_data, SearchInput):
            return await self._search(input_data)
        elif isinstance(input_data, ListDirectoryInput):
            return await self._list_directory(input_data)
        else:
            # Try to determine action from dict
            if isinstance(input_data, dict):
                action = input_data.get("action", EditorAction.READ_FILE)
                if action == EditorAction.READ_FILE:
                    return await self._read_file(ReadFileInput(**input_data))
                elif action == EditorAction.WRITE_FILE:
                    return await self._write_file(WriteFileInput(**input_data))
                elif action == EditorAction.APPLY_PATCH:
                    return await self._apply_patch(ApplyPatchInput(**input_data))
                elif action == EditorAction.SEARCH:
                    return await self._search(SearchInput(**input_data))
                elif action == EditorAction.LIST_DIRECTORY:
                    return await self._list_directory(ListDirectoryInput(**input_data))
            
            raise ValueError(f"Unknown input type: {type(input_data)}")


# Convenience function to create an editor tool
def create_editor_tool(working_directory: str | None = None) -> EditorTool:
    """Create an editor tool with default settings."""
    return EditorTool(working_directory=working_directory)
