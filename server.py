"""
Mini-Devin API Server v5.0
Phase 34: Better Error Handling
Phase 35: Task History & Export
Phase 36: Multi-Model Support (OpenAI, Anthropic, Ollama)
Phase 37: File Upload
Phase 38: Agent Memory (cross-session)
"""

import os
import asyncio
import subprocess
import sqlite3
import re
import json
import uuid
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("MINI_DEVIN_DB_PATH", "/root/mini-devin/mini_devin.db")
WORKSPACE_DIR = os.environ.get("MINI_DEVIN_WORKSPACE", "/root/mini-devin/workspace")
UPLOADS_DIR = os.environ.get("MINI_DEVIN_UPLOADS", "/root/mini-devin/uploads")
MEMORY_DIR = os.environ.get("MINI_DEVIN_MEMORY", "/root/mini-devin/memory")

class APIError(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400, details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

ERROR_SUGGESTIONS = {
    "file_not_found": ["Check if the file path is correct", "Use list_files to see available files", "Create the file first using file_write"],
    "permission_denied": ["File is in a restricted directory", "Use /tmp or workspace directory instead"],
    "command_timeout": ["Command took too long (60s limit)", "Try breaking into smaller commands", "Check if command is stuck"],
    "command_blocked": ["Command contains dangerous patterns", "Use safer alternatives"],
    "api_error": ["Check API key configuration", "Verify model name is correct", "Try again later"],
    "rate_limit": ["Too many requests", "Wait a moment and try again"],
    "invalid_json": ["Tool call JSON is malformed", "Check JSON syntax"],
    "unknown_tool": ["Tool name not recognized", "Check available tools list"],
}

def get_error_suggestions(error_code: str) -> List[str]:
    return ERROR_SUGGESTIONS.get(error_code, ["Try a different approach", "Check the error message for details"])

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(REPOS_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TEXT,
        status TEXT,
        working_directory TEXT,
        model TEXT,
        provider TEXT DEFAULT 'openai',
        max_iterations INTEGER,
        current_task TEXT,
        iteration INTEGER DEFAULT 0,
        total_tasks INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        session_id TEXT,
        description TEXT,
        status TEXT,
        created_at TEXT,
        started_at TEXT,
        completed_at TEXT,
        result TEXT,
        error_message TEXT,
        error_code TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS task_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        output_type TEXT,
        content TEXT,
        created_at TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS uploaded_files (
        file_id TEXT PRIMARY KEY,
        session_id TEXT,
        original_name TEXT,
        stored_path TEXT,
        file_size INTEGER,
        mime_type TEXT,
        uploaded_at TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
        memory_id TEXT PRIMARY KEY,
        session_id TEXT,
        memory_type TEXT,
        content TEXT,
        created_at TEXT,
        last_accessed TEXT,
        access_count INTEGER DEFAULT 0,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )''')
    # Phase 41: GitHub Repo Integration
    c.execute('''CREATE TABLE IF NOT EXISTS github_repos (
        repo_id TEXT PRIMARY KEY,
        repo_url TEXT NOT NULL,
        repo_name TEXT NOT NULL,
        owner TEXT NOT NULL,
        github_token TEXT,
        local_path TEXT,
        default_branch TEXT DEFAULT 'main',
        created_at TEXT,
        last_synced TEXT,
        status TEXT DEFAULT 'pending'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS session_repos (
        session_id TEXT,
        repo_id TEXT,
        branch TEXT DEFAULT 'main',
        linked_at TEXT,
        PRIMARY KEY (session_id, repo_id),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (repo_id) REFERENCES github_repos(repo_id)
    )''')
    try:
        c.execute("ALTER TABLE sessions ADD COLUMN provider TEXT DEFAULT 'openai'")
    except:
        pass
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN error_message TEXT")
    except:
        pass
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN error_code TEXT")
    except:
        pass
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

active_websockets: Dict[str, List[WebSocket]] = {}
llm_clients: Dict[str, Any] = {}

class CreateSessionRequest(BaseModel):
    working_directory: str = Field(default=".", description="Working directory")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    provider: str = Field(default="openai", description="LLM provider: openai, anthropic, ollama")
    max_iterations: int = Field(default=10, description="Max iterations per task")

class CreateTaskRequest(BaseModel):
    description: str = Field(..., description="Task description")
    files: Optional[List[str]] = Field(default=None, description="List of uploaded file IDs")

class MemoryEntry(BaseModel):
    key: str
    value: str
    memory_type: str = "fact"

class ToolResult(BaseModel):
    tool: str
    success: bool
    output: str
    error: Optional[str] = None
    error_code: Optional[str] = None
    suggestions: Optional[List[str]] = None

# Phase 41: GitHub Repo Integration Models
class AddRepoRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL (e.g., https://github.com/owner/repo)")
    github_token: Optional[str] = Field(default=None, description="GitHub Personal Access Token for private repos")
    branch: str = Field(default="main", description="Default branch to use")

class LinkRepoRequest(BaseModel):
    repo_id: str = Field(..., description="Repository ID to link")
    branch: str = Field(default="main", description="Branch to work on")

class GitOperationRequest(BaseModel):
    repo_id: str = Field(..., description="Repository ID")
    operation: str = Field(..., description="Git operation: clone, pull, commit, push, create_branch, create_pr")
    message: Optional[str] = Field(default=None, description="Commit message or PR title")
    branch: Optional[str] = Field(default=None, description="Branch name for operations")
    body: Optional[str] = Field(default=None, description="PR body/description")
    base_branch: Optional[str] = Field(default="main", description="Base branch for PR")

REPOS_DIR = os.environ.get("MINI_DEVIN_REPOS", "/root/mini-devin/repos")

DANGEROUS_COMMANDS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){", "fork bomb",
    "chmod -R 777 /", "chown -R", "> /dev/sda", "mv /* /dev/null",
    "wget http", "curl http", "nc -e", "bash -i", "/dev/tcp"
]

ALLOWED_DIRS = ["/tmp", WORKSPACE_DIR, UPLOADS_DIR]

def is_path_allowed(path: str) -> bool:
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(d) for d in ALLOWED_DIRS)

def is_command_safe(command: str) -> tuple[bool, str]:
    cmd_lower = command.lower()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in cmd_lower:
            return False, f"Command blocked: contains '{dangerous}'"
    return True, ""

def execute_terminal(command: str, working_dir: str = "/tmp") -> ToolResult:
    try:
        safe, reason = is_command_safe(command)
        if not safe:
            return ToolResult(tool="terminal", success=False, output="", error=reason, error_code="command_blocked", suggestions=get_error_suggestions("command_blocked"))
        
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, cwd=working_dir,
            env={**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"}
        )
        output = result.stdout[:10000] if result.stdout else ""
        if result.returncode != 0:
            error = result.stderr[:2000] if result.stderr else "Command failed with no error output"
            return ToolResult(tool="terminal", success=False, output=output, error=error, error_code="command_failed", suggestions=["Check command syntax", "Verify file paths exist"])
        return ToolResult(tool="terminal", success=True, output=output)
    except subprocess.TimeoutExpired:
        return ToolResult(tool="terminal", success=False, output="", error="Command timed out (60s limit)", error_code="command_timeout", suggestions=get_error_suggestions("command_timeout"))
    except FileNotFoundError:
        return ToolResult(tool="terminal", success=False, output="", error=f"Working directory not found: {working_dir}", error_code="file_not_found", suggestions=get_error_suggestions("file_not_found"))
    except Exception as e:
        return ToolResult(tool="terminal", success=False, output="", error=str(e), error_code="unknown_error", suggestions=["Check command syntax", "Try a simpler command"])

def execute_file_read(path: str) -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="file_read", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied", suggestions=get_error_suggestions("permission_denied"))
        if not os.path.exists(path):
            return ToolResult(tool="file_read", success=False, output="", error=f"File not found: {path}", error_code="file_not_found", suggestions=get_error_suggestions("file_not_found"))
        with open(path, 'r') as f:
            content = f.read(100000)
        return ToolResult(tool="file_read", success=True, output=content)
    except UnicodeDecodeError:
        return ToolResult(tool="file_read", success=False, output="", error="File is not a text file (binary content)", error_code="binary_file", suggestions=["Use a different tool for binary files"])
    except Exception as e:
        return ToolResult(tool="file_read", success=False, output="", error=str(e), error_code="unknown_error")

def execute_file_write(path: str, content: str) -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="file_write", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied", suggestions=get_error_suggestions("permission_denied"))
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return ToolResult(tool="file_write", success=True, output=f"Written {len(content)} bytes to {path}")
    except Exception as e:
        return ToolResult(tool="file_write", success=False, output="", error=str(e), error_code="unknown_error")

def execute_list_files(path: str = "/tmp") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="list_files", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied", suggestions=get_error_suggestions("permission_denied"))
        if not os.path.exists(path):
            return ToolResult(tool="list_files", success=False, output="", error=f"Directory not found: {path}", error_code="file_not_found", suggestions=get_error_suggestions("file_not_found"))
        entries = []
        for entry in os.scandir(path):
            entry_type = "dir" if entry.is_dir() else "file"
            size = entry.stat().st_size if entry.is_file() else 0
            entries.append(f"{entry_type}\t{size}\t{entry.name}")
        return ToolResult(tool="list_files", success=True, output="\n".join(entries[:200]) if entries else "Directory is empty")
    except Exception as e:
        return ToolResult(tool="list_files", success=False, output="", error=str(e), error_code="unknown_error")

def execute_git(command: str, working_dir: str = "/tmp") -> ToolResult:
    try:
        allowed_git_commands = ["status", "log", "diff", "branch", "show", "ls-files", "rev-parse", "config --get"]
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return ToolResult(tool="git", success=False, output="", error="No git command provided", error_code="invalid_input")
        
        git_cmd = cmd_parts[0] if len(cmd_parts) == 1 else " ".join(cmd_parts[:2])
        if not any(git_cmd.startswith(allowed) for allowed in allowed_git_commands):
            return ToolResult(tool="git", success=False, output="", error=f"Git command not allowed. Allowed: {allowed_git_commands}", error_code="command_blocked", suggestions=["Use read-only git commands"])
        
        full_command = f"git {command}"
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True, timeout=30, cwd=working_dir)
        if result.returncode != 0:
            return ToolResult(tool="git", success=False, output="", error=result.stderr[:1000], error_code="command_failed", suggestions=["Check if directory is a git repository"])
        return ToolResult(tool="git", success=True, output=result.stdout[:5000])
    except subprocess.TimeoutExpired:
        return ToolResult(tool="git", success=False, output="", error="Git command timed out", error_code="command_timeout")
    except Exception as e:
        return ToolResult(tool="git", success=False, output="", error=str(e), error_code="unknown_error")

def execute_code_analysis(path: str, analysis_type: str = "structure") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="code_analysis", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied")
        if not os.path.exists(path):
            return ToolResult(tool="code_analysis", success=False, output="", error=f"File not found: {path}", error_code="file_not_found", suggestions=get_error_suggestions("file_not_found"))
        
        with open(path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        if analysis_type == "structure":
            functions = re.findall(r'(?:def|function|func)\s+(\w+)', content)
            classes = re.findall(r'(?:class)\s+(\w+)', content)
            imports = re.findall(r'(?:import|from|require|include)\s+[\w.]+', content)
            result = f"File: {path}\nLines: {len(lines)}\nClasses: {', '.join(classes) if classes else 'None'}\nFunctions: {', '.join(functions) if functions else 'None'}\nImports: {len(imports)} found\n"
            return ToolResult(tool="code_analysis", success=True, output=result)
        elif analysis_type == "complexity":
            indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            max_indent = max(indent_levels) if indent_levels else 0
            avg_indent = sum(indent_levels) / len(indent_levels) if indent_levels else 0
            result = f"File: {path}\nTotal lines: {len(lines)}\nNon-empty lines: {len([l for l in lines if l.strip()])}\nMax nesting depth: {max_indent // 4}\nAverage indentation: {avg_indent:.1f} spaces\n"
            return ToolResult(tool="code_analysis", success=True, output=result)
        else:
            return ToolResult(tool="code_analysis", success=False, output="", error=f"Unknown analysis type: {analysis_type}. Use 'structure' or 'complexity'", error_code="invalid_input")
    except Exception as e:
        return ToolResult(tool="code_analysis", success=False, output="", error=str(e), error_code="unknown_error")

def execute_search_files(path: str, pattern: str, file_pattern: str = "*") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="search_files", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}", error_code="permission_denied")
        if not os.path.exists(path):
            return ToolResult(tool="search_files", success=False, output="", error=f"Directory not found: {path}", error_code="file_not_found")
        
        import fnmatch
        results = []
        for root, dirs, files in os.walk(path):
            for filename in fnmatch.filter(files, file_pattern):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        for i, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                results.append(f"{filepath}:{i}: {line.strip()[:100]}")
                                if len(results) >= 50:
                                    break
                except:
                    pass
                if len(results) >= 50:
                    break
            if len(results) >= 50:
                break
        
        if results:
            return ToolResult(tool="search_files", success=True, output="\n".join(results))
        else:
            return ToolResult(tool="search_files", success=True, output=f"No matches found for '{pattern}'")
    except Exception as e:
        return ToolResult(tool="search_files", success=False, output="", error=str(e), error_code="unknown_error")

def execute_memory_store(key: str, value: str, session_id: str) -> ToolResult:
    try:
        conn = get_db()
        c = conn.cursor()
        memory_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        c.execute("INSERT OR REPLACE INTO agent_memory (memory_id, session_id, memory_type, content, created_at, last_accessed, access_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (memory_id, session_id, "user_stored", json.dumps({"key": key, "value": value}), now, now, 1))
        conn.commit()
        conn.close()
        return ToolResult(tool="memory_store", success=True, output=f"Stored memory: {key}")
    except Exception as e:
        return ToolResult(tool="memory_store", success=False, output="", error=str(e), error_code="unknown_error")

def execute_memory_recall(key: str, session_id: str) -> ToolResult:
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT content FROM agent_memory WHERE session_id=? AND content LIKE ?", (session_id, f'%"key": "{key}"%'))
        row = c.fetchone()
        if row:
            data = json.loads(row[0])
            c.execute("UPDATE agent_memory SET last_accessed=?, access_count=access_count+1 WHERE session_id=? AND content LIKE ?",
                      (datetime.utcnow().isoformat(), session_id, f'%"key": "{key}"%'))
            conn.commit()
            conn.close()
            return ToolResult(tool="memory_recall", success=True, output=data.get("value", ""))
        conn.close()
        return ToolResult(tool="memory_recall", success=False, output="", error=f"No memory found for key: {key}", error_code="not_found")
    except Exception as e:
        return ToolResult(tool="memory_recall", success=False, output="", error=str(e), error_code="unknown_error")

def get_llm_client(provider: str):
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            import openai
            return openai.OpenAI(api_key=api_key)
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                return None
    elif provider == "ollama":
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        return {"type": "ollama", "url": ollama_url}
    return None

async def call_llm(provider: str, model: str, messages: list, max_tokens: int = 2000) -> str:
    client = get_llm_client(provider)
    if not client:
        raise APIError(f"LLM provider '{provider}' not configured", "api_not_configured", 500)
    
    try:
        if provider == "openai":
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.7)
            return response.choices[0].message.content
        elif provider == "anthropic":
            system_msg = ""
            user_msgs = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_msgs.append(msg)
            response = client.messages.create(model=model, max_tokens=max_tokens, system=system_msg, messages=user_msgs)
            return response.content[0].text
        elif provider == "ollama":
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                response = await http_client.post(f"{client['url']}/api/chat", json={"model": model, "messages": messages, "stream": False})
                if response.status_code == 200:
                    return response.json().get("message", {}).get("content", "")
                else:
                    raise APIError(f"Ollama error: {response.text}", "ollama_error", response.status_code)
        raise APIError(f"Unknown provider: {provider}", "unknown_provider", 400)
    except Exception as e:
        if isinstance(e, APIError):
            raise
        raise APIError(f"LLM call failed: {str(e)}", "api_error", 500, {"original_error": str(e)})

def get_session_memory(session_id: str) -> str:
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT content FROM agent_memory WHERE session_id=? ORDER BY last_accessed DESC LIMIT 10", (session_id,))
        rows = c.fetchall()
        conn.close()
        if rows:
            memories = []
            for row in rows:
                try:
                    data = json.loads(row[0])
                    memories.append(f"- {data.get('key', 'unknown')}: {data.get('value', '')}")
                except:
                    pass
            if memories:
                return "\n\n## Your Memory (from previous interactions)\n" + "\n".join(memories)
        return ""
    except:
        return ""

SYSTEM_PROMPT = """You are Mini-Devin, an autonomous AI software engineer. You MUST use tools to accomplish tasks - do not just describe what you would do.

## CRITICAL: You MUST output tool calls as JSON blocks

When you need to perform an action, you MUST output a JSON block like this:

```json
{"tool": "terminal", "command": "python3 --version"}
```

## Available Tools

1. **terminal** - Run shell commands
   ```json
   {"tool": "terminal", "command": "ls -la"}
   ```

2. **file_write** - Create or overwrite a file
   ```json
   {"tool": "file_write", "path": "/tmp/hello.py", "content": "print('Hello!')"}
   ```

3. **file_read** - Read file contents
   ```json
   {"tool": "file_read", "path": "/tmp/hello.py"}
   ```

4. **list_files** - List directory contents
   ```json
   {"tool": "list_files", "path": "/tmp"}
   ```

5. **git** - Run git commands (read-only: status, log, diff, branch, show)
   ```json
   {"tool": "git", "command": "status"}
   ```

6. **code_analysis** - Analyze code structure or complexity
   ```json
   {"tool": "code_analysis", "path": "/tmp/script.py", "analysis_type": "structure"}
   ```

7. **search_files** - Search for text in files
   ```json
   {"tool": "search_files", "path": "/tmp", "pattern": "TODO", "file_pattern": "*.py"}
   ```

8. **memory_store** - Store information for later recall
   ```json
   {"tool": "memory_store", "key": "user_preference", "value": "prefers Python"}
   ```

9. **memory_recall** - Recall stored information
   ```json
   {"tool": "memory_recall", "key": "user_preference"}
   ```

## Working Directory
You can create and modify files in: /tmp and the workspace directory.

## Error Handling
If a tool fails, you'll receive an error message with suggestions. Use these to fix the issue and try again.

## Instructions
1. ALWAYS use tools - never just describe what you would do
2. After each tool execution, you'll see the results
3. If a tool fails, read the error and suggestions, then try a different approach
4. Continue using tools until the task is complete
5. Provide a summary when done

REMEMBER: Always output the JSON tool block, never just describe what you would do!"""

def parse_tool_calls(response: str) -> list:
    tools = []
    pattern = r'```json\s*(\{[^`]+\})\s*```'
    matches = re.findall(pattern, response, re.DOTALL)
    for match in matches:
        try:
            tool_call = json.loads(match.strip())
            if "tool" in tool_call:
                tools.append(tool_call)
        except json.JSONDecodeError:
            pass
    
    if not tools:
        pattern2 = r'\{[^{}]*"tool"\s*:\s*"[^"]+\"[^{}]*\}'
        matches2 = re.findall(pattern2, response)
        for match in matches2:
            try:
                tool_call = json.loads(match)
                if "tool" in tool_call:
                    tools.append(tool_call)
            except:
                pass
    
    return tools

def execute_tool(tool_call: dict, session_id: str = "") -> ToolResult:
    tool = tool_call.get("tool")
    try:
        if tool == "terminal":
            return execute_terminal(tool_call.get("command", ""), tool_call.get("working_dir", "/tmp"))
        elif tool == "file_read":
            return execute_file_read(tool_call.get("path", ""))
        elif tool == "file_write":
            return execute_file_write(tool_call.get("path", ""), tool_call.get("content", ""))
        elif tool == "list_files":
            return execute_list_files(tool_call.get("path", "/tmp"))
        elif tool == "git":
            return execute_git(tool_call.get("command", ""), tool_call.get("working_dir", "/tmp"))
        elif tool == "code_analysis":
            return execute_code_analysis(tool_call.get("path", ""), tool_call.get("analysis_type", "structure"))
        elif tool == "search_files":
            return execute_search_files(tool_call.get("path", "/tmp"), tool_call.get("pattern", ""), tool_call.get("file_pattern", "*"))
        elif tool == "memory_store":
            return execute_memory_store(tool_call.get("key", ""), tool_call.get("value", ""), session_id)
        elif tool == "memory_recall":
            return execute_memory_recall(tool_call.get("key", ""), session_id)
        else:
            return ToolResult(tool=str(tool), success=False, output="", error=f"Unknown tool: {tool}", error_code="unknown_tool", suggestions=get_error_suggestions("unknown_tool"))
    except Exception as e:
        return ToolResult(tool=str(tool), success=False, output="", error=f"Tool execution error: {str(e)}", error_code="execution_error", suggestions=["Check tool parameters", "Try again"])

async def broadcast_to_session(session_id: str, message: dict):
    if session_id in active_websockets:
        dead_connections = []
        for ws in active_websockets[session_id]:
            try:
                await ws.send_json(message)
            except:
                dead_connections.append(ws)
        for ws in dead_connections:
            active_websockets[session_id].remove(ws)

async def execute_agent_task(session_id: str, task_id: str, description: str, model: str, provider: str = "openai", max_iterations: int = 10):
    conn = get_db()
    c = conn.cursor()
    
    llm_client = get_llm_client(provider)
    if not llm_client:
        error_msg = f"LLM provider '{provider}' not configured"
        error_code = "api_not_configured"
        suggestions = get_error_suggestions("api_error")
        c.execute("UPDATE tasks SET status=?, result=?, error_message=?, error_code=? WHERE task_id=?", ("failed", error_msg, error_msg, error_code, task_id))
        c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "error", json.dumps({"message": error_msg, "code": error_code, "suggestions": suggestions}), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        await broadcast_to_session(session_id, {"type": "task_failed", "task_id": task_id, "error": error_msg, "error_code": error_code, "suggestions": suggestions})
        return
    
    c.execute("UPDATE tasks SET status=?, started_at=? WHERE task_id=?", ("running", datetime.utcnow().isoformat(), task_id))
    conn.commit()
    
    await broadcast_to_session(session_id, {"type": "task_started", "task_id": task_id})
    
    c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "thinking", f"Analyzing task: {description}", datetime.utcnow().isoformat()))
    conn.commit()
    await broadcast_to_session(session_id, {"type": "thinking", "task_id": task_id, "content": f"Analyzing task: {description}"})
    
    memory_context = get_session_memory(session_id)
    system_prompt = SYSTEM_PROMPT + memory_context
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": description}]
    iteration = 0
    final_response = ""
    
    try:
        while iteration < max_iterations:
            iteration += 1
            
            c.execute("UPDATE sessions SET iteration=? WHERE session_id=?", (iteration, session_id))
            conn.commit()
            await broadcast_to_session(session_id, {"type": "iteration", "task_id": task_id, "iteration": iteration, "max": max_iterations})
            
            try:
                agent_response = await call_llm(provider, model, messages, max_tokens=2000)
            except APIError as e:
                c.execute("UPDATE tasks SET status=?, result=?, error_message=?, error_code=? WHERE task_id=?", ("failed", e.message, e.message, e.code, task_id))
                c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "error", json.dumps({"message": e.message, "code": e.code, "suggestions": get_error_suggestions(e.code)}), datetime.utcnow().isoformat()))
                conn.commit()
                await broadcast_to_session(session_id, {"type": "task_failed", "task_id": task_id, "error": e.message, "error_code": e.code, "suggestions": get_error_suggestions(e.code)})
                conn.close()
                return
            
            final_response = agent_response
            
            c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "response", agent_response, datetime.utcnow().isoformat()))
            conn.commit()
            await broadcast_to_session(session_id, {"type": "response", "task_id": task_id, "content": agent_response, "iteration": iteration})
            
            tool_calls = parse_tool_calls(agent_response)
            
            if not tool_calls:
                break
            
            tool_results = []
            for tool_call in tool_calls:
                await broadcast_to_session(session_id, {"type": "tool_started", "task_id": task_id, "tool": tool_call})
                
                result = execute_tool(tool_call, session_id)
                tool_results.append(result)
                
                tool_output = {"tool": result.tool, "success": result.success, "output": result.output[:5000], "error": result.error, "error_code": result.error_code, "suggestions": result.suggestions}
                c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "tool", json.dumps(tool_output), datetime.utcnow().isoformat()))
                conn.commit()
                await broadcast_to_session(session_id, {"type": "tool_result", "task_id": task_id, "result": tool_output})
            
            tool_output_text = "\n\n".join([
                f"**Tool: {r.tool}**\nSuccess: {r.success}\n```\n{r.output}\n```\n{f'Error: {r.error}' if r.error else ''}{f' (Suggestions: {r.suggestions})' if r.suggestions else ''}"
                for r in tool_results
            ])
            
            messages.append({"role": "assistant", "content": agent_response})
            messages.append({"role": "user", "content": f"Tool execution results:\n\n{tool_output_text}\n\nContinue with the task. If you need to use more tools, output the JSON block. If the task is complete, provide a summary."})
        
        c.execute("UPDATE tasks SET status=?, completed_at=?, result=? WHERE task_id=?", ("completed", datetime.utcnow().isoformat(), final_response, task_id))
        conn.commit()
        await broadcast_to_session(session_id, {"type": "task_completed", "task_id": task_id, "result": final_response})
        
    except Exception as e:
        error_msg = str(e)
        error_code = "unknown_error"
        tb = traceback.format_exc()
        c.execute("UPDATE tasks SET status=?, result=?, error_message=?, error_code=? WHERE task_id=?", ("failed", f"Error: {error_msg}", error_msg, error_code, task_id))
        c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "error", json.dumps({"message": error_msg, "code": error_code, "traceback": tb[:1000]}), datetime.utcnow().isoformat()))
        conn.commit()
        await broadcast_to_session(session_id, {"type": "task_failed", "task_id": task_id, "error": error_msg, "error_code": error_code})
    
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Starting Mini-Devin API v5.0...")
    print("Features: Tool Execution, SQLite Storage, WebSocket, Multi-Model, File Upload, Agent Memory, Export")
    init_db()
    print(f"SQLite database initialized at {DB_PATH}")
    print(f"Workspace directory: {WORKSPACE_DIR}")
    print(f"Uploads directory: {UPLOADS_DIR}")
    print(f"Memory directory: {MEMORY_DIR}")
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    ollama_url = os.environ.get("OLLAMA_URL")
    
    providers = []
    if openai_key:
        providers.append("openai")
        print(f"OpenAI API key configured")
    if anthropic_key:
        providers.append("anthropic")
        print(f"Anthropic API key configured")
    if ollama_url:
        providers.append("ollama")
        print(f"Ollama URL configured: {ollama_url}")
    
    if not providers:
        print("Warning: No LLM providers configured - set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_URL")
    else:
        print(f"Available providers: {', '.join(providers)}")
    
    yield
    print("Shutting down Mini-Devin API...")

app = FastAPI(title="Mini-Devin API", version="5.0.0", description="Autonomous AI Software Engineer with Multi-Model Support, File Upload, Agent Memory, and Export", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_configured_providers():
    providers = []
    if os.environ.get("OPENAI_API_KEY"):
        providers.append({"id": "openai", "name": "OpenAI", "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]})
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append({"id": "anthropic", "name": "Anthropic", "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]})
    if os.environ.get("OLLAMA_URL"):
        providers.append({"id": "ollama", "name": "Ollama (Local)", "models": ["llama3", "codellama", "mistral"]})
    return providers

@app.get("/")
async def root():
    providers = get_configured_providers()
    return {
        "name": "Mini-Devin API",
        "version": "5.0.0",
        "status": "running",
        "mode": "full-agent" if providers else "limited",
        "llm_configured": bool(providers),
        "providers": [p["id"] for p in providers],
        "features": ["tool_execution", "persistent_storage", "websocket", "multi_model", "file_upload", "agent_memory", "export", "error_handling"],
        "tools": ["terminal", "file_read", "file_write", "list_files", "git", "code_analysis", "search_files", "memory_store", "memory_recall"],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "5.0.0"}

@app.get("/api/health")
async def api_health():
    providers = get_configured_providers()
    return {"status": "healthy", "mode": "full-agent" if providers else "limited", "llm_configured": bool(providers), "providers": [p["id"] for p in providers], "version": "5.0.0"}

@app.get("/api/status")
async def get_status():
    providers = get_configured_providers()
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sessions")
    session_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tasks WHERE status='completed'")
    completed_tasks = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM uploaded_files")
    file_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM agent_memory")
    memory_count = c.fetchone()[0]
    conn.close()
    return {
        "status": "running",
        "mode": "full-agent" if providers else "limited",
        "version": "5.0.0",
        "active_sessions": session_count,
        "completed_tasks": completed_tasks,
        "uploaded_files": file_count,
        "memory_entries": memory_count,
        "llm_configured": bool(providers),
        "providers": [p["id"] for p in providers],
        "features": ["tool_execution", "persistent_storage", "websocket", "multi_model", "file_upload", "agent_memory", "export", "error_handling"]
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in active_websockets:
        active_websockets[session_id] = []
    active_websockets[session_id].append(websocket)
    
    try:
        await websocket.send_json({"type": "connected", "session_id": session_id})
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except:
                pass
    except WebSocketDisconnect:
        if session_id in active_websockets:
            active_websockets[session_id].remove(websocket)
            if not active_websockets[session_id]:
                del active_websockets[session_id]

@app.get("/api/sessions")
async def list_sessions():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    sessions = []
    for row in rows:
        sessions.append({
            "session_id": row[0], "created_at": row[1], "status": row[2],
            "working_directory": row[3], "model": row[4], "max_iterations": row[5],
            "current_task": row[6], "iteration": row[7], "total_tasks": row[8],
            "llm_enabled": bool(os.environ.get("OPENAI_API_KEY"))
        })
    return {"sessions": sessions, "total": len(sessions)}

@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest):
    providers = get_configured_providers()
    session_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().isoformat()
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO sessions (session_id, created_at, status, working_directory, model, provider, max_iterations, iteration, total_tasks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (session_id, created_at, "active", request.working_directory, request.model, request.provider, request.max_iterations, 0, 0))
    conn.commit()
    conn.close()
    return {"session_id": session_id, "created_at": created_at, "status": "active", "provider": request.provider, "model": request.model, "llm_enabled": bool(providers)}

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": row[0], "created_at": row[1], "status": row[2],
        "working_directory": row[3], "model": row[4], "max_iterations": row[5],
        "current_task": row[6], "iteration": row[7], "total_tasks": row[8],
        "llm_enabled": bool(os.environ.get("OPENAI_API_KEY"))
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM task_outputs WHERE task_id IN (SELECT task_id FROM tasks WHERE session_id=?)", (session_id,))
    c.execute("DELETE FROM tasks WHERE session_id=?", (session_id,))
    c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "session_id": session_id}

@app.post("/api/sessions/{session_id}/tasks")
async def create_task(session_id: str, request: CreateTaskRequest):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT model, provider, max_iterations FROM sessions WHERE session_id=?", (session_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    model, provider, max_iterations = row[0], row[1] or "openai", row[2]
    providers = get_configured_providers()
    task_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().isoformat()
    c.execute("INSERT INTO tasks (task_id, session_id, description, status, created_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, session_id, request.description, "queued", created_at))
    c.execute("UPDATE sessions SET total_tasks = total_tasks + 1, current_task = ? WHERE session_id = ?", (task_id, session_id))
    conn.commit()
    conn.close()
    if providers:
        asyncio.create_task(execute_agent_task(session_id, task_id, request.description, model, provider, max_iterations))
    return {"task_id": task_id, "session_id": session_id, "description": request.description, "status": "queued", "created_at": created_at, "provider": provider, "model": model}

@app.get("/api/sessions/{session_id}/tasks")
async def list_tasks(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE session_id=? ORDER BY created_at DESC", (session_id,))
    rows = c.fetchall()
    conn.close()
    tasks = []
    for row in rows:
        tasks.append({
            "task_id": row[0], "session_id": row[1], "description": row[2],
            "status": row[3], "created_at": row[4], "started_at": row[5],
            "completed_at": row[6], "result": row[7]
        })
    return {"tasks": tasks, "total": len(tasks)}

@app.get("/api/sessions/{session_id}/tasks/{task_id}")
async def get_task(session_id: str, task_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE task_id=? AND session_id=?", (task_id, session_id))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": row[0], "session_id": row[1], "description": row[2],
        "status": row[3], "created_at": row[4], "started_at": row[5],
        "completed_at": row[6], "result": row[7]
    }

@app.get("/api/sessions/{session_id}/tasks/{task_id}/output")
async def get_task_output(session_id: str, task_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT status, result FROM tasks WHERE task_id=? AND session_id=?", (task_id, session_id))
    task_row = c.fetchone()
    if not task_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")
    c.execute("SELECT output_type, content FROM task_outputs WHERE task_id=? ORDER BY created_at", (task_id,))
    output_rows = c.fetchall()
    conn.close()
    outputs = [{"type": row[0], "content": row[1]} for row in output_rows]
    return {"task_id": task_id, "status": task_row[0], "outputs": outputs, "result": task_row[1]}

@app.get("/api/tools")
async def list_tools():
    return {
        "tools": [
            {"name": "terminal", "description": "Run shell commands", "params": ["command", "working_dir"]},
            {"name": "file_read", "description": "Read file contents", "params": ["path"]},
            {"name": "file_write", "description": "Write content to file", "params": ["path", "content"]},
            {"name": "list_files", "description": "List directory contents", "params": ["path"]},
            {"name": "git", "description": "Run git commands (read-only)", "params": ["command", "working_dir"]},
            {"name": "code_analysis", "description": "Analyze code structure/complexity", "params": ["path", "analysis_type"]},
            {"name": "search_files", "description": "Search for text in files", "params": ["path", "pattern", "file_pattern"]},
            {"name": "memory_store", "description": "Store information for later recall", "params": ["key", "value"]},
            {"name": "memory_recall", "description": "Recall stored information", "params": ["key"]}
        ]
    }

@app.get("/api/skills")
async def list_skills():
    return {"skills": [
        {"id": "write-script", "name": "Write Script", "description": "Write and run a script", "tags": ["python", "scripting"], "version": "1.0.0", "is_custom": False},
        {"id": "analyze-code", "name": "Analyze Code", "description": "Analyze code structure and complexity", "tags": ["analysis"], "version": "1.0.0", "is_custom": False},
        {"id": "search-codebase", "name": "Search Codebase", "description": "Search for patterns in code", "tags": ["search"], "version": "1.0.0", "is_custom": False},
        {"id": "git-status", "name": "Git Status", "description": "Check git repository status", "tags": ["git"], "version": "1.0.0", "is_custom": False},
        {"id": "file-operations", "name": "File Operations", "description": "Read, write, and list files", "tags": ["files"], "version": "1.0.0", "is_custom": False}
    ]}

@app.get("/api/skills/tags")
async def list_skill_tags():
    return {"tags": ["python", "scripting", "analysis", "search", "git", "files"]}

@app.get("/api/providers")
async def list_providers():
    return {"providers": get_configured_providers()}

@app.post("/api/sessions/{session_id}/files")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    file_id = str(uuid.uuid4())[:8]
    session_upload_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)
    
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ""
    stored_filename = f"{file_id}{file_ext}"
    stored_path = os.path.join(session_upload_dir, stored_filename)
    
    content = await file.read()
    with open(stored_path, "wb") as f:
        f.write(content)
    
    c.execute("INSERT INTO uploaded_files (file_id, session_id, original_name, stored_path, file_size, mime_type, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (file_id, session_id, file.filename, stored_path, len(content), file.content_type, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    
    await broadcast_to_session(session_id, {"type": "file_uploaded", "file_id": file_id, "filename": file.filename, "size": len(content)})
    return {"file_id": file_id, "filename": file.filename, "size": len(content), "mime_type": file.content_type}

@app.get("/api/sessions/{session_id}/files")
async def list_files_endpoint(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT file_id, original_name, file_size, mime_type, uploaded_at FROM uploaded_files WHERE session_id=?", (session_id,))
    rows = c.fetchall()
    conn.close()
    files = [{"file_id": r[0], "filename": r[1], "size": r[2], "mime_type": r[3], "uploaded_at": r[4]} for r in rows]
    return {"files": files, "total": len(files)}

@app.get("/api/sessions/{session_id}/files/{file_id}")
async def download_file(session_id: str, file_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT stored_path, original_name, mime_type FROM uploaded_files WHERE file_id=? AND session_id=?", (file_id, session_id))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(row[0], filename=row[1], media_type=row[2])

@app.delete("/api/sessions/{session_id}/files/{file_id}")
async def delete_file(session_id: str, file_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT stored_path FROM uploaded_files WHERE file_id=? AND session_id=?", (file_id, session_id))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="File not found")
    try:
        os.remove(row[0])
    except:
        pass
    c.execute("DELETE FROM uploaded_files WHERE file_id=?", (file_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "file_id": file_id}

@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "json"):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    session_row = c.fetchone()
    if not session_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    c.execute("SELECT * FROM tasks WHERE session_id=? ORDER BY created_at", (session_id,))
    task_rows = c.fetchall()
    
    tasks_data = []
    for task in task_rows:
        c.execute("SELECT output_type, content, created_at FROM task_outputs WHERE task_id=? ORDER BY created_at", (task[0],))
        outputs = [{"type": o[0], "content": o[1], "created_at": o[2]} for o in c.fetchall()]
        tasks_data.append({
            "task_id": task[0], "description": task[2], "status": task[3],
            "created_at": task[4], "started_at": task[5], "completed_at": task[6],
            "result": task[7], "outputs": outputs
        })
    
    c.execute("SELECT memory_id, memory_type, content, created_at FROM agent_memory WHERE session_id=?", (session_id,))
    memories = [{"memory_id": m[0], "type": m[1], "content": m[2], "created_at": m[3]} for m in c.fetchall()]
    
    c.execute("SELECT file_id, original_name, file_size, uploaded_at FROM uploaded_files WHERE session_id=?", (session_id,))
    files = [{"file_id": f[0], "filename": f[1], "size": f[2], "uploaded_at": f[3]} for f in c.fetchall()]
    conn.close()
    
    export_data = {
        "session_id": session_id,
        "created_at": session_row[1],
        "model": session_row[4],
        "provider": session_row[5] if len(session_row) > 5 else "openai",
        "tasks": tasks_data,
        "memories": memories,
        "files": files,
        "exported_at": datetime.utcnow().isoformat()
    }
    
    if format == "json":
        return JSONResponse(content=export_data)
    elif format == "markdown":
        md = f"# Session Export: {session_id}\n\n"
        md += f"**Created:** {session_row[1]}\n**Model:** {session_row[4]}\n\n"
        md += "## Tasks\n\n"
        for task in tasks_data:
            md += f"### Task: {task['description'][:50]}...\n"
            md += f"- Status: {task['status']}\n- Created: {task['created_at']}\n\n"
            md += "#### Outputs\n"
            for output in task['outputs']:
                md += f"**{output['type']}:**\n```\n{output['content'][:500]}\n```\n\n"
        if memories:
            md += "## Memories\n\n"
            for mem in memories:
                md += f"- {mem['content']}\n"
        return JSONResponse(content={"format": "markdown", "content": md})
    elif format == "txt":
        txt = f"Session Export: {session_id}\n{'='*50}\n\n"
        txt += f"Created: {session_row[1]}\nModel: {session_row[4]}\n\n"
        txt += "TASKS\n" + "-"*50 + "\n\n"
        for task in tasks_data:
            txt += f"Task: {task['description']}\nStatus: {task['status']}\n\n"
            for output in task['outputs']:
                txt += f"[{output['type']}]\n{output['content'][:500]}\n\n"
        return JSONResponse(content={"format": "txt", "content": txt})
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use: json, markdown, or txt")

@app.get("/api/sessions/{session_id}/memory")
async def get_session_memory_endpoint(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT memory_id, memory_type, content, created_at, last_accessed, access_count FROM agent_memory WHERE session_id=? ORDER BY last_accessed DESC", (session_id,))
    rows = c.fetchall()
    conn.close()
    memories = []
    for r in rows:
        try:
            content = json.loads(r[2])
        except:
            content = {"raw": r[2]}
        memories.append({"memory_id": r[0], "type": r[1], "content": content, "created_at": r[3], "last_accessed": r[4], "access_count": r[5]})
    return {"memories": memories, "total": len(memories)}

@app.post("/api/sessions/{session_id}/memory")
async def store_memory_endpoint(session_id: str, entry: MemoryEntry):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    memory_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    content = json.dumps({"key": entry.key, "value": entry.value})
    c.execute("INSERT INTO agent_memory (memory_id, session_id, memory_type, content, created_at, last_accessed, access_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (memory_id, session_id, entry.memory_type, content, now, now, 1))
    conn.commit()
    conn.close()
    
    await broadcast_to_session(session_id, {"type": "memory_stored", "memory_id": memory_id, "key": entry.key})
    return {"memory_id": memory_id, "key": entry.key, "stored_at": now}

@app.delete("/api/sessions/{session_id}/memory/{memory_id}")
async def delete_memory(session_id: str, memory_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM agent_memory WHERE memory_id=? AND session_id=?", (memory_id, session_id))
    conn.commit()
    conn.close()
    return {"status": "deleted", "memory_id": memory_id}

# ============================================
# Phase 41: GitHub Repo Integration Endpoints
# ============================================

def parse_github_url(url: str) -> tuple[str, str]:
    """Parse GitHub URL to extract owner and repo name."""
    import re
    patterns = [
        r'github\.com[:/]([^/]+)/([^/\.]+)',
        r'github\.com/([^/]+)/([^/]+?)(?:\.git)?$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2).replace('.git', '')
    raise ValueError(f"Invalid GitHub URL: {url}")

def get_repo_clone_url(repo_url: str, token: Optional[str] = None) -> str:
    """Get clone URL with token for authentication."""
    if token:
        # Insert token into URL for authentication
        if repo_url.startswith("https://"):
            return repo_url.replace("https://", f"https://{token}@")
    return repo_url

def execute_git_command(command: str, working_dir: str, timeout: int = 120) -> tuple[bool, str]:
    """Execute a git command and return success status and output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=working_dir,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

@app.get("/api/repos")
async def list_repos():
    """List all connected GitHub repositories."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT repo_id, repo_url, repo_name, owner, default_branch, local_path, created_at, last_synced, status FROM github_repos ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    
    repos = []
    for row in rows:
        repos.append({
            "repo_id": row[0],
            "repo_url": row[1],
            "repo_name": row[2],
            "owner": row[3],
            "default_branch": row[4],
            "local_path": row[5],
            "created_at": row[6],
            "last_synced": row[7],
            "status": row[8],
            "has_token": bool(row[1])  # Don't expose actual token
        })
    return {"repos": repos}

@app.post("/api/repos")
async def add_repo(request: AddRepoRequest):
    """Add a new GitHub repository."""
    try:
        owner, repo_name = parse_github_url(request.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    repo_id = str(uuid.uuid4())[:8]
    local_path = os.path.join(REPOS_DIR, f"{owner}_{repo_name}_{repo_id}")
    now = datetime.utcnow().isoformat()
    
    # Normalize repo URL
    normalized_url = f"https://github.com/{owner}/{repo_name}.git"
    
    conn = get_db()
    c = conn.cursor()
    c.execute("""INSERT INTO github_repos 
                 (repo_id, repo_url, repo_name, owner, github_token, local_path, default_branch, created_at, status)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (repo_id, normalized_url, repo_name, owner, request.github_token, local_path, request.branch, now, "pending"))
    conn.commit()
    conn.close()
    
    return {
        "repo_id": repo_id,
        "repo_url": normalized_url,
        "repo_name": repo_name,
        "owner": owner,
        "local_path": local_path,
        "status": "pending",
        "message": "Repository added. Use POST /api/repos/{repo_id}/clone to clone it."
    }

@app.get("/api/repos/{repo_id}")
async def get_repo(repo_id: str):
    """Get repository details."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT repo_id, repo_url, repo_name, owner, default_branch, local_path, created_at, last_synced, status FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    # Check if local path exists and get git status
    local_status = None
    current_branch = None
    if row[5] and os.path.exists(row[5]):
        success, output = execute_git_command("git status --porcelain", row[5])
        if success:
            local_status = "clean" if not output.strip() else "modified"
        success, branch_output = execute_git_command("git branch --show-current", row[5])
        if success:
            current_branch = branch_output.strip()
    
    return {
        "repo_id": row[0],
        "repo_url": row[1],
        "repo_name": row[2],
        "owner": row[3],
        "default_branch": row[4],
        "local_path": row[5],
        "created_at": row[6],
        "last_synced": row[7],
        "status": row[8],
        "local_status": local_status,
        "current_branch": current_branch
    }

@app.delete("/api/repos/{repo_id}")
async def delete_repo(repo_id: str):
    """Delete a repository and its local clone."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path = row[0]
    
    # Delete from database
    c.execute("DELETE FROM session_repos WHERE repo_id=?", (repo_id,))
    c.execute("DELETE FROM github_repos WHERE repo_id=?", (repo_id,))
    conn.commit()
    conn.close()
    
    # Delete local clone if exists
    if local_path and os.path.exists(local_path):
        import shutil
        shutil.rmtree(local_path, ignore_errors=True)
    
    return {"status": "deleted", "repo_id": repo_id}

@app.post("/api/repos/{repo_id}/clone")
async def clone_repo(repo_id: str):
    """Clone a repository to local storage."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT repo_url, github_token, local_path, default_branch FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    repo_url, token, local_path, default_branch = row
    
    # Check if already cloned
    if os.path.exists(local_path):
        conn.close()
        return {"status": "already_cloned", "local_path": local_path}
    
    # Clone the repository
    clone_url = get_repo_clone_url(repo_url, token)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    success, output = execute_git_command(f"git clone --branch {default_branch} {clone_url} {local_path}", "/tmp", timeout=300)
    
    if not success:
        c.execute("UPDATE github_repos SET status=? WHERE repo_id=?", ("clone_failed", repo_id))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Clone failed: {output}")
    
    # Update status
    now = datetime.utcnow().isoformat()
    c.execute("UPDATE github_repos SET status=?, last_synced=? WHERE repo_id=?", ("cloned", now, repo_id))
    conn.commit()
    conn.close()
    
    return {"status": "cloned", "local_path": local_path, "message": "Repository cloned successfully"}

@app.post("/api/repos/{repo_id}/pull")
async def pull_repo(repo_id: str):
    """Pull latest changes from remote."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path, github_token, repo_url FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path, token, repo_url = row
    
    if not os.path.exists(local_path):
        conn.close()
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    # Set remote URL with token if needed
    if token:
        clone_url = get_repo_clone_url(repo_url, token)
        execute_git_command(f"git remote set-url origin {clone_url}", local_path)
    
    success, output = execute_git_command("git pull", local_path)
    
    if not success:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Pull failed: {output}")
    
    now = datetime.utcnow().isoformat()
    c.execute("UPDATE github_repos SET last_synced=? WHERE repo_id=?", (now, repo_id))
    conn.commit()
    conn.close()
    
    return {"status": "pulled", "output": output}

@app.post("/api/repos/{repo_id}/branch")
async def create_branch(repo_id: str, branch_name: str):
    """Create a new branch."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path = row[0]
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    success, output = execute_git_command(f"git checkout -b {branch_name}", local_path)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Branch creation failed: {output}")
    
    return {"status": "created", "branch": branch_name}

@app.post("/api/repos/{repo_id}/checkout")
async def checkout_branch(repo_id: str, branch_name: str):
    """Checkout an existing branch."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path = row[0]
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    success, output = execute_git_command(f"git checkout {branch_name}", local_path)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Checkout failed: {output}")
    
    return {"status": "checked_out", "branch": branch_name}

@app.post("/api/repos/{repo_id}/commit")
async def commit_changes(repo_id: str, message: str):
    """Commit all changes."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path = row[0]
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    # Stage all changes
    execute_git_command("git add -A", local_path)
    
    # Commit
    success, output = execute_git_command(f'git commit -m "{message}"', local_path)
    
    if not success:
        if "nothing to commit" in output:
            return {"status": "no_changes", "message": "Nothing to commit"}
        raise HTTPException(status_code=500, detail=f"Commit failed: {output}")
    
    return {"status": "committed", "message": message, "output": output}

@app.post("/api/repos/{repo_id}/push")
async def push_changes(repo_id: str, branch: Optional[str] = None):
    """Push changes to remote."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path, github_token, repo_url FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path, token, repo_url = row
    conn.close()
    
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    # Set remote URL with token if needed
    if token:
        clone_url = get_repo_clone_url(repo_url, token)
        execute_git_command(f"git remote set-url origin {clone_url}", local_path)
    
    # Get current branch if not specified
    if not branch:
        success, branch = execute_git_command("git branch --show-current", local_path)
        if success:
            branch = branch.strip()
        else:
            branch = "main"
    
    success, output = execute_git_command(f"git push -u origin {branch}", local_path)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Push failed: {output}")
    
    return {"status": "pushed", "branch": branch, "output": output}

@app.post("/api/repos/{repo_id}/pr")
async def create_pull_request(repo_id: str, title: str, body: str = "", base_branch: str = "main", head_branch: Optional[str] = None):
    """Create a pull request on GitHub."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT local_path, github_token, owner, repo_name FROM github_repos WHERE repo_id=?", (repo_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    local_path, token, owner, repo_name = row
    
    if not token:
        raise HTTPException(status_code=400, detail="GitHub token required to create PR")
    
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Repository not cloned yet")
    
    # Get current branch if not specified
    if not head_branch:
        success, head_branch = execute_git_command("git branch --show-current", local_path)
        if success:
            head_branch = head_branch.strip()
        else:
            raise HTTPException(status_code=500, detail="Could not determine current branch")
    
    # Create PR using GitHub API
    import urllib.request
    
    pr_data = json.dumps({
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo_name}/pulls",
        data=pr_data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            pr_response = json.loads(response.read().decode('utf-8'))
            return {
                "status": "created",
                "pr_number": pr_response.get("number"),
                "pr_url": pr_response.get("html_url"),
                "title": title
            }
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise HTTPException(status_code=e.code, detail=f"GitHub API error: {error_body}")

@app.post("/api/sessions/{session_id}/repos")
async def link_repo_to_session(session_id: str, request: LinkRepoRequest):
    """Link a repository to a session."""
    conn = get_db()
    c = conn.cursor()
    
    # Verify session exists
    c.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Verify repo exists
    c.execute("SELECT repo_id, local_path FROM github_repos WHERE repo_id=?", (request.repo_id,))
    repo_row = c.fetchone()
    if not repo_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    
    now = datetime.utcnow().isoformat()
    c.execute("""INSERT OR REPLACE INTO session_repos (session_id, repo_id, branch, linked_at)
                 VALUES (?, ?, ?, ?)""", (session_id, request.repo_id, request.branch, now))
    
    # Update session working directory to repo path
    c.execute("UPDATE sessions SET working_directory=? WHERE session_id=?", (repo_row[1], session_id))
    
    conn.commit()
    conn.close()
    
    return {"status": "linked", "session_id": session_id, "repo_id": request.repo_id, "branch": request.branch}

@app.get("/api/sessions/{session_id}/repos")
async def get_session_repos(session_id: str):
    """Get repositories linked to a session."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""SELECT r.repo_id, r.repo_url, r.repo_name, r.owner, r.local_path, sr.branch, sr.linked_at
                 FROM session_repos sr
                 JOIN github_repos r ON sr.repo_id = r.repo_id
                 WHERE sr.session_id=?""", (session_id,))
    rows = c.fetchall()
    conn.close()
    
    repos = []
    for row in rows:
        repos.append({
            "repo_id": row[0],
            "repo_url": row[1],
            "repo_name": row[2],
            "owner": row[3],
            "local_path": row[4],
            "branch": row[5],
            "linked_at": row[6]
        })
    
    return {"repos": repos}
