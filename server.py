"""
Mini-Devin API Server v4.0
Phase 29: Improved Tool Execution Reliability
Phase 30: Server Code in Repo
Phase 31: WebSocket Support
Phase 32: More Tools (git, web search, code analysis)
"""

import os
import asyncio
import subprocess
import sqlite3
import re
import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import openai

DB_PATH = os.environ.get("MINI_DEVIN_DB_PATH", "/root/mini-devin/mini_devin.db")
WORKSPACE_DIR = os.environ.get("MINI_DEVIN_WORKSPACE", "/root/mini-devin/workspace")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TEXT,
        status TEXT,
        working_directory TEXT,
        model TEXT,
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
    conn.commit()
    conn.close()

def get_db():
    return sqlite3.connect(DB_PATH)

active_websockets: Dict[str, List[WebSocket]] = {}
client: Optional[openai.OpenAI] = None

class CreateSessionRequest(BaseModel):
    working_directory: str = Field(default=".", description="Working directory")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    max_iterations: int = Field(default=10, description="Max iterations per task")

class CreateTaskRequest(BaseModel):
    description: str = Field(..., description="Task description")

class ToolResult(BaseModel):
    tool: str
    success: bool
    output: str
    error: Optional[str] = None

DANGEROUS_COMMANDS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){", "fork bomb",
    "chmod -R 777 /", "chown -R", "> /dev/sda", "mv /* /dev/null",
    "wget http", "curl http", "nc -e", "bash -i", "/dev/tcp"
]

ALLOWED_DIRS = ["/tmp", WORKSPACE_DIR]

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
            return ToolResult(tool="terminal", success=False, output="", error=reason)
        
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, cwd=working_dir,
            env={**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"}
        )
        output = result.stdout[:10000] if result.stdout else ""
        error = result.stderr[:2000] if result.stderr and result.returncode != 0 else None
        return ToolResult(tool="terminal", success=result.returncode == 0, output=output, error=error)
    except subprocess.TimeoutExpired:
        return ToolResult(tool="terminal", success=False, output="", error="Command timed out (60s limit)")
    except Exception as e:
        return ToolResult(tool="terminal", success=False, output="", error=str(e))

def execute_file_read(path: str) -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="file_read", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}")
        with open(path, 'r') as f:
            content = f.read(100000)
        return ToolResult(tool="file_read", success=True, output=content)
    except Exception as e:
        return ToolResult(tool="file_read", success=False, output="", error=str(e))

def execute_file_write(path: str, content: str) -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="file_write", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}")
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return ToolResult(tool="file_write", success=True, output=f"Written {len(content)} bytes to {path}")
    except Exception as e:
        return ToolResult(tool="file_write", success=False, output="", error=str(e))

def execute_list_files(path: str = "/tmp") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="list_files", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}")
        entries = []
        for entry in os.scandir(path):
            entry_type = "dir" if entry.is_dir() else "file"
            size = entry.stat().st_size if entry.is_file() else 0
            entries.append(f"{entry_type}\t{size}\t{entry.name}")
        return ToolResult(tool="list_files", success=True, output="\n".join(entries[:200]))
    except Exception as e:
        return ToolResult(tool="list_files", success=False, output="", error=str(e))

def execute_git(command: str, working_dir: str = "/tmp") -> ToolResult:
    try:
        allowed_git_commands = ["status", "log", "diff", "branch", "show", "ls-files", "rev-parse", "config --get"]
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return ToolResult(tool="git", success=False, output="", error="No git command provided")
        
        git_cmd = cmd_parts[0] if len(cmd_parts) == 1 else " ".join(cmd_parts[:2])
        if not any(git_cmd.startswith(allowed) for allowed in allowed_git_commands):
            return ToolResult(tool="git", success=False, output="", error=f"Git command not allowed. Allowed: {allowed_git_commands}")
        
        full_command = f"git {command}"
        result = subprocess.run(
            full_command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=working_dir
        )
        return ToolResult(tool="git", success=result.returncode == 0, output=result.stdout[:5000], error=result.stderr[:1000] if result.returncode != 0 else None)
    except Exception as e:
        return ToolResult(tool="git", success=False, output="", error=str(e))

def execute_code_analysis(path: str, analysis_type: str = "structure") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="code_analysis", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}")
        
        if not os.path.exists(path):
            return ToolResult(tool="code_analysis", success=False, output="", error=f"File not found: {path}")
        
        with open(path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        if analysis_type == "structure":
            functions = re.findall(r'(?:def|function|func)\s+(\w+)', content)
            classes = re.findall(r'(?:class)\s+(\w+)', content)
            imports = re.findall(r'(?:import|from|require|include)\s+[\w.]+', content)
            
            result = f"File: {path}\n"
            result += f"Lines: {len(lines)}\n"
            result += f"Classes: {', '.join(classes) if classes else 'None'}\n"
            result += f"Functions: {', '.join(functions) if functions else 'None'}\n"
            result += f"Imports: {len(imports)} found\n"
            return ToolResult(tool="code_analysis", success=True, output=result)
        
        elif analysis_type == "complexity":
            indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            max_indent = max(indent_levels) if indent_levels else 0
            avg_indent = sum(indent_levels) / len(indent_levels) if indent_levels else 0
            
            result = f"File: {path}\n"
            result += f"Total lines: {len(lines)}\n"
            result += f"Non-empty lines: {len([l for l in lines if l.strip()])}\n"
            result += f"Max nesting depth: {max_indent // 4}\n"
            result += f"Average indentation: {avg_indent:.1f} spaces\n"
            return ToolResult(tool="code_analysis", success=True, output=result)
        
        else:
            return ToolResult(tool="code_analysis", success=False, output="", error=f"Unknown analysis type: {analysis_type}")
    except Exception as e:
        return ToolResult(tool="code_analysis", success=False, output="", error=str(e))

def execute_search_files(path: str, pattern: str, file_pattern: str = "*") -> ToolResult:
    try:
        if not is_path_allowed(path):
            return ToolResult(tool="search_files", success=False, output="", error=f"Access denied. Allowed: {ALLOWED_DIRS}")
        
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
        return ToolResult(tool="search_files", success=False, output="", error=str(e))

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

## Working Directory
You can create and modify files in: /tmp and the workspace directory.

## Instructions
1. ALWAYS use tools - never just describe what you would do
2. After each tool execution, you'll see the results
3. Continue using tools until the task is complete
4. Provide a summary when done

## Example Task Flow
User: "Create a Python script that prints hello world and run it"

Your response should be:
"I'll create a Python script and run it.

```json
{"tool": "file_write", "path": "/tmp/hello.py", "content": "print('Hello, World!')"}
```"

Then after seeing the result, you continue:
"File created. Now I'll run it:

```json
{"tool": "terminal", "command": "python3 /tmp/hello.py"}
```"

REMEMBER: Always output the JSON tool block, never just describe what you would do!"""

def parse_tool_calls(response: str) -> list:
    tools = []
    pattern = r'```json\s*(\{[^`]+\})\s*```'
    matches = re.findall(pattern, response, re.DOTALL)
    for match in matches:
        try:
            match_clean = match.strip()
            tool_call = json.loads(match_clean)
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

def execute_tool(tool_call: dict) -> ToolResult:
    tool = tool_call.get("tool")
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
    else:
        return ToolResult(tool=str(tool), success=False, output="", error=f"Unknown tool: {tool}")

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

async def execute_agent_task(session_id: str, task_id: str, description: str, model: str, max_iterations: int = 10):
    global client
    conn = get_db()
    c = conn.cursor()
    
    if not client:
        c.execute("UPDATE tasks SET status=?, result=? WHERE task_id=?", ("failed", "OpenAI API key not configured", task_id))
        c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "error", "OpenAI API key not configured", datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        await broadcast_to_session(session_id, {"type": "task_failed", "task_id": task_id, "error": "OpenAI API key not configured"})
        return
    
    c.execute("UPDATE tasks SET status=?, started_at=? WHERE task_id=?", ("running", datetime.utcnow().isoformat(), task_id))
    conn.commit()
    
    await broadcast_to_session(session_id, {"type": "task_started", "task_id": task_id})
    
    c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "thinking", f"Analyzing task: {description}", datetime.utcnow().isoformat()))
    conn.commit()
    await broadcast_to_session(session_id, {"type": "thinking", "task_id": task_id, "content": f"Analyzing task: {description}"})
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": description}]
    iteration = 0
    final_response = ""
    
    try:
        while iteration < max_iterations:
            iteration += 1
            
            c.execute("UPDATE sessions SET iteration=? WHERE session_id=?", (iteration, session_id))
            conn.commit()
            await broadcast_to_session(session_id, {"type": "iteration", "task_id": task_id, "iteration": iteration, "max": max_iterations})
            
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=2000, temperature=0.7)
            agent_response = response.choices[0].message.content
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
                
                result = execute_tool(tool_call)
                tool_results.append(result)
                
                tool_output = {"tool": result.tool, "success": result.success, "output": result.output[:5000], "error": result.error}
                c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "tool", json.dumps(tool_output), datetime.utcnow().isoformat()))
                conn.commit()
                await broadcast_to_session(session_id, {"type": "tool_result", "task_id": task_id, "result": tool_output})
            
            tool_output_text = "\n\n".join([
                f"**Tool: {r.tool}**\nSuccess: {r.success}\n```\n{r.output}\n```\n{f'Error: {r.error}' if r.error else ''}"
                for r in tool_results
            ])
            
            messages.append({"role": "assistant", "content": agent_response})
            messages.append({"role": "user", "content": f"Tool execution results:\n\n{tool_output_text}\n\nContinue with the task. If you need to use more tools, output the JSON block. If the task is complete, provide a summary."})
        
        c.execute("UPDATE tasks SET status=?, completed_at=?, result=? WHERE task_id=?", ("completed", datetime.utcnow().isoformat(), final_response, task_id))
        conn.commit()
        await broadcast_to_session(session_id, {"type": "task_completed", "task_id": task_id, "result": final_response})
        
    except Exception as e:
        error_msg = str(e)
        c.execute("UPDATE tasks SET status=?, result=? WHERE task_id=?", ("failed", f"Error: {error_msg}", task_id))
        c.execute("INSERT INTO task_outputs (task_id, output_type, content, created_at) VALUES (?, ?, ?, ?)", (task_id, "error", error_msg, datetime.utcnow().isoformat()))
        conn.commit()
        await broadcast_to_session(session_id, {"type": "task_failed", "task_id": task_id, "error": error_msg})
    
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global client
    print("Starting Mini-Devin API v4.0...")
    print("Features: Tool Execution, SQLite Storage, WebSocket, Git, Code Analysis, Search")
    init_db()
    print(f"SQLite database initialized at {DB_PATH}")
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    print(f"Workspace directory: {WORKSPACE_DIR}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = openai.OpenAI(api_key=api_key)
        print(f"OpenAI API key configured (length: {len(api_key)} chars)")
    else:
        print("Warning: No OpenAI API key found - set OPENAI_API_KEY environment variable")
    yield
    print("Shutting down Mini-Devin API...")

app = FastAPI(title="Mini-Devin API", version="4.0.0", description="Autonomous AI Software Engineer with Tool Execution, WebSocket, and Code Analysis", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    api_key = os.environ.get("OPENAI_API_KEY")
    return {
        "name": "Mini-Devin API",
        "version": "4.0.0",
        "status": "running",
        "mode": "full-agent" if api_key else "limited",
        "llm_configured": bool(api_key),
        "features": ["tool_execution", "persistent_storage", "websocket", "git", "code_analysis", "search"],
        "tools": ["terminal", "file_read", "file_write", "list_files", "git", "code_analysis", "search_files"],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "4.0.0"}

@app.get("/api/health")
async def api_health():
    api_key = os.environ.get("OPENAI_API_KEY")
    return {"status": "healthy", "mode": "full-agent" if api_key else "limited", "llm_configured": bool(api_key), "version": "4.0.0"}

@app.get("/api/status")
async def get_status():
    api_key = os.environ.get("OPENAI_API_KEY")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sessions")
    session_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tasks WHERE status='completed'")
    completed_tasks = c.fetchone()[0]
    conn.close()
    return {
        "status": "running",
        "mode": "full-agent" if api_key else "limited",
        "version": "4.0.0",
        "active_sessions": session_count,
        "completed_tasks": completed_tasks,
        "llm_configured": bool(api_key),
        "features": ["tool_execution", "persistent_storage", "websocket", "git", "code_analysis", "search"]
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
    api_key = os.environ.get("OPENAI_API_KEY")
    session_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().isoformat()
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO sessions (session_id, created_at, status, working_directory, model, max_iterations, iteration, total_tasks) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (session_id, created_at, "active", request.working_directory, request.model, request.max_iterations, 0, 0))
    conn.commit()
    conn.close()
    return {"session_id": session_id, "created_at": created_at, "status": "active", "llm_enabled": bool(api_key)}

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
    c.execute("SELECT model, max_iterations FROM sessions WHERE session_id=?", (session_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    model, max_iterations = row
    api_key = os.environ.get("OPENAI_API_KEY")
    task_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().isoformat()
    c.execute("INSERT INTO tasks (task_id, session_id, description, status, created_at) VALUES (?, ?, ?, ?, ?)",
              (task_id, session_id, request.description, "queued", created_at))
    c.execute("UPDATE sessions SET total_tasks = total_tasks + 1, current_task = ? WHERE session_id = ?", (task_id, session_id))
    conn.commit()
    conn.close()
    if api_key:
        asyncio.create_task(execute_agent_task(session_id, task_id, request.description, model, max_iterations))
    return {"task_id": task_id, "session_id": session_id, "description": request.description, "status": "queued", "created_at": created_at}

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
            {"name": "search_files", "description": "Search for text in files", "params": ["path", "pattern", "file_pattern"]}
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
