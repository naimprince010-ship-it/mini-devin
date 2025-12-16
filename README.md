# Mini-Devin

An autonomous AI software engineer agent with access to Terminal, Code Editor, and Web Browser.

## Overview

Mini-Devin is an AI agent that can autonomously solve software engineering tasks by using tools to interact with:
- **Terminal**: Execute shell commands in a sandboxed environment
- **Code Editor**: Read, write, search, and modify code files with LSP support
- **Web Browser**: Search the web, fetch pages, and interact with websites

## Architecture

The system consists of five core layers:

1. **Agent Runtime (Orchestration Layer)**: State machine controlling the agent workflow
2. **Tooling Layer**: Terminal, Editor, and Browser tools with strict schemas
3. **Workspace & Sandbox Layer**: Isolated execution environment
4. **Memory & Context Management**: Short-term, project, and long-term memory
5. **Observability & Evaluation Layer**: Tracing and verification

## Project Structure

```
mini-devin/
├── mini_devin/
│   ├── __init__.py
│   ├── schemas/           # Pydantic schemas
│   │   ├── tools.py       # Tool input/output schemas
│   │   ├── state.py       # Agent and task state schemas
│   │   └── verification.py # Verification and done signals
│   ├── core/              # Core infrastructure
│   │   └── tool_interface.py  # Base tool interface and registry
│   ├── tools/             # Tool implementations (Phase 1)
│   ├── memory/            # Memory management (Phase 4)
│   ├── orchestrator/      # Agent orchestration (Phase 1)
│   └── sandbox/           # Execution sandbox (Phase 2)
├── tests/
├── configs/
└── pyproject.toml
```

## Installation

```bash
# Install dependencies
poetry install

# Install Playwright browsers (for browser tool)
poetry run playwright install
```

## Development Roadmap

- **Phase 0** (Current): Define tool schemas and project structure
- **Phase 1**: MVP agent loop with terminal and editor tools
- **Phase 2**: Verification and recovery mechanisms
- **Phase 3**: Browser integration
- **Phase 4**: Memory and indexing
- **Phase 5**: Productization (web UI, multi-session)
- **Phase 6**: Advanced autonomy (multi-agent, skills library)

## Usage (Coming in Phase 1)

```python
from mini_devin import AgentState, TaskState, ToolRegistry
from mini_devin.core import BaseTool, register_tool

# Create a task
task = TaskState(
    task_id="fix-bug-123",
    goal=TaskGoal(
        description="Fix the null pointer exception in user service",
        acceptance_criteria=["Tests pass", "No new lint errors"],
    ),
)

# Run the agent (Phase 1)
# agent = Agent(task)
# result = await agent.run()
```

## License

MIT
