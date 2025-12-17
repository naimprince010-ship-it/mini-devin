# Mini-Devin Architecture

This document provides a detailed overview of Mini-Devin's system architecture, component interactions, and design decisions.

## System Overview

Mini-Devin is an autonomous AI software engineer agent that uses Large Language Models (LLMs) to understand tasks and execute them using a set of tools. The system is designed with safety, reliability, and extensibility as core principles.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │     CLI     │  │  Web UI     │  │      REST/WS API        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Session Manager                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Session   │  │    Task     │  │      Persistence        │ │
│  │   State     │  │   Queue     │  │      (PostgreSQL)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Runtime                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Planner   │  │    Agent    │  │       Reviewer          │ │
│  │   Agent     │  │    Loop     │  │       Agent             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                         │                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    Gates    │  │   Repair    │  │      Verification       │ │
│  │   System    │  │    Loop     │  │       Runner            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Tool Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Terminal   │  │   Editor    │  │    Browser Tools        │ │
│  │   Tool      │  │   Tool      │  │  (Search/Fetch/Interact)│ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                         │                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    LSP      │  │   Skills    │  │      Tool Registry      │ │
│  │   Tools     │  │   Library   │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Memory Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Symbol    │  │   Vector    │  │      Working            │ │
│  │   Index     │  │   Store     │  │      Memory             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Safety & Sandbox                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Safety    │  │   Docker    │  │      Secrets            │ │
│  │   Guards    │  │   Sandbox   │  │      Manager            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent Runtime

The Agent Runtime is the heart of Mini-Devin, responsible for orchestrating task execution.

#### Agent Loop (`mini_devin/orchestrator/agent.py`)

The main agent loop follows this state machine:

```
                    ┌─────────────┐
                    │   IDLE      │
                    └──────┬──────┘
                           │ start_task()
                           ▼
                    ┌─────────────┐
                    │  PLANNING   │◄─────────────┐
                    └──────┬──────┘              │
                           │ plan_created        │ replan
                           ▼                     │
                    ┌─────────────┐              │
              ┌────►│  EXECUTING  │──────────────┘
              │     └──────┬──────┘
              │            │ step_complete
              │            ▼
              │     ┌─────────────┐
              │     │  VERIFYING  │
              │     └──────┬──────┘
              │            │
              │     ┌──────┴──────┐
              │     │             │
              │     ▼             ▼
              │  success       failure
              │     │             │
              │     ▼             ▼
              │ ┌─────────┐  ┌─────────────┐
              │ │REVIEWING│  │  REPAIRING  │
              │ └────┬────┘  └──────┬──────┘
              │      │              │
              │      ▼              │ max_retries
              │   approved         │
              │      │             ▼
              │      ▼        ┌─────────┐
              │ ┌─────────┐   │ BLOCKED │
              │ │COMPLETED│   └─────────┘
              │ └─────────┘
              │
              └──── next_step
```

**Key Methods:**
- `run()`: Main entry point, executes the full task
- `_plan_task()`: Creates execution plan using Planner Agent
- `_execute_step()`: Executes a single plan step
- `_verify_changes()`: Runs lint, test, typecheck
- `_review_changes()`: Gets feedback from Reviewer Agent
- `_repair_issues()`: Attempts to fix verification failures

#### Planner Agent (`mini_devin/agents/planner.py`)

The Planner Agent analyzes tasks and creates structured execution plans:

```python
class PlannerAgent:
    async def create_plan(self, task: str, context: dict) -> ExecutionPlan:
        """
        1. Analyze the task description
        2. Identify required tools and resources
        3. Break down into discrete steps
        4. Estimate complexity and dependencies
        5. Return structured plan
        """
```

**Plan Structure:**
```json
{
  "task_id": "uuid",
  "steps": [
    {
      "id": "step-1",
      "description": "Read the failing test file",
      "tool": "editor",
      "expected_outcome": "Understand test structure",
      "dependencies": []
    },
    {
      "id": "step-2", 
      "description": "Identify the root cause",
      "tool": "terminal",
      "expected_outcome": "Find the bug",
      "dependencies": ["step-1"]
    }
  ],
  "estimated_iterations": 5,
  "risk_assessment": "low"
}
```

#### Reviewer Agent (`mini_devin/agents/reviewer.py`)

The Reviewer Agent evaluates code changes before committing:

```python
class ReviewerAgent:
    async def review_diff(self, diff: str, context: dict) -> ReviewResult:
        """
        1. Analyze the diff for correctness
        2. Check for common issues (security, performance)
        3. Verify alignment with task goals
        4. Provide feedback and severity rating
        """
```

**Review Result:**
```json
{
  "approved": false,
  "severity": "high",
  "findings": [
    {
      "type": "security",
      "description": "SQL injection vulnerability",
      "location": "src/db.py:45",
      "suggestion": "Use parameterized queries"
    }
  ],
  "summary": "Changes introduce security vulnerability"
}
```

### 2. Tool Layer

Tools are the interface between the agent and the environment.

#### Base Tool Interface (`mini_devin/core/tool_interface.py`)

```python
class BaseTool(Generic[TInput, TOutput]):
    name: str
    description: str
    input_schema: Type[TInput]
    output_schema: Type[TOutput]
    
    def execute(self, input_data: TInput | dict) -> TOutput:
        """Execute the tool with given input."""
        pass
    
    def get_json_schema(self) -> dict:
        """Return JSON schema for LLM function calling."""
        pass
```

#### Terminal Tool (`mini_devin/tools/terminal.py`)

Executes shell commands with safety guards:

```
┌─────────────────────────────────────────┐
│            Terminal Tool                │
├─────────────────────────────────────────┤
│  Input Validation                       │
│  ├── Command blocklist check            │
│  ├── Working directory validation       │
│  └── Timeout configuration              │
├─────────────────────────────────────────┤
│  Execution                              │
│  ├── asyncio subprocess                 │
│  ├── stdout/stderr capture              │
│  └── Exit code handling                 │
├─────────────────────────────────────────┤
│  Output Processing                      │
│  ├── Truncation (max 10000 chars)       │
│  ├── Secret redaction                   │
│  └── Error classification               │
└─────────────────────────────────────────┘
```

**Blocked Commands:**
- `rm -rf /`, `rm -rf /*`
- Fork bombs: `:(){ :|:& };:`
- Force push: `git push --force`
- Disk operations: `mkfs`, `dd if=/dev/zero`

#### Editor Tool (`mini_devin/tools/editor.py`)

File operations with LSP integration:

```
┌─────────────────────────────────────────┐
│             Editor Tool                 │
├─────────────────────────────────────────┤
│  File Operations                        │
│  ├── read_file(path)                    │
│  ├── write_file(path, content)          │
│  ├── patch_file(path, diff)             │
│  ├── search_files(pattern, path)        │
│  └── list_directory(path)               │
├─────────────────────────────────────────┤
│  LSP Integration                        │
│  ├── goto_definition(file, line, col)   │
│  ├── find_references(symbol)            │
│  ├── hover_info(file, line, col)        │
│  └── diagnostics(file)                  │
├─────────────────────────────────────────┤
│  Safety                                 │
│  ├── Max lines per edit (300)           │
│  ├── Path validation                    │
│  └── Backup before write                │
└─────────────────────────────────────────┘
```

#### Browser Tools (`mini_devin/tools/browser/`)

Three-tier browser capability:

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser Tools                           │
├─────────────────────────────────────────────────────────────┤
│  Search Tool (API-based)                                    │
│  ├── Tavily API                                             │
│  ├── SerpAPI                                                │
│  └── DuckDuckGo (fallback)                                  │
├─────────────────────────────────────────────────────────────┤
│  Fetch Tool (Headless HTTP)                                 │
│  ├── httpx for requests                                     │
│  ├── Readability extraction                                 │
│  └── Citation storage                                       │
├─────────────────────────────────────────────────────────────┤
│  Interactive Tool (Selenium)                                │
│  ├── Full browser automation                                │
│  ├── JavaScript execution                                   │
│  └── Form interaction                                       │
└─────────────────────────────────────────────────────────────┘
```

### 3. Memory Layer

The memory system provides context and knowledge retrieval.

#### Symbol Index (`mini_devin/memory/symbol_index.py`)

Indexes code symbols for fast lookup:

```
┌─────────────────────────────────────────┐
│            Symbol Index                 │
├─────────────────────────────────────────┤
│  Indexing                               │
│  ├── Regex-based parsing                │
│  ├── Language detection                 │
│  └── Incremental updates                │
├─────────────────────────────────────────┤
│  Symbol Types                           │
│  ├── Functions/Methods                  │
│  ├── Classes                            │
│  ├── Variables/Constants                │
│  └── Imports                            │
├─────────────────────────────────────────┤
│  Queries                                │
│  ├── find_symbol(name)                  │
│  ├── find_references(symbol)            │
│  └── get_file_symbols(path)             │
└─────────────────────────────────────────┘
```

#### Vector Store (`mini_devin/memory/vector_store.py`)

Semantic search using embeddings:

```
┌─────────────────────────────────────────┐
│            Vector Store                 │
├─────────────────────────────────────────┤
│  Embedding                              │
│  ├── OpenAI text-embedding-3-small      │
│  ├── Chunking (512 tokens)              │
│  └── Metadata attachment                │
├─────────────────────────────────────────┤
│  Storage                                │
│  ├── In-memory (default)                │
│  ├── ChromaDB (optional)                │
│  └── Pinecone (optional)                │
├─────────────────────────────────────────┤
│  Retrieval                              │
│  ├── Cosine similarity                  │
│  ├── Top-k results                      │
│  └── Metadata filtering                 │
└─────────────────────────────────────────┘
```

#### Working Memory (`mini_devin/memory/working_memory.py`)

Short-term context for the current task:

```python
class WorkingMemory:
    """Maintains context for the current task execution."""
    
    # Recent tool calls and results
    tool_history: list[ToolCall]
    
    # Current plan and progress
    current_plan: ExecutionPlan
    completed_steps: list[str]
    
    # Relevant code snippets
    code_context: list[CodeSnippet]
    
    # Error history for repair loop
    error_history: list[Error]
```

### 4. Safety Layer

Multiple layers of protection against harmful operations.

#### Safety Guards (`mini_devin/safety/guards.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                    Safety Guards                            │
├─────────────────────────────────────────────────────────────┤
│  Command Checks                                             │
│  ├── Blocklist matching                                     │
│  ├── Pattern detection (rm -rf, fork bombs)                 │
│  └── Sudo/privilege escalation                              │
├─────────────────────────────────────────────────────────────┤
│  File Edit Checks                                           │
│  ├── Max lines per edit (300)                               │
│  ├── Max files per delete (1)                               │
│  └── Protected paths (/etc, /usr, etc.)                     │
├─────────────────────────────────────────────────────────────┤
│  Dependency Checks                                          │
│  ├── package.json modifications                             │
│  ├── requirements.txt modifications                         │
│  └── Cargo.toml modifications                               │
├─────────────────────────────────────────────────────────────┤
│  Iteration Limits                                           │
│  ├── Max iterations (50)                                    │
│  ├── Max repair iterations (3)                              │
│  └── Consecutive error limit (5)                            │
└─────────────────────────────────────────────────────────────┘
```

#### Docker Sandbox (`mini_devin/sandbox/docker_sandbox.py`)

Container-based isolation:

```yaml
Security Features:
  - Non-root user (UID 1000)
  - Dropped capabilities
  - Resource limits (CPU, memory, PIDs)
  - Optional read-only filesystem
  - Network isolation
  - No new privileges
```

### 5. LLM Integration

#### Provider Abstraction (`mini_devin/core/providers.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                   Provider Registry                         │
├─────────────────────────────────────────────────────────────┤
│  Supported Providers                                        │
│  ├── OpenAI (GPT-4o, GPT-4o-mini, o1)                      │
│  ├── Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)          │
│  ├── Azure OpenAI                                           │
│  └── Ollama (local models)                                  │
├─────────────────────────────────────────────────────────────┤
│  Model Registry                                             │
│  ├── Model capabilities (vision, function calling)          │
│  ├── Context window sizes                                   │
│  └── Provider-specific configuration                        │
└─────────────────────────────────────────────────────────────┘
```

#### LLM Client (`mini_devin/core/llm_client.py`)

Unified interface using LiteLLM:

```python
class LLMClient:
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        stream: bool = False
    ) -> CompletionResult:
        """
        1. Format messages for provider
        2. Add tool schemas if provided
        3. Call LiteLLM completion
        4. Parse response and tool calls
        5. Return structured result
        """
```

## Data Flow

### Task Execution Flow

```
User Request
     │
     ▼
┌─────────────┐
│   Parse     │
│   Task      │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Create    │────►│   Planner   │
│   Plan      │◄────│   Agent     │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│  Execute    │◄──────────────────┐
│   Step      │                   │
└──────┬──────┘                   │
       │                          │
       ▼                          │
┌─────────────┐     ┌─────────────┐
│   Tool      │────►│   Tool      │
│   Call      │◄────│   Execution │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│  Verify     │
│  Changes    │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
 Pass    Fail
   │       │
   ▼       ▼
┌─────┐ ┌─────────┐
│Review│ │ Repair  │────┐
└──┬──┘ └────┬────┘    │
   │         │         │
   │         ▼         │
   │    ┌─────────┐    │
   │    │ Retry?  │────┘
   │    └────┬────┘
   │         │ max_retries
   │         ▼
   │    ┌─────────┐
   │    │ BLOCKED │
   │    └─────────┘
   │
   ▼
┌─────────────┐
│  Commit     │
│  Changes    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Complete   │
└─────────────┘
```

### WebSocket Streaming Flow

```
Client                    Server                    Agent
  │                         │                         │
  │──── Connect ───────────►│                         │
  │                         │                         │
  │◄─── Authenticated ──────│                         │
  │                         │                         │
  │──── Start Task ────────►│                         │
  │                         │──── Create Task ───────►│
  │                         │                         │
  │                         │◄─── Plan Created ───────│
  │◄─── Plan Update ────────│                         │
  │                         │                         │
  │                         │◄─── Token Stream ───────│
  │◄─── Token ──────────────│                         │
  │◄─── Token ──────────────│                         │
  │                         │                         │
  │                         │◄─── Tool Call ──────────│
  │◄─── Tool Call ──────────│                         │
  │                         │                         │
  │                         │◄─── Tool Result ────────│
  │◄─── Tool Result ────────│                         │
  │                         │                         │
  │                         │◄─── Status: Complete ───│
  │◄─── Status ─────────────│                         │
  │                         │                         │
```

## Configuration

### Environment-Based Configuration

```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Hierarchy                  │
├─────────────────────────────────────────────────────────────┤
│  1. Environment Variables (highest priority)                │
│     └── OPENAI_API_KEY, LLM_MODEL, etc.                    │
│                                                             │
│  2. .env File                                               │
│     └── Local development settings                          │
│                                                             │
│  3. Default Values (lowest priority)                        │
│     └── Defined in config/settings.py                       │
└─────────────────────────────────────────────────────────────┘
```

### Settings Classes

```python
# mini_devin/config/settings.py

class LLMSettings(BaseSettings):
    provider: Provider = Provider.OPENAI
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096

class SafetySettings(BaseSettings):
    max_iterations: int = 50
    max_repair_iterations: int = 3
    max_lines_edit: int = 300
    allow_dependency_bump: bool = False

class SandboxSettings(BaseSettings):
    enabled: bool = True
    run_as_non_root: bool = True
    read_only_root: bool = False
    network_isolation: bool = False
```

## Extensibility Points

### Adding New Tools

1. Define input/output schemas in `schemas/tools.py`
2. Implement tool class extending `BaseTool`
3. Register in tool registry
4. Add safety checks if needed

### Adding New Providers

1. Add provider enum in `core/providers.py`
2. Add model information to registry
3. Update LLM client for provider-specific handling
4. Add configuration options

### Adding New Skills

1. Create skill class extending `Skill`
2. Define parameters and steps
3. Register in skill registry
4. Add tests

## Performance Considerations

### Caching

- Symbol index cached per file (invalidated on change)
- Vector embeddings cached with TTL
- LLM responses not cached (non-deterministic)

### Concurrency

- Async/await throughout for I/O operations
- Tool execution is sequential (safety)
- Multiple sessions can run in parallel

### Resource Limits

- Docker container limits (CPU, memory, PIDs)
- LLM token limits per request
- File size limits for editor operations
- Timeout limits for tool execution

## Security Model

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│  Untrusted Zone                                             │
│  ├── User input (task descriptions)                         │
│  ├── LLM outputs (tool calls, code)                         │
│  └── External data (web content)                            │
├─────────────────────────────────────────────────────────────┤
│  Validation Layer                                           │
│  ├── Input sanitization                                     │
│  ├── Safety guards                                          │
│  └── Schema validation                                      │
├─────────────────────────────────────────────────────────────┤
│  Trusted Zone                                               │
│  ├── Core agent logic                                       │
│  ├── Tool implementations                                   │
│  └── Configuration                                          │
└─────────────────────────────────────────────────────────────┘
```

### Secrets Handling

- Secrets encrypted at rest (Fernet/AES-128-CBC)
- PBKDF2 key derivation (480k iterations)
- Secrets redacted from logs and artifacts
- API keys never exposed to LLM
