# Mini-Devin

An autonomous AI software engineer agent with access to Terminal, Code Editor, and Web Browser.

## Overview

Mini-Devin is an AI-powered autonomous agent that can solve software engineering tasks by using tools to interact with your codebase. It can:

- Execute shell commands in a sandboxed environment
- Read, write, search, and modify code files with LSP support
- Search the web, fetch documentation, and interact with websites
- Plan complex tasks and break them into steps
- Review its own code changes before committing
- Learn from reusable skills and procedures

## Features

### Core Capabilities

- **Terminal Tool**: Execute shell commands with safety guards and command blocking
- **Editor Tool**: Full file operations with LSP integration (go-to-definition, find references, hover info)
- **Browser Tools**: Web search (Tavily/SerpAPI), page fetching, and Selenium-based interaction
- **Memory System**: Code symbol indexing, vector store for semantic search, working memory

### Agent Intelligence

- **Planner Agent**: Creates detailed execution plans before starting work
- **Reviewer Agent**: Reviews code changes and provides feedback before commits
- **Gates System**: Enforces planning and review requirements with configurable policies
- **Repair Loop**: Automatic fix attempts with bounded iterations

### Safety & Security

- **Safety Guards**: Blocks dangerous commands, large edits, and unauthorized dependency changes
- **Docker Sandbox**: Non-root user, resource limits, capability restrictions, optional read-only filesystem
- **Secrets Management**: Fernet encryption (AES-128-CBC) with PBKDF2 key derivation
- **Stop Conditions**: Automatic stopping on errors, timeouts, or dangerous operations

### Production Features

- **Web Dashboard**: React + TypeScript frontend with real-time streaming
- **Multi-Model Support**: OpenAI, Anthropic Claude, Azure OpenAI, Ollama (local models)
- **Database Persistence**: PostgreSQL with SQLAlchemy/Alembic migrations
- **Authentication**: JWT tokens and API key authentication
- **Skills Library**: Reusable procedures for common tasks

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Docker and Docker Compose (for containerized deployment)
- An LLM API key (OpenAI, Anthropic, or Ollama for local models)

### Installation

```bash
# Clone the repository
git clone https://github.com/naimprince010-ship-it/mini-devin.git
cd mini-devin

# Install dependencies
poetry install

# Install Playwright browsers (for browser tool)
poetry run playwright install

# Copy environment configuration
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

### Running Mini-Devin

**CLI Mode:**
```bash
# Basic task execution
poetry run mini-devin run "Fix the failing test in tests/test_api.py" --dir ./my-project

# Interactive mode
poetry run mini-devin interactive --dir ./my-project

# With specific run mode
RUN_MODE=browse poetry run mini-devin run "Research and implement a caching solution" --dir ./my-project
```

**Web Dashboard:**
```bash
# Start the backend
poetry run uvicorn mini_devin.api.app:app --reload --port 8000

# Start the frontend (in another terminal)
cd frontend
npm install
npm run dev

# Open http://localhost:5173 in your browser
```

**Docker Mode:**
```bash
# Start with Docker Compose
docker-compose up -d

# Run a task
docker-compose exec mini-devin mini-devin run "Your task here"

# With interactive browser (Selenium)
docker-compose --profile interactive up -d
```

## Architecture

Mini-Devin consists of five core layers:

```
+--------------------------------------------------+
|              Web Dashboard (React)               |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|              FastAPI Backend (REST/WS)           |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|              Agent Runtime Layer                 |
|  +------------+  +------------+  +------------+  |
|  |  Planner   |  |   Agent    |  |  Reviewer  |  |
|  |   Agent    |  |   Loop     |  |   Agent    |  |
|  +------------+  +------------+  +------------+  |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|                 Tooling Layer                    |
|  +----------+  +----------+  +----------------+  |
|  | Terminal |  |  Editor  |  | Browser Tools  |  |
|  +----------+  +----------+  +----------------+  |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|           Memory & Context Layer                 |
|  +-------------+  +-------------+  +-----------+ |
|  | Symbol Index|  | Vector Store|  |  Working  | |
|  |             |  |             |  |  Memory   | |
|  +-------------+  +-------------+  +-----------+ |
+--------------------------------------------------+
                        |
+--------------------------------------------------+
|           Safety & Sandbox Layer                 |
|  +------------+  +------------+  +------------+  |
|  |   Guards   |  |  Sandbox   |  |  Secrets   |  |
|  +------------+  +------------+  +------------+  |
+--------------------------------------------------+
```

## Project Structure

```
mini-devin/
├── mini_devin/
│   ├── agents/           # Planner and Reviewer agents
│   ├── api/              # FastAPI backend (routes, WebSocket)
│   ├── auth/             # Authentication (JWT, API keys)
│   ├── config/           # Settings and configuration
│   ├── core/             # LLM client, providers, tool interface
│   ├── database/         # SQLAlchemy models and repository
│   ├── evaluation/       # Benchmark and evaluation harness
│   ├── lsp/              # Language Server Protocol integration
│   ├── memory/           # Symbol index, vector store, working memory
│   ├── orchestrator/     # Main agent loop and state machine
│   ├── reliability/      # Repair signals, diff discipline
│   ├── safety/           # Guards and stop conditions
│   ├── sandbox/          # Docker sandbox management
│   ├── schemas/          # Pydantic schemas for tools and state
│   ├── secrets/          # Encrypted secrets management
│   ├── sessions/         # Session and task management
│   ├── skills/           # Skills library and built-in skills
│   ├── tools/            # Terminal, Editor, Browser tools
│   └── verification/     # Lint, test, typecheck runners
├── frontend/             # React + TypeScript dashboard
├── tests/
│   ├── unit/             # Unit tests (202 tests)
│   ├── e2e/              # End-to-end tests
│   └── acceptance/       # Acceptance test scenarios
├── alembic/              # Database migrations
├── docker-compose.yml    # Docker deployment
├── Dockerfile            # Container image
└── pyproject.toml        # Python dependencies
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes (or alternative) |
| `ANTHROPIC_API_KEY` | Anthropic API key | Alternative to OpenAI |
| `OLLAMA_BASE_URL` | Ollama server URL | For local models |
| `DATABASE_URL` | PostgreSQL connection string | For persistence |
| `JWT_SECRET_KEY` | Secret for JWT tokens | For authentication |

### LLM Provider Configuration

Mini-Devin supports multiple LLM providers:

```bash
# OpenAI (default)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...

# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...

# Ollama (local)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Azure OpenAI
LLM_PROVIDER=azure
LLM_MODEL=gpt-4o
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Run Modes

| Mode | Tools Available | Use Case |
|------|-----------------|----------|
| `offline` | Terminal, Editor | Local code changes |
| `browse` | + Search, Fetch | Research and documentation |
| `interactive` | + Selenium | Web app testing |

## Documentation

- [Operations Guide](OPERATIONS.md) - Running and maintaining Mini-Devin
- [API Documentation](docs/API.md) - REST and WebSocket API reference
- [Developer Guide](docs/DEVELOPER.md) - Contributing and extending Mini-Devin
- [Architecture](docs/ARCHITECTURE.md) - Detailed system architecture

## Development

### Running Tests

```bash
# Run all unit tests
poetry run pytest tests/unit/ -v

# Run with coverage
poetry run pytest tests/unit/ --cov=mini_devin

# Run E2E tests
poetry run pytest tests/e2e/ -v
```

### Linting and Type Checking

```bash
# Run ruff linter
poetry run ruff check mini_devin/

# Run mypy type checker
poetry run mypy mini_devin/

# Run frontend linting
cd frontend && npm run lint
```

### Database Migrations

```bash
# Create a new migration
poetry run alembic revision --autogenerate -m "Description"

# Apply migrations
poetry run alembic upgrade head

# Rollback one migration
poetry run alembic downgrade -1
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `poetry run pytest tests/unit/ -v`
5. Run linting: `poetry run ruff check mini_devin/`
6. Commit your changes: `git commit -m "Add my feature"`
7. Push to the branch: `git push origin feature/my-feature`
8. Create a Pull Request

## License

MIT

## Acknowledgments

Mini-Devin is inspired by autonomous AI coding agents like Devin by Cognition AI. This project demonstrates how to build a similar system from scratch using modern Python, FastAPI, React, and LLM APIs.
