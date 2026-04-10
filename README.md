# Mini-Devin 🤖

**An autonomous AI software engineer** — give it a task in plain English, it writes code, runs tests, fixes errors, and commits to Git — all automatically.

![Mini-Devin UI](docs/screenshot.png)

---

## What it does

| Capability | Description |
|-----------|-------------|
| 🖥️ **Terminal** | Runs shell commands, installs packages, executes tests |
| 📝 **Code Editor** | Creates, reads, edits files across any language |
| 🌐 **Web Browser** | Searches the web, reads docs, fetches pages |
| 🧠 **Memory** | Remembers previous work, learns from past tasks |
| 🔁 **Self-Correction** | Detects errors and fixes them automatically (up to 3 retries) |
| 🔀 **Git Integration** | Auto-commits and pushes after task completion |
| ✅ **Verification** | Runs lint + tests and auto-repairs failures |

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Node 18+
- [Poetry](https://python-poetry.org/docs/#installation)

### 1. Clone and configure
```bash
git clone https://github.com/yourname/mini-devin.git
cd mini-devin
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Install dependencies
```bash
poetry install
cd frontend && npm install && cd ..
```

### 3. Start backend
```bash
poetry run uvicorn mini_devin.api.app:app --host 127.0.0.1 --port 8000
```

### 4. Start frontend (new terminal)
```bash
cd frontend
npm run dev
```

### 5. Open browser
```
http://localhost:5173
```

---

## Quick Start (Docker)

One command to run everything:

```bash
# 1. Copy and configure
cp .env.example .env
# Edit .env: set OPENAI_API_KEY

# 2. Start all services
docker compose up --build

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

To run with Selenium browser support:
```bash
docker compose --profile interactive up --build
```

---

## How to use

1. **Create a Session** — click **+** and set:
   - Working Directory (where files will be created)
   - Model (gpt-4o, gpt-4o-mini, claude-3-5-sonnet, etc.)
   - Git options (auto-commit, push to remote)

2. **Give a task** — type in plain English:
   ```
   Create a FastAPI web app with user authentication using JWT.
   Include a /health endpoint and full pytest test suite.
   ```

3. **Watch it work** — the agent:
   - Plans the steps
   - Writes the code
   - Runs tests
   - Fixes any errors
   - Reports results

4. **View files** — click the **IDE** tab to see and edit created files

---

## Example Prompts

```
Build a Python web scraper for news.ycombinator.com that saves results to CSV
```

```
Fix the bug in calculator.py where divide by zero crashes the program
```

```
Add dark mode to this React app. Use Tailwind CSS.
```

```
Create a REST API with CRUD operations for a todo list using FastAPI + SQLite
```

---

## Architecture

```
mini-devin/
├── mini_devin/
│   ├── api/              # FastAPI backend (WebSocket + REST)
│   ├── orchestrator/     # Agent loop, planning, LLM calls
│   ├── tools/            # Terminal, Editor, Browser, GitHub
│   ├── memory/           # Vector store, symbol index, working memory
│   ├── verification/     # Lint/test runner + LLM repair loop
│   ├── integrations/     # GitHub, Vercel, test fix loop
│   ├── sessions/         # Session management + database
│   └── core/             # LLM client, providers, tool interface
├── frontend/             # React + TypeScript + Tailwind UI
├── docker-compose.yml    # Full stack deployment
└── .env.example          # Configuration template
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required for GPT models) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (for Claude models) |
| `LLM_MODEL` | `gpt-4o` | Default model |
| `GITHUB_TOKEN` | — | For auto-commit/push |
| `TAVILY_API_KEY` | — | Web search (optional) |
| `MAX_ITERATIONS` | `50` | Max agent iterations per task |
| `WORKSPACE_DIR` | `./workspace` | Default project directory |

---

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| Anthropic | claude-3-5-sonnet, claude-3-haiku |
| Ollama | llama3, codellama (local) |
| Azure OpenAI | via LiteLLM |

---

## License

MIT
