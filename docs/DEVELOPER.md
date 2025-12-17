# Mini-Devin Developer Guide

This guide covers how to contribute to Mini-Devin, extend its functionality, and understand its internal architecture.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Poetry 1.7+
- Node.js 20+ (for frontend)
- Docker and Docker Compose (optional, for containerized development)
- PostgreSQL 15+ (for database features)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/naimprince010-ship-it/mini-devin.git
cd mini-devin

# Install Python dependencies
poetry install

# Install development dependencies
poetry install --with dev

# Install Playwright browsers
poetry run playwright install

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Set up the database (optional)
docker-compose up -d postgres
poetry run alembic upgrade head

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running in Development Mode

**Backend:**
```bash
# Start the FastAPI server with hot reload
poetry run uvicorn mini_devin.api.app:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

**Full Stack with Docker:**
```bash
docker-compose up -d
```

## Project Architecture

### Directory Structure

```
mini_devin/
├── agents/              # AI agents (planner, reviewer)
│   ├── planner.py       # Task planning and decomposition
│   └── reviewer.py      # Code review and feedback
├── api/                 # FastAPI application
│   ├── app.py           # Application factory
│   ├── routes.py        # REST endpoints
│   └── websocket.py     # WebSocket handlers
├── auth/                # Authentication
│   ├── jwt.py           # JWT token handling
│   └── api_keys.py      # API key management
├── config/              # Configuration
│   ├── settings.py      # Application settings
│   └── sandbox.py       # Sandbox configuration
├── core/                # Core infrastructure
│   ├── llm_client.py    # LLM API client
│   ├── providers.py     # Provider registry
│   └── tool_interface.py # Base tool interface
├── database/            # Database layer
│   ├── models.py        # SQLAlchemy models
│   └── repository.py    # Data access layer
├── lsp/                 # Language Server Protocol
│   ├── client.py        # LSP client
│   ├── tools.py         # LSP-based tools
│   └── types.py         # LSP type definitions
├── memory/              # Memory systems
│   ├── symbol_index.py  # Code symbol indexing
│   ├── vector_store.py  # Vector embeddings
│   └── working_memory.py # Short-term memory
├── orchestrator/        # Agent orchestration
│   └── agent.py         # Main agent loop
├── safety/              # Safety mechanisms
│   ├── guards.py        # Safety guards
│   └── stop_conditions.py # Stop condition checks
├── schemas/             # Pydantic schemas
│   ├── tools.py         # Tool input/output schemas
│   ├── state.py         # Agent state schemas
│   └── verification.py  # Verification schemas
├── secrets/             # Secrets management
│   └── manager.py       # Encrypted secrets
├── sessions/            # Session management
│   ├── manager.py       # In-memory sessions
│   └── db_manager.py    # Database-backed sessions
├── skills/              # Skills library
│   ├── base.py          # Base skill class
│   ├── registry.py      # Skill registry
│   └── builtin/         # Built-in skills
├── tools/               # Tool implementations
│   ├── terminal.py      # Terminal tool
│   ├── editor.py        # Editor tool
│   └── browser/         # Browser tools
└── verification/        # Verification runners
    ├── runner.py        # Verification orchestration
    └── repair.py        # Repair loop
```

## Adding New Tools

Tools are the primary way Mini-Devin interacts with the environment. Here's how to create a new tool:

### 1. Define the Schema

Create input/output schemas in `mini_devin/schemas/tools.py`:

```python
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    """Input schema for MyTool."""
    param1: str = Field(..., description="First parameter")
    param2: int = Field(default=10, description="Second parameter")

class MyToolOutput(BaseModel):
    """Output schema for MyTool."""
    result: str
    success: bool
```

### 2. Implement the Tool

Create the tool in `mini_devin/tools/my_tool.py`:

```python
from mini_devin.core.tool_interface import BaseTool
from mini_devin.schemas.tools import MyToolInput, MyToolOutput

class MyTool(BaseTool[MyToolInput, MyToolOutput]):
    """My custom tool description."""
    
    name = "my_tool"
    description = "Does something useful"
    input_schema = MyToolInput
    output_schema = MyToolOutput
    
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config or {}
    
    def execute(self, input_data: MyToolInput | dict) -> MyToolOutput:
        """Execute the tool."""
        # Validate input
        if isinstance(input_data, dict):
            input_data = MyToolInput(**input_data)
        
        # Implement your logic here
        result = f"Processed {input_data.param1} with {input_data.param2}"
        
        return MyToolOutput(result=result, success=True)
```

### 3. Register the Tool

Add the tool to the registry in `mini_devin/tools/__init__.py`:

```python
from mini_devin.tools.my_tool import MyTool

def create_my_tool(config: dict | None = None) -> MyTool:
    """Factory function for MyTool."""
    return MyTool(config)
```

### 4. Add Tests

Create tests in `tests/unit/tools/test_my_tool.py`:

```python
import pytest
from mini_devin.tools.my_tool import MyTool

class TestMyTool:
    def test_execute_success(self):
        tool = MyTool()
        result = tool.execute({"param1": "test", "param2": 5})
        assert result.success
        assert "test" in result.result
    
    def test_execute_with_defaults(self):
        tool = MyTool()
        result = tool.execute({"param1": "test"})
        assert result.success
```

## Adding New Skills

Skills are reusable procedures that combine multiple tool calls. Here's how to create a new skill:

### 1. Create the Skill

Create the skill in `mini_devin/skills/builtin/my_skill.py`:

```python
from mini_devin.skills.base import Skill, SkillParameter, SkillStep

class MySkill(Skill):
    """My custom skill description."""
    
    name = "my_skill"
    description = "Performs a complex operation"
    
    parameters = [
        SkillParameter(
            name="target",
            type="string",
            description="The target to operate on",
            required=True
        ),
        SkillParameter(
            name="options",
            type="object",
            description="Additional options",
            required=False
        )
    ]
    
    async def execute(self, context: dict) -> dict:
        """Execute the skill."""
        target = context["target"]
        options = context.get("options", {})
        
        # Step 1: Analyze
        self.log_step("Analyzing target")
        analysis = await self._analyze(target)
        
        # Step 2: Process
        self.log_step("Processing")
        result = await self._process(analysis, options)
        
        # Step 3: Verify
        self.log_step("Verifying result")
        verified = await self._verify(result)
        
        return {
            "success": verified,
            "result": result
        }
    
    async def _analyze(self, target: str) -> dict:
        # Use tools via self.tool_registry
        terminal = self.tool_registry.get("terminal")
        result = terminal.execute({"command": f"ls -la {target}"})
        return {"files": result.output}
    
    async def _process(self, analysis: dict, options: dict) -> str:
        # Processing logic
        return "processed"
    
    async def _verify(self, result: str) -> bool:
        # Verification logic
        return True
```

### 2. Register the Skill

Add to `mini_devin/skills/builtin/__init__.py`:

```python
from mini_devin.skills.builtin.my_skill import MySkill

BUILTIN_SKILLS = [
    # ... existing skills
    MySkill,
]
```

## Adding New LLM Providers

To add support for a new LLM provider:

### 1. Add Provider Enum

Update `mini_devin/core/providers.py`:

```python
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE = "azure"
    MY_PROVIDER = "my_provider"  # Add new provider
```

### 2. Add Model Information

```python
MY_PROVIDER_MODELS = [
    ModelInfo(
        id="my-model-v1",
        name="My Model v1",
        provider=Provider.MY_PROVIDER,
        context_window=32000,
        supports_vision=False,
        supports_function_calling=True,
    ),
]
```

### 3. Update LLM Client

Update `mini_devin/core/llm_client.py` to handle the new provider:

```python
def _setup_provider(self) -> None:
    if self.config.provider == Provider.MY_PROVIDER:
        self._setup_my_provider()

def _setup_my_provider(self) -> None:
    api_key = os.getenv("MY_PROVIDER_API_KEY")
    if not api_key:
        raise ValueError("MY_PROVIDER_API_KEY not set")
    # Configure LiteLLM or direct API client
```

## Testing

### Running Tests

```bash
# Run all unit tests
poetry run pytest tests/unit/ -v

# Run specific test file
poetry run pytest tests/unit/tools/test_terminal.py -v

# Run with coverage
poetry run pytest tests/unit/ --cov=mini_devin --cov-report=html

# Run E2E tests
poetry run pytest tests/e2e/ -v
```

### Writing Tests

Follow these conventions:

1. **Test file naming:** `test_<module>.py`
2. **Test class naming:** `Test<ClassName>`
3. **Test method naming:** `test_<behavior>`
4. **Use fixtures** for common setup
5. **Mock external dependencies** (API calls, file system)

Example test structure:

```python
import pytest
from unittest.mock import MagicMock, patch

class TestMyFeature:
    @pytest.fixture
    def my_fixture(self):
        """Set up test fixture."""
        return MyClass()
    
    def test_success_case(self, my_fixture):
        """Test successful execution."""
        result = my_fixture.do_something()
        assert result.success
    
    def test_failure_case(self, my_fixture):
        """Test failure handling."""
        with pytest.raises(ValueError):
            my_fixture.do_something_invalid()
    
    @patch("mini_devin.external.api_call")
    def test_with_mock(self, mock_api, my_fixture):
        """Test with mocked external dependency."""
        mock_api.return_value = {"status": "ok"}
        result = my_fixture.call_external()
        assert result.success
        mock_api.assert_called_once()
```

## Code Style

### Python

- Use **ruff** for linting: `poetry run ruff check mini_devin/`
- Use **mypy** for type checking: `poetry run mypy mini_devin/`
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Write docstrings for public classes and methods

### TypeScript (Frontend)

- Use **ESLint**: `npm run lint`
- Use **Prettier** for formatting
- Follow React best practices
- Use TypeScript strict mode

### Commit Messages

Follow conventional commits:

```
feat: add new browser tool for PDF extraction
fix: resolve race condition in session manager
docs: update API documentation
test: add unit tests for planner agent
refactor: simplify tool registry implementation
```

## Database Migrations

### Creating Migrations

```bash
# Auto-generate migration from model changes
poetry run alembic revision --autogenerate -m "Add new column to tasks"

# Create empty migration
poetry run alembic revision -m "Custom migration"
```

### Running Migrations

```bash
# Apply all pending migrations
poetry run alembic upgrade head

# Rollback one migration
poetry run alembic downgrade -1

# View migration history
poetry run alembic history
```

## Debugging

### Logging

Configure logging level via environment:

```bash
LOG_LEVEL=DEBUG poetry run mini-devin run "task"
```

### Debugging the Agent Loop

Add breakpoints or logging in `mini_devin/orchestrator/agent.py`:

```python
import logging
logger = logging.getLogger(__name__)

async def _execute_iteration(self):
    logger.debug(f"Iteration {self.iteration}: state={self.state}")
    # ... rest of method
```

### Debugging Tools

Use the `--dry-run` flag to see what the agent would do:

```bash
poetry run mini-devin run "task" --dry-run
```

### Viewing Artifacts

Check the `runs/<task_id>/` directory for:
- `plan.json` - Execution plan
- `tool_calls.json` - All tool invocations
- `verification_results.json` - Lint/test results
- `diff.patch` - Code changes

## Production Deployment

### VPS Deployment (DigitalOcean)

Mini-Devin is deployed on a DigitalOcean VPS with the following configuration:

**URLs:**
- Frontend: https://jomiye.com
- Backend API: https://api.jomiye.com
- API Docs: https://api.jomiye.com/docs

**Server Stack:**
- Ubuntu 22.04 LTS
- Caddy (reverse proxy with automatic SSL)
- Python 3.12 + Poetry
- Systemd service for backend

**Directory Structure on VPS:**
```
/root/mini-devin/          # Backend application
/var/www/jomiye.com/       # Frontend static files
/etc/caddy/Caddyfile       # Caddy configuration
```

**Caddy Configuration:**
```
api.jomiye.com {
    reverse_proxy localhost:8000
}

jomiye.com {
    root * /var/www/jomiye.com
    file_server
    try_files {path} /index.html
}

www.jomiye.com {
    redir https://jomiye.com{uri}
}
```

**Systemd Service:**
The backend runs as a systemd service (`mini-devin.service`) that starts automatically on boot.

### Deploying Updates

**Backend Updates:**
```bash
# SSH into VPS
ssh root@165.22.223.43

# Pull latest code
cd /root/mini-devin
git pull origin main

# Install dependencies
poetry install

# Restart service
systemctl restart mini-devin
```

**Frontend Updates:**
```bash
# Build frontend locally
cd frontend
npm run build

# Upload to VPS
scp -r dist/* root@165.22.223.43:/var/www/jomiye.com/

# Reload Caddy (if config changed)
ssh root@165.22.223.43 "systemctl reload caddy"
```

### DNS Configuration

Configure these DNS records in your domain registrar (GoDaddy):

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | @ | 165.22.223.43 | 1/2 Hour |
| A | api | 165.22.223.43 | 1/2 Hour |

### Environment Variables

Set these on the VPS in `/root/mini-devin/.env`:
- `OPENAI_API_KEY` - OpenAI API key for LLM
- `DATABASE_URL` - PostgreSQL connection string (optional)
- `SECRET_KEY` - JWT secret key

## Pull Request Guidelines

1. **Create a feature branch** from the main branch
2. **Write tests** for new functionality
3. **Run linting** before committing
4. **Update documentation** if needed
5. **Create a PR** with a clear description
6. **Wait for CI** to pass
7. **Address review feedback**

### PR Checklist

- [ ] Tests pass locally
- [ ] Linting passes
- [ ] Documentation updated
- [ ] No secrets committed
- [ ] Conventional commit messages used
