# Mini-Devin Operations Guide

This document provides comprehensive instructions for running and maintaining Mini-Devin, an autonomous AI software engineer agent.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Run Modes](#run-modes)
3. [Configuration](#configuration)
4. [Docker Deployment](#docker-deployment)
5. [Safety Features](#safety-features)
6. [Monitoring and Artifacts](#monitoring-and-artifacts)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

---

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Docker and Docker Compose (for containerized deployment)
- An LLM API key (OpenAI or Anthropic)

### Installation

```bash
# Clone and enter the project
cd /path/to/mini-devin

# Install dependencies
poetry install

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

**Docker Mode:**
```bash
# Start with Docker Compose
docker-compose up -d

# Run a task
docker-compose exec mini-devin mini-devin run "Your task here"

# With interactive browser (Selenium)
docker-compose --profile interactive up -d
```

---

## Run Modes

Mini-Devin supports three run modes that control which tools are available:

### Offline Mode (Default)

```bash
RUN_MODE=offline
```

**Available Tools:**
- Terminal (shell commands)
- Editor (read, write, search, patch files)

**Use Cases:**
- Code refactoring
- Bug fixes in local code
- Test writing
- Documentation updates

**Network Access:** None

### Browse Mode

```bash
RUN_MODE=browse
```

**Available Tools:**
- Terminal
- Editor
- Browser Search (API-based web search)
- Browser Fetch (read-only page fetching)

**Use Cases:**
- Researching documentation
- Finding solutions to errors
- Implementing features that require external knowledge

**Network Access:** Read-only (search APIs, HTTP GET)

**Requirements:**
- `TAVILY_API_KEY` or `SERPAPI_API_KEY`

### Interactive Mode

```bash
RUN_MODE=interactive
```

**Available Tools:**
- Terminal
- Editor
- Browser Search
- Browser Fetch
- Browser Interactive (Selenium-based automation)

**Use Cases:**
- Testing web applications
- Interacting with web forms
- Scraping dynamic content
- End-to-end testing

**Network Access:** Full (including form submissions, JavaScript execution)

**Requirements:**
- `TAVILY_API_KEY` or `SERPAPI_API_KEY`
- Selenium Grid (provided via Docker Compose with `--profile interactive`)

---

## Configuration

### Environment Variables

All configuration is done through environment variables. Copy `.env.example` to `.env` and customize:

#### Required Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key (alternative) | `sk-ant-...` |

#### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o` | Model to use |
| `LLM_TEMPERATURE` | `0.1` | Response temperature |

#### Run Mode Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_MODE` | `offline` | Run mode: offline, browse, interactive |
| `TAVILY_API_KEY` | - | Tavily search API key |
| `SERPAPI_API_KEY` | - | SerpAPI key (alternative) |

#### Safety Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ITERATIONS` | `50` | Maximum agent iterations |
| `MAX_REPAIR_ITERATIONS` | `3` | Maximum repair loop iterations |
| `ALLOW_DEPENDENCY_BUMP` | `false` | Allow modifying dependency files |
| `MAX_LINES_EDIT` | `300` | Maximum lines per edit operation |
| `MAX_FILES_DELETE` | `1` | Maximum files per delete operation |

#### Artifact Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTIFACT_DIR` | `./runs` | Directory for run artifacts |
| `VERBOSE` | `true` | Enable verbose logging |
| `LOG_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |

---

## Docker Deployment

### Basic Deployment

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f mini-devin

# Stop
docker-compose down
```

### With Interactive Browser

```bash
# Start with Selenium Grid
docker-compose --profile interactive up -d

# Access Selenium VNC viewer (for debugging)
# Open http://localhost:7900 in your browser
# Password: secret
```

### Resource Limits

Configure in `.env` or `docker-compose.yml`:

```yaml
CPU_LIMIT=2.0      # CPU cores
MEMORY_LIMIT=4G    # Memory limit
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./workspace` | `/workspace` | Working directory |
| `./runs` | `/workspace/runs` | Run artifacts |

---

## Sandbox Security (Phase 6D)

Mini-Devin runs in a security-hardened Docker sandbox with the following protections:

### Non-Root User

By default, the container runs as a non-root user (`minidevin` with UID/GID 1000) to prevent privilege escalation attacks.

```bash
# Customize user ID to match host user (for file permissions)
USER_ID=1000 GROUP_ID=1000 docker-compose up -d
```

### Resource Limits

The sandbox enforces strict resource limits to prevent resource exhaustion:

| Setting | Default | Description |
|---------|---------|-------------|
| `CPU_LIMIT` | `2.0` | Maximum CPU cores |
| `MEMORY_LIMIT` | `4G` | Maximum memory |
| `PID_LIMIT` | `256` | Maximum processes |
| `TMP_SIZE` | `512MB` | Temporary filesystem size |
| `NOFILE_SOFT` | `65536` | Soft limit for open files |
| `NOFILE_HARD` | `65536` | Hard limit for open files |
| `NPROC_SOFT` | `256` | Soft limit for processes |
| `NPROC_HARD` | `512` | Hard limit for processes |

### Capability Restrictions

The container drops all Linux capabilities by default and only adds back the minimum required:

- `CHOWN` - Change file ownership
- `DAC_OVERRIDE` - Bypass file permission checks
- `FOWNER` - Bypass permission checks for file owner
- `SETGID` - Set group ID
- `SETUID` - Set user ID

Additional security options:
- `no-new-privileges` - Prevents privilege escalation via setuid binaries

### Read-Only Filesystem

Enable read-only root filesystem for maximum security (disabled by default for compatibility):

```bash
READ_ONLY_ROOT=true docker-compose up -d
```

When enabled, only `/workspace`, `/workspace/runs`, and `/tmp` are writable.

### Network Isolation

Enable network isolation to restrict outbound connections:

```bash
NETWORK_ISOLATION=true docker-compose up -d
```

### Security Configuration

All security settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_ENABLED` | `true` | Enable sandbox security features |
| `RUN_AS_NON_ROOT` | `true` | Run container as non-root user |
| `USER_ID` | `1000` | User ID for container user |
| `GROUP_ID` | `1000` | Group ID for container user |
| `READ_ONLY_ROOT` | `false` | Enable read-only root filesystem |
| `NETWORK_ISOLATION` | `false` | Enable network isolation |
| `DROP_ALL_CAPABILITIES` | `true` | Drop all Linux capabilities |
| `NO_NEW_PRIVILEGES` | `true` | Prevent privilege escalation |

---

## Safety Features

Mini-Devin includes multiple safety mechanisms to prevent dangerous operations.

### Hard Blocks

The following operations are **always blocked**:

1. **Bulk File Deletion:** Deleting more than 1 file in a single operation
2. **Large Edits:** Editing more than 300 lines in a single iteration
3. **Dependency Modifications:** Modifying package.json, requirements.txt, etc. (unless `ALLOW_DEPENDENCY_BUMP=true`)
4. **Dangerous Commands:**
   - `rm -rf /` and variants
   - Fork bombs
   - Disk formatting commands
   - Force push to git
   - Piping untrusted content to shell

### Stop Conditions

The agent will **STOP** execution when:

| Condition | Behavior |
|-----------|----------|
| Max iterations reached | Stops with `MAX_ITERATIONS_REACHED` |
| Max repair iterations | Stops with `MAX_REPAIR_ITERATIONS_REACHED` |
| Timeout exceeded | Stops with `TIMEOUT_EXCEEDED` |
| 5 consecutive errors | Stops with `UNRECOVERABLE_ERROR` |
| Dangerous command detected | Stops with `DANGEROUS_COMMAND_BLOCKED` |

### Blocked Conditions

The agent enters **BLOCKED** state (requires user intervention) when:

| Condition | Resolution |
|-----------|------------|
| Missing API key | Add the required API key to `.env` |
| Permission denied | Grant necessary permissions |
| Dependency bump required | Set `ALLOW_DEPENDENCY_BUMP=true` or manually update |
| Ambiguous task | Provide clarification |

### Viewing Safety Status

Check the run artifacts for safety-related information:

```bash
# View verification results
cat runs/<task_id>/verification_results.json

# View final summary
cat runs/<task_id>/final_summary.md
```

---

## Monitoring and Artifacts

### Run Artifacts

Each task execution creates a directory under `runs/<task_id>/` containing:

| File | Description |
|------|-------------|
| `plan.json` | Initial and updated task plans |
| `tool_calls.json` | All tool invocations with inputs/outputs |
| `verification_results.json` | Lint, test, and typecheck results |
| `diff.patch` | Git diff of all changes |
| `final_summary.md` | Human-readable execution summary |

### Viewing Artifacts

```bash
# List all runs
ls -la runs/

# View a specific run
ls -la runs/<task_id>/

# View tool calls
cat runs/<task_id>/tool_calls.json | jq .

# View the diff
cat runs/<task_id>/diff.patch
```

### Log Levels

Set `LOG_LEVEL` to control verbosity:

- `DEBUG`: All messages including internal state
- `INFO`: Normal operation messages
- `WARNING`: Potential issues
- `ERROR`: Errors only

---

## Troubleshooting

### Common Issues

#### "No LLM API key configured"

**Cause:** Missing `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

**Solution:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
# or add to .env file
```

#### "Run mode 'browse' requires a search API key"

**Cause:** Using browse/interactive mode without search API key

**Solution:**
```bash
export TAVILY_API_KEY="tvly-your-key-here"
# or use offline mode
export RUN_MODE=offline
```

#### "Maximum iterations reached"

**Cause:** Task is too complex or agent is stuck in a loop

**Solutions:**
1. Increase `MAX_ITERATIONS` (not recommended for production)
2. Break the task into smaller subtasks
3. Check `tool_calls.json` to understand where the agent got stuck

#### "Dangerous command blocked"

**Cause:** Agent attempted a potentially harmful command

**Solution:** This is expected behavior. Review the blocked command in the logs and determine if it's actually needed. If so, modify the task to use safer alternatives.

#### Docker container won't start

**Cause:** Various Docker issues

**Solutions:**
```bash
# Check logs
docker-compose logs mini-devin

# Rebuild
docker-compose build --no-cache

# Check resource limits
docker stats
```

#### Selenium not connecting

**Cause:** Selenium Grid not started or network issues

**Solutions:**
```bash
# Ensure interactive profile is used
docker-compose --profile interactive up -d

# Check Selenium status
curl http://localhost:4444/status

# View Selenium logs
docker-compose logs selenium
```

### Getting Help

1. Check the run artifacts for detailed error information
2. Review `tool_calls.json` to trace the execution path
3. Enable `DEBUG` logging for more details
4. Check the `final_summary.md` for the agent's perspective

---

## Maintenance

### Cleaning Up Artifacts

```bash
# Remove old runs (older than 7 days)
find runs/ -type d -mtime +7 -exec rm -rf {} +

# Remove all runs
rm -rf runs/*
```

### Updating Mini-Devin

```bash
# Pull latest changes
git pull

# Update dependencies
poetry install

# Rebuild Docker image
docker-compose build --no-cache
```

### Backup and Restore

**Backup:**
```bash
# Backup runs
tar -czvf mini-devin-runs-backup.tar.gz runs/

# Backup configuration
cp .env .env.backup
```

**Restore:**
```bash
# Restore runs
tar -xzvf mini-devin-runs-backup.tar.gz

# Restore configuration
cp .env.backup .env
```

### Health Checks

```bash
# Check if Mini-Devin is healthy
docker-compose exec mini-devin python -c "import mini_devin; print('OK')"

# Check Selenium (if using interactive mode)
curl http://localhost:4444/status
```

### Performance Tuning

For better performance:

1. **Increase resources:**
   ```yaml
   CPU_LIMIT=4.0
   MEMORY_LIMIT=8G
   ```

2. **Use faster models:**
   ```bash
   LLM_MODEL=gpt-4o-mini  # Faster but less capable
   ```

3. **Reduce safety limits (use with caution):**
   ```bash
   MAX_ITERATIONS=100
   MAX_LINES_EDIT=500
   ```

---

## End-to-End Testing

Mini-Devin includes a comprehensive end-to-end test suite that validates all components work together correctly.

### Running E2E Tests Locally

```bash
# Run all E2E tests
poetry run pytest tests/e2e/ -v

# Run with detailed output
poetry run pytest tests/e2e/ -v --tb=short

# Generate a test report
poetry run python tests/e2e/generate_report.py
```

The test report will be generated at `tests/e2e/e2e_test_report.md`.

### Running E2E Tests in CI

E2E tests can be triggered in CI in three ways:

1. **Manual trigger (workflow_dispatch):**
   - Go to Actions > CI > Run workflow
   - Check "Run E2E tests" checkbox
   - Click "Run workflow"

2. **Pull request label:**
   - Add the `run-e2e` label to your PR
   - E2E tests will run automatically

3. **Always run (E2E_REQUIRED=true):**
   - Set the repository variable `E2E_REQUIRED` to `true`
   - E2E tests will run on every push and PR

### E2E Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `E2E_REQUIRED` | `false` | When true, E2E tests run on every CI and block merge on failure |
| `E2E_TIMEOUT` | `300` | Timeout in seconds for E2E test execution |
| `E2E_REPORT_DIR` | `./tests/e2e` | Directory for E2E test reports |

### E2E Test Categories

The E2E test suite covers:

1. **Terminal & Editor Tools:** Tests for command execution, file operations, and integration between terminal and editor.

2. **Browser Tools:** Tests for browser tool registration and schemas (search, fetch, interactive).

3. **Gates Integration:** Tests for planner and reviewer gates in the execution flow, including configuration via environment variables.

### Viewing E2E Test Reports

When E2E tests run in CI, the test report is uploaded as an artifact:

1. Go to the workflow run in GitHub Actions
2. Scroll to "Artifacts" section
3. Download `e2e-test-report`
4. Open `e2e_test_report.md` to view results

---

## Appendix: CLI Reference

```
mini-devin - Autonomous AI Software Engineer Agent

Usage:
  mini-devin run <task> [options]     Run a task
  mini-devin interactive [options]    Start interactive mode
  mini-devin version                  Show version

Options:
  --dir PATH          Working directory (default: current directory)
  --model MODEL       LLM model to use (default: from env)
  --verbose           Enable verbose output
  --dry-run           Show plan without executing

Environment Variables:
  OPENAI_API_KEY      OpenAI API key
  ANTHROPIC_API_KEY   Anthropic API key
  RUN_MODE            Run mode: offline, browse, interactive
  MAX_ITERATIONS      Maximum iterations (default: 50)
  
See OPERATIONS.md for full documentation.
```
