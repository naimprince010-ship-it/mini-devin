# Plodder platform surface

Plodder is structured like a **product family** (similar in spirit to OpenHands): one agent core, multiple ways to run it, and optional enterprise hooks.

## 1. SDK (Python library)

Import and drive the agent from your own code:

```python
from mini_devin.sdk import PlodderClient

async def main():
    client = PlodderClient(working_directory="/path/to/repo", model="gpt-4o")
    print(await client.run_task("Run tests and fix failures"))
```

Lower-level access:

```python
from mini_devin.sdk import create_agent, Agent

agent = await create_agent(working_directory=".")
await agent.run_simple("Your task")
```

## 2. CLI

| Command | Purpose |
|--------|---------|
| `plodder run "…"` | One-shot task in a directory |
| `plodder interactive` | REPL-style tasks |
| `plodder serve` | Start **FastAPI** (`mini_devin.api.app:app`) for the web UI |
| `plodder version` | Print version |

Install: `poetry install` (entry points `plodder` / `mini-devin` from `pyproject.toml`).

## 3. Local GUI

1. `plodder serve --host 0.0.0.0 --port 8000` (or `--reload` in development).
2. From `frontend/`: `npm install` then `npm run dev` (see repo `README.md`).

This is the **self-hosted** “local OpenHands GUI” equivalent.

## 4. Cloud (self-hosted)

Use `docker-compose.yml` and environment variables (`.env`) to deploy API + DB + optional services on your own infra. There is no separate Plodder SaaS product in this repo; **cloud** means *your* cloud.

## 5. Enterprise (RBAC & integrations)

- **RBAC**: `mini_devin.enterprise.rbac` defines `Role`, `Permission`, and `role_allows()`. FastAPI dependency `RequirePermission` in `mini_devin.auth.enterprise_deps` enforces checks (e.g. API key routes require `MANAGE_API_KEYS`).
- **Slack Events (stub)**: `POST /api/integrations/slack/events` handles Slack `url_verification` challenges. Set `SLACK_SIGNING_SECRET` to enforce signature verification (`X-Slack-Signature`). `GET /api/integrations/me/permissions` returns the caller’s derived role and permissions (Bearer / `X-API-Key`).
- **Slack notify (stub)**: `POST /api/integrations/slack/notify` — requires `MANAGE_INTEGRATIONS` (admins only with the default matrix); implement outbound webhook posting there.
- **Jira / Linear**: extend `mini_devin/api/integration_routes.py` similarly.

## Comparison at a glance

| Surface | Status in this repo |
|--------|----------------------|
| SDK | `mini_devin.sdk` |
| CLI | `mini_devin.cli` |
| Local GUI | React `frontend/` + `plodder serve` |
| Hosted SaaS | Bring your own hosting |
| Enterprise RBAC | Starter enums + `role_allows` |
| Slack/Jira/Linear | Documented extension point |
