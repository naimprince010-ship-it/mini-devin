# Security notes for operators

## Secrets

- **Never commit** `.env` or real API keys. Use `.env.example` as a template only.
- Rotate keys if they appear in logs, screenshots, or shared session exports.
- CI should use repository **Secrets** for live LLM keys (e.g. optional E2E), not hard-coded values.

## Network and data

- Self-hosted deployments should run behind HTTPS and restrict who can reach the API/WebSocket.
- The agent can run shell commands and edit files in the configured workspace: treat **workspace path** and **run mode** (`RUN_MODE`) as part of your threat model.

## Reporting

- Open a **private** security advisory on GitHub (or contact maintainers) for sensitive issues instead of public issues when disclosure could increase risk.
