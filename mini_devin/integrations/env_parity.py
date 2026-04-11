"""
Environment Parity

Ensures local sandbox and production environments are identical.
Provides:
  - Dockerfile auto-generation from a project's requirements/package.json
  - .env.example auto-generation from running environment or existing .env
  - docker-compose.yml generation for local dev parity
  - Environment diff (local vs cloud) report
"""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dockerfile generation
# ---------------------------------------------------------------------------

PYTHON_DOCKERFILE_TEMPLATE = """\
FROM python:{python_version}-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl build-essential \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (layer cache)
COPY {requirements_file} ./
RUN pip install --no-cache-dir -r {requirements_file}

# App source
COPY . .

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE {port}

{healthcheck}
CMD {cmd}
"""

NODE_DOCKERFILE_TEMPLATE = """\
FROM node:{node_version}-slim

WORKDIR /app

# Dependencies first (layer cache)
COPY package*.json ./
RUN npm ci --only=production

# App source
COPY . .

# Build (if needed)
{build_cmd}

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE {port}

{healthcheck}
CMD {cmd}
"""

FULLSTACK_DOCKERFILE_TEMPLATE = """\
# ── Stage 1: Build frontend ────────────────────────────────────────────────
FROM node:{node_version}-slim AS frontend-builder
WORKDIR /frontend
COPY {frontend_dir}/package*.json ./
RUN npm ci
COPY {frontend_dir}/ ./
RUN npm run build

# ── Stage 2: Python backend ────────────────────────────────────────────────
FROM python:{python_version}-slim
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl build-essential \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY {requirements_file} ./
RUN pip install --no-cache-dir -r {requirements_file}

COPY . .
COPY --from=frontend-builder /frontend/dist ./{frontend_dir}/dist

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE {port}
{healthcheck}
CMD {cmd}
"""


def _detect_python_version(project_root: Path) -> str:
    """Read .python-version, pyproject.toml, or fall back to 3.11."""
    pv = project_root / ".python-version"
    if pv.exists():
        return pv.read_text().strip()
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        m = re.search(r'python_requires\s*=\s*"[>=!<~^]*(\d+\.\d+)', text)
        if m:
            return m.group(1)
    return "3.11"


def _detect_node_version(project_root: Path) -> str:
    """Read .nvmrc, .node-version, or package.json engines."""
    for f in [".nvmrc", ".node-version"]:
        p = project_root / f
        if p.exists():
            return p.read_text().strip().lstrip("v")
    pkg = project_root / "package.json"
    if pkg.exists():
        try:
            engines = json.loads(pkg.read_text()).get("engines", {})
            raw = engines.get("node", "")
            m = re.search(r"(\d+)", raw)
            if m:
                return m.group(1)
        except Exception:
            pass
    return "20"


def _detect_port(project_root: Path) -> int:
    """Try to infer the application port from common config files."""
    # Check .env / .env.example
    for envfile in [".env", ".env.example", ".env.production"]:
        p = project_root / envfile
        if p.exists():
            m = re.search(r"^PORT\s*=\s*(\d+)", p.read_text(), re.MULTILINE)
            if m:
                return int(m.group(1))
    return 8000


def generate_dockerfile(
    project_root: str,
    *,
    project_type: str = "auto",   # python | node | fullstack | auto
    frontend_dir: str = "frontend",
    requirements_file: str = "requirements.txt",
    port: Optional[int] = None,
    cmd: Optional[List[str]] = None,
    health_path: str = "/health",
    output_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Generate a production-ready Dockerfile.

    Returns (dockerfile_content, output_path_used).
    """
    root = Path(project_root)

    # Auto-detect project type
    if project_type == "auto":
        has_py = (root / requirements_file).exists() or (root / "pyproject.toml").exists()
        has_node = (root / "package.json").exists()
        has_frontend = (root / frontend_dir / "package.json").exists()
        if has_py and (has_node or has_frontend):
            project_type = "fullstack"
        elif has_py:
            project_type = "python"
        elif has_node:
            project_type = "node"
        else:
            project_type = "python"   # default

    py_ver = _detect_python_version(root)
    node_ver = _detect_node_version(root)
    app_port = port or _detect_port(root)
    healthcheck_line = (
        f'HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\\n'
        f'    CMD curl -f http://localhost:{app_port}{health_path} || exit 1'
    )

    if project_type == "python":
        default_cmd = ["uvicorn", "mini_devin.api.app:app", "--host", "0.0.0.0", f"--port={app_port}"]
        content = PYTHON_DOCKERFILE_TEMPLATE.format(
            python_version=py_ver,
            requirements_file=requirements_file,
            port=app_port,
            healthcheck=healthcheck_line,
            cmd=json.dumps(cmd or default_cmd),
        )
    elif project_type == "node":
        pkg_path = root / "package.json"
        start_script = "node server.js"
        if pkg_path.exists():
            try:
                scripts = json.loads(pkg_path.read_text()).get("scripts", {})
                start_script = scripts.get("start", "node server.js")
            except Exception:
                pass
        content = NODE_DOCKERFILE_TEMPLATE.format(
            node_version=node_ver,
            port=app_port,
            build_cmd="RUN npm run build 2>/dev/null || true",
            healthcheck=healthcheck_line,
            cmd=json.dumps(cmd or ["sh", "-c", start_script]),
        )
    else:  # fullstack
        default_cmd = ["uvicorn", "mini_devin.api.app:app", "--host", "0.0.0.0", f"--port={app_port}"]
        content = FULLSTACK_DOCKERFILE_TEMPLATE.format(
            node_version=node_ver,
            python_version=py_ver,
            frontend_dir=frontend_dir,
            requirements_file=requirements_file,
            port=app_port,
            healthcheck=healthcheck_line,
            cmd=json.dumps(cmd or default_cmd),
        )

    out = output_path or str(root / "Dockerfile.generated")
    Path(out).write_text(content, encoding="utf-8")
    return content, out


# ---------------------------------------------------------------------------
# .env.example generation
# ---------------------------------------------------------------------------

def generate_env_example(
    project_root: str,
    *,
    source_env_file: str = ".env",
    output_path: Optional[str] = None,
    include_current_values: bool = False,
) -> Tuple[str, str]:
    """
    Parse .env and generate .env.example with placeholder values.
    Secrets are always blanked; non-secret values can be kept optionally.

    Returns (content, output_path_used).
    """
    root = Path(project_root)
    src = root / source_env_file

    SECRET_PATTERNS = re.compile(
        r"(key|secret|password|passwd|token|credential|private|auth|api_key|access|jwt)",
        re.IGNORECASE,
    )

    lines: List[str] = []
    if src.exists():
        for raw in src.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                lines.append(raw)
                continue
            if "=" not in line:
                lines.append(raw)
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if SECRET_PATTERNS.search(key):
                lines.append(f"{key}=")
            elif include_current_values:
                lines.append(raw)
            else:
                lines.append(f"{key}=your_{key.lower()}_here")
    else:
        # Scaffold from environment variables that look config-like
        for k in sorted(os.environ.keys()):
            if SECRET_PATTERNS.search(k):
                lines.append(f"{k}=")
            else:
                if include_current_values:
                    lines.append(f"{k}={os.environ[k]}")
                else:
                    lines.append(f"{k}=")

    content = "\n".join(lines) + "\n"
    out = output_path or str(root / ".env.example")
    Path(out).write_text(content, encoding="utf-8")
    return content, out


# ---------------------------------------------------------------------------
# docker-compose.yml generation
# ---------------------------------------------------------------------------

COMPOSE_TEMPLATE = """\
version: "3.9"

services:
  app:
    build: .
    ports:
      - "{port}:{port}"
    env_file:
      - .env
    volumes:
      - .:/app
    restart: unless-stopped
{extra_services}
"""

REDIS_SERVICE = """\
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
"""

POSTGRES_SERVICE = """\
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: appdb
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD: apppassword
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  pgdata:
"""


def generate_docker_compose(
    project_root: str,
    *,
    port: Optional[int] = None,
    include_redis: bool = False,
    include_postgres: bool = False,
    output_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Generate a docker-compose.yml for local development parity.

    Returns (content, output_path_used).
    """
    root = Path(project_root)
    app_port = port or _detect_port(root)

    # Auto-detect services from requirements / env
    extra = ""
    req_path = root / "requirements.txt"
    if req_path.exists():
        req_text = req_path.read_text()
        if "redis" in req_text.lower() or include_redis:
            extra += REDIS_SERVICE
        if any(x in req_text.lower() for x in ("psycopg", "asyncpg", "pg8000")) or include_postgres:
            extra += POSTGRES_SERVICE

    content = COMPOSE_TEMPLATE.format(port=app_port, extra_services=extra)
    out = output_path or str(root / "docker-compose.yml")
    Path(out).write_text(content, encoding="utf-8")
    return content, out


# ---------------------------------------------------------------------------
# Environment diff (local vs production)
# ---------------------------------------------------------------------------

def diff_environments(
    local_env_file: str = ".env",
    production_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Compare local .env with production env vars.

    production_env: dict of {KEY: value} from cloud provider, or None to use os.environ.
    Returns a dict with: only_local, only_production, value_mismatch, identical.
    """
    local: Dict[str, str] = {}
    env_path = Path(local_env_file)
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                local[k.strip()] = v.strip()

    prod = production_env if production_env is not None else dict(os.environ)

    SECRET_PATTERNS = re.compile(
        r"(key|secret|password|token|credential|private|auth|jwt)", re.IGNORECASE
    )

    only_local = sorted(set(local) - set(prod))
    only_production = sorted(set(prod) - set(local))
    mismatch = []
    identical = []

    for k in sorted(set(local) & set(prod)):
        lv = local[k]
        pv = prod[k]
        if lv != pv:
            if SECRET_PATTERNS.search(k):
                mismatch.append({"key": k, "local": "***", "production": "***"})
            else:
                mismatch.append({"key": k, "local": lv[:80], "production": pv[:80]})
        else:
            identical.append(k)

    return {
        "only_local": only_local,
        "only_production": only_production,
        "value_mismatch": mismatch,
        "identical_count": len(identical),
        "summary": (
            f"{len(only_local)} vars only in local, "
            f"{len(only_production)} only in production, "
            f"{len(mismatch)} value mismatches"
        ),
    }
