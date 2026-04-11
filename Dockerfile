FROM python:3.11-slim

WORKDIR /app

# Must be set before `poetry install`: playwright is a dependency and otherwise
# tries to download browser binaries during install (fails / OOM on Railway).
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_HTTP_TIMEOUT=300
ENV POETRY_NO_INTERACTION=1

# Install system dependencies — git, curl, Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Pin Poetry 1.x (stable with existing lockfiles)
RUN pip install "poetry>=1.8.5,<2.0" --no-cache-dir

# Copy backend files and install Python deps first (layer cache)
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Copy frontend and build it
COPY frontend/ ./frontend/
WORKDIR /app/frontend
# Large Vite bundle: avoid OOM on small Railway builders
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi \
    && NODE_OPTIONS=--max-old-space-size=4096 npm run build

# Copy rest of the app
WORKDIR /app
COPY . .

# Do not set PORT here — Railway injects PORT at runtime (often 8080). Local runs
# use scripts/bootstrap.py fallback when PORT is unset. EXPOSE documents typical PaaS port.
EXPOSE 8080

# Environment hardening
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONIOENCODING=UTF-8

# Run the bootstrap watchdog
CMD ["python", "scripts/bootstrap.py"]
