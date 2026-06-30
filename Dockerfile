FROM python:3.11-slim

WORKDIR /app

# Skip auto-download during poetry install; we install Chromium explicitly below.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_HTTP_TIMEOUT=300
ENV POETRY_NO_INTERACTION=1

# Install system dependencies — git, curl, Node.js + Chromium system libs
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    # Chromium / Playwright runtime dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Pin Poetry 1.x (stable with existing lockfiles). Also keep common agent
# verification tools globally available for cloned workspaces that do not have
# their own virtualenv yet.
RUN pip install \
    "poetry>=1.8.5,<2.0" \
    "pytest>=7.4,<9" \
    "pytest-asyncio>=0.21,<1" \
    "ruff>=0.1,<1" \
    --no-cache-dir

# Copy backend files and install Python deps first (layer cache)
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Install Playwright Chromium browser binary (needed for browser_playwright tool)
# Store binaries in /ms-playwright so all UIDs can access them (not /root/.cache)
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python -m playwright install chromium \
    && chmod -R 755 /ms-playwright

# Copy frontend and build it
COPY frontend/ ./frontend/
WORKDIR /app/frontend
# Large Vite bundle: avoid OOM on small builders
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi \
    && NODE_OPTIONS=--max-old-space-size=4096 npm run build

# Copy rest of the app (includes ``scripts/container_entrypoint.py``)
WORKDIR /app
COPY . .

# Do not hardcode PORT here; the hosting platform may inject it at runtime.
EXPOSE 8080

# Environment hardening
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONIOENCODING=UTF-8

# Python entrypoint avoids CRLF/shebang issues with ``.sh`` from Windows checkouts.
# Local watchdog: ``python scripts/bootstrap.py``
CMD ["python", "/app/scripts/container_entrypoint.py"]
