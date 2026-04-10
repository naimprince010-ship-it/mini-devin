FROM python:3.11-slim

WORKDIR /app

# Install system dependencies — git, curl, Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry --no-cache-dir

# Copy backend files and install Python deps first (layer cache)
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Copy frontend and build it
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install && npm run build

# Copy rest of the app
WORKDIR /app
COPY . .

# Expose port
EXPOSE 8000

# Environment hardening
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONIOENCODING=UTF-8
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# Run the bootstrap watchdog
CMD ["python", "scripts/bootstrap.py"]
