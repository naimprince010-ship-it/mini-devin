FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including git (required by GitPython)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry --no-cache-dir

# Copy project files
COPY pyproject.toml poetry.lock* ./
COPY mini_devin/ ./mini_devin/
COPY scripts/ ./scripts/
COPY README.md ./

# Install Python dependencies (skip playwright/docker for production to save memory)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Expose port
EXPOSE 8000

# Silence GitPython warning if git binary path detection fails
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONIOENCODING=UTF-8

# Run the bootstrap watchdog
CMD ["python", "scripts/bootstrap.py"]
