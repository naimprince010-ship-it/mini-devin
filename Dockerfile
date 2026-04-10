FROM python:3.11-slim

WORKDIR /app

# Install system dependencies — git is required by GitPython
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry --no-cache-dir

# Copy everything
COPY . .

# Install Python dependencies (skip playwright browser download — not needed for API)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Expose port
EXPOSE 8000

# Environment hardening
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONIOENCODING=UTF-8
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# Run the bootstrap watchdog
CMD ["python", "scripts/bootstrap.py"]
