FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g vercel railway \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry --no-cache-dir

# Copy ALL source files first (poetry needs mini_devin package and README.md)
COPY . .

# Install dependencies (no-root because Docker handles packaging)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root && \
    python -m playwright install chromium --with-deps

# Expose port
EXPOSE 8000

# Run the bootstrap script as a watchdog
CMD ["python", "scripts/bootstrap.py"]
