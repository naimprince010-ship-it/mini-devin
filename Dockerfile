FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry --no-cache-dir

# Copy ALL source files first (poetry needs mini_devin package and README.md)
COPY . .

# Install dependencies (no-root because Docker handles packaging)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "mini_devin.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
