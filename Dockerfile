FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and README (required by poetry)
COPY pyproject.toml poetry.lock README.md ./

# Install poetry and dependencies
RUN pip install poetry --no-cache-dir && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Copy application code
COPY app/ ./app/
COPY mini_devin/ ./mini_devin/

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
