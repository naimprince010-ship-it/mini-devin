# Mini-Devin Sandbox Image
FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    nano \
    build-essential \
    ca-certificates \
    gnupg \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install common Python tools
RUN pip3 install --no-cache-dir \
    pytest \
    ruff \
    mypy \
    black \
    poetry

# Install common Node tools
RUN npm install -g \
    typescript \
    eslint \
    prettier

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set up git config
RUN git config --global user.email "mini-devin@example.com" \
    && git config --global user.name "Mini-Devin" \
    && git config --global init.defaultBranch main

# Default command
CMD ["bash"]
