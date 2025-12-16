# Mini-Devin Sandbox Image
# Security-hardened container for autonomous code execution
FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Security: Define non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=minidevin

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
    sudo \
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

# Security: Create non-root user with specific UID/GID
RUN groupadd --gid ${GROUP_ID} ${USERNAME} \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: /usr/bin/apt-get,/usr/bin/apt" >> /etc/sudoers.d/${USERNAME}

# Create workspace directory with proper permissions
RUN mkdir -p /workspace /workspace/runs /tmp/minidevin \
    && chown -R ${USERNAME}:${USERNAME} /workspace /tmp/minidevin

# Security: Create read-only directories for system configs
RUN mkdir -p /etc/minidevin \
    && chmod 755 /etc/minidevin

# Set up git config for the non-root user
RUN su - ${USERNAME} -c "git config --global user.email 'mini-devin@example.com'" \
    && su - ${USERNAME} -c "git config --global user.name 'Mini-Devin'" \
    && su - ${USERNAME} -c "git config --global init.defaultBranch main" \
    && su - ${USERNAME} -c "git config --global --add safe.directory /workspace"

# Security: Set restrictive umask
RUN echo "umask 027" >> /home/${USERNAME}/.bashrc

# Security: Remove unnecessary setuid/setgid binaries
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# Set working directory
WORKDIR /workspace

# Security: Switch to non-root user
USER ${USERNAME}

# Environment variables for the non-root user
ENV HOME=/home/${USERNAME}
ENV PATH="${HOME}/.local/bin:${PATH}"

# Default command
CMD ["bash"]
