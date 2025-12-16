"""
Docker Sandbox for Mini-Devin

This module implements Docker-based sandboxing for safe code execution:
- Isolated container environment
- Repo mounted as /workspace
- No host filesystem access
- Resource limits (CPU, memory)
"""

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SandboxStatus(str, Enum):
    """Status of the sandbox."""
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SandboxConfig:
    """Configuration for the Docker sandbox."""
    # Image settings
    image: str = "mini-devin-sandbox:latest"
    
    # Resource limits
    memory_limit: str = "2g"
    cpu_limit: float = 2.0
    
    # Timeout
    timeout_seconds: int = 3600  # 1 hour default
    
    # Network
    network_enabled: bool = True
    network_mode: str = "bridge"
    
    # Workspace
    workspace_path: str = "/workspace"
    
    # Environment variables
    env_vars: dict[str, str] = field(default_factory=dict)
    
    # Additional mounts (read-only)
    additional_mounts: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SandboxResult:
    """Result of a sandbox command execution."""
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool = False


class DockerSandbox:
    """
    Docker-based sandbox for safe code execution.
    
    Provides an isolated environment where the agent can:
    - Execute shell commands
    - Modify files in /workspace
    - Run tests and builds
    
    Without access to the host filesystem or network (optionally).
    """
    
    def __init__(
        self,
        repo_path: str,
        config: SandboxConfig | None = None,
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.config = config or SandboxConfig()
        self.container_id: str | None = None
        self.status = SandboxStatus.CREATED
        self.sandbox_id = str(uuid.uuid4())[:8]
    
    async def start(self) -> bool:
        """Start the sandbox container."""
        if self.status == SandboxStatus.RUNNING:
            return True
        
        try:
            # Build docker run command
            cmd_parts = [
                "docker", "run", "-d",
                "--name", f"mini-devin-{self.sandbox_id}",
                "-v", f"{self.repo_path}:{self.config.workspace_path}",
                "-w", self.config.workspace_path,
                "--memory", self.config.memory_limit,
                "--cpus", str(self.config.cpu_limit),
            ]
            
            # Network settings
            if not self.config.network_enabled:
                cmd_parts.extend(["--network", "none"])
            else:
                cmd_parts.extend(["--network", self.config.network_mode])
            
            # Environment variables
            for key, value in self.config.env_vars.items():
                cmd_parts.extend(["-e", f"{key}={value}"])
            
            # Additional mounts (read-only)
            for host_path, container_path in self.config.additional_mounts:
                cmd_parts.extend(["-v", f"{host_path}:{container_path}:ro"])
            
            # Image and command (keep container running)
            cmd_parts.extend([self.config.image, "tail", "-f", "/dev/null"])
            
            # Run docker command
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.container_id = stdout.decode().strip()
                self.status = SandboxStatus.RUNNING
                return True
            else:
                self.status = SandboxStatus.ERROR
                print(f"Failed to start sandbox: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.status = SandboxStatus.ERROR
            print(f"Error starting sandbox: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop and remove the sandbox container."""
        if not self.container_id:
            return True
        
        try:
            # Stop container
            process = await asyncio.create_subprocess_exec(
                "docker", "stop", self.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            
            # Remove container
            process = await asyncio.create_subprocess_exec(
                "docker", "rm", self.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            
            self.container_id = None
            self.status = SandboxStatus.STOPPED
            return True
            
        except Exception as e:
            print(f"Error stopping sandbox: {e}")
            return False
    
    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> SandboxResult:
        """
        Execute a command in the sandbox.
        
        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds (default: config timeout)
            working_dir: Working directory inside container (default: /workspace)
            
        Returns:
            SandboxResult with stdout, stderr, exit_code
        """
        if self.status != SandboxStatus.RUNNING:
            return SandboxResult(
                stdout="",
                stderr="Sandbox is not running",
                exit_code=-1,
                duration_ms=0,
            )
        
        timeout = timeout or self.config.timeout_seconds
        work_dir = working_dir or self.config.workspace_path
        
        start_time = datetime.utcnow()
        
        try:
            # Build docker exec command
            cmd_parts = [
                "docker", "exec",
                "-w", work_dir,
                self.container_id,
                "sh", "-c", command,
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                timed_out = False
            except asyncio.TimeoutError:
                process.kill()
                stdout, stderr = b"", b"Command timed out"
                timed_out = True
            
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return SandboxResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode or 0,
                duration_ms=duration_ms,
                timed_out=timed_out,
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return SandboxResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_ms=duration_ms,
            )
    
    async def read_file(self, path: str) -> str | None:
        """Read a file from the sandbox."""
        result = await self.execute(f"cat {path}")
        if result.exit_code == 0:
            return result.stdout
        return None
    
    async def write_file(self, path: str, content: str) -> bool:
        """Write a file in the sandbox."""
        result = await self.execute(f"cat > {path} << 'MINI_DEVIN_EOF'\n{content}\nMINI_DEVIN_EOF")
        return result.exit_code == 0
    
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in the sandbox."""
        result = await self.execute(f"test -f {path}")
        return result.exit_code == 0
    
    async def list_directory(self, path: str = ".") -> list[str]:
        """List directory contents in the sandbox."""
        result = await self.execute(f"ls -la {path}")
        if result.exit_code == 0:
            return result.stdout.strip().split("\n")
        return []
    
    async def get_diff(self) -> str:
        """Get git diff of changes in the sandbox."""
        result = await self.execute("git diff")
        return result.stdout if result.exit_code == 0 else ""
    
    async def commit_changes(self, message: str) -> bool:
        """Commit changes in the sandbox."""
        await self.execute("git add -A")
        result = await self.execute(f'git commit -m "{message}"')
        return result.exit_code == 0
    
    def is_running(self) -> bool:
        """Check if the sandbox is running."""
        return self.status == SandboxStatus.RUNNING
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Dockerfile content for the sandbox image
DOCKERFILE_CONTENT = '''# Mini-Devin Sandbox Image
FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    wget \\
    vim \\
    nano \\
    build-essential \\
    ca-certificates \\
    gnupg \\
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-venv \\
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \\
    && apt-get install -y nodejs \\
    && rm -rf /var/lib/apt/lists/*

# Install common Python tools
RUN pip3 install --no-cache-dir \\
    pytest \\
    ruff \\
    mypy \\
    black \\
    poetry

# Install common Node tools
RUN npm install -g \\
    typescript \\
    eslint \\
    prettier

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set up git config
RUN git config --global user.email "mini-devin@example.com" \\
    && git config --global user.name "Mini-Devin" \\
    && git config --global init.defaultBranch main

# Default command
CMD ["bash"]
'''


def generate_dockerfile(output_path: str = "Dockerfile") -> str:
    """Generate the Dockerfile for the sandbox image."""
    with open(output_path, "w") as f:
        f.write(DOCKERFILE_CONTENT)
    return output_path


async def build_sandbox_image(
    dockerfile_path: str = "Dockerfile",
    image_name: str = "mini-devin-sandbox:latest",
) -> bool:
    """Build the sandbox Docker image."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", image_name, "-f", dockerfile_path, ".",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print(f"Successfully built image: {image_name}")
            return True
        else:
            print(f"Failed to build image: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"Error building image: {e}")
        return False


def create_sandbox(
    repo_path: str,
    memory_limit: str = "2g",
    cpu_limit: float = 2.0,
    network_enabled: bool = True,
) -> DockerSandbox:
    """Create a Docker sandbox with custom settings."""
    config = SandboxConfig(
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        network_enabled=network_enabled,
    )
    return DockerSandbox(repo_path, config)
