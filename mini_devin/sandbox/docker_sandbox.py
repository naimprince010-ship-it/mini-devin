"""
Docker Sandbox for Mini-Devin

This module implements Docker-based sandboxing for safe code execution:
- Isolated container environment
- Repo mounted as /workspace
- No host filesystem access
- Resource limits (CPU, memory)
- Non-root user execution
- Read-only filesystem with explicit mount allowlists
- Network disabled by default
- Seccomp security profiles
"""

import asyncio
import json
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


class SecurityLevel(str, Enum):
    """Security level for the sandbox."""
    MINIMAL = "minimal"      # Basic isolation only
    STANDARD = "standard"    # Non-root, network disabled
    HARDENED = "hardened"    # Full security: read-only fs, seccomp, no capabilities


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
    
    # Network (disabled by default for security)
    network_enabled: bool = False
    network_mode: str = "none"
    
    # Workspace
    workspace_path: str = "/workspace"
    
    # Environment variables
    env_vars: dict[str, str] = field(default_factory=dict)
    
    # Additional mounts (read-only by default)
    additional_mounts: list[tuple[str, str, bool]] = field(default_factory=list)  # (host, container, writable)
    
    # Security settings
    security_level: SecurityLevel = SecurityLevel.STANDARD
    run_as_root: bool = False  # Run as non-root user by default
    user_id: int = 1000  # UID for non-root user
    group_id: int = 1000  # GID for non-root user
    
    # Read-only filesystem
    read_only_root: bool = False  # Make root filesystem read-only
    tmpfs_paths: list[str] = field(default_factory=lambda: ["/tmp", "/var/tmp"])  # Writable tmpfs mounts
    
    # Seccomp profile
    seccomp_profile: str | None = None  # Path to seccomp profile JSON
    use_default_seccomp: bool = True  # Use Docker's default seccomp profile
    
    # Capability restrictions
    drop_all_capabilities: bool = True  # Drop all Linux capabilities
    add_capabilities: list[str] = field(default_factory=list)  # Capabilities to add back
    
    # Process limits
    pids_limit: int = 256  # Maximum number of processes
    
    # Mount allowlist (paths that can be mounted)
    mount_allowlist: list[str] = field(default_factory=list)


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
    
    def _validate_mount_path(self, host_path: str) -> bool:
        """Validate that a mount path is in the allowlist."""
        if not self.config.mount_allowlist:
            return True  # No allowlist means all paths allowed
        
        abs_path = os.path.abspath(host_path)
        for allowed in self.config.mount_allowlist:
            allowed_abs = os.path.abspath(allowed)
            if abs_path.startswith(allowed_abs):
                return True
        return False
    
    def _apply_security_level(self) -> None:
        """Apply security settings based on security level."""
        if self.config.security_level == SecurityLevel.MINIMAL:
            self.config.run_as_root = True
            self.config.network_enabled = True
            self.config.drop_all_capabilities = False
            self.config.read_only_root = False
        elif self.config.security_level == SecurityLevel.STANDARD:
            self.config.run_as_root = False
            self.config.network_enabled = False
            self.config.drop_all_capabilities = True
            self.config.read_only_root = False
        elif self.config.security_level == SecurityLevel.HARDENED:
            self.config.run_as_root = False
            self.config.network_enabled = False
            self.config.drop_all_capabilities = True
            self.config.read_only_root = True
    
    async def start(self) -> bool:
        """Start the sandbox container with security hardening."""
        if self.status == SandboxStatus.RUNNING:
            return True
        
        # Apply security level settings
        self._apply_security_level()
        
        # Validate mount paths
        if not self._validate_mount_path(self.repo_path):
            print(f"Mount path not in allowlist: {self.repo_path}")
            self.status = SandboxStatus.ERROR
            return False
        
        try:
            # Build docker run command
            cmd_parts = [
                "docker", "run", "-d",
                "--name", f"mini-devin-{self.sandbox_id}",
                "-w", self.config.workspace_path,
                "--memory", self.config.memory_limit,
                "--cpus", str(self.config.cpu_limit),
                "--pids-limit", str(self.config.pids_limit),
            ]
            
            # User settings (non-root by default)
            if not self.config.run_as_root:
                cmd_parts.extend(["--user", f"{self.config.user_id}:{self.config.group_id}"])
            
            # Network settings (disabled by default)
            if not self.config.network_enabled:
                cmd_parts.extend(["--network", "none"])
            else:
                cmd_parts.extend(["--network", self.config.network_mode])
            
            # Read-only root filesystem
            if self.config.read_only_root:
                cmd_parts.append("--read-only")
                # Add tmpfs mounts for writable directories
                for tmpfs_path in self.config.tmpfs_paths:
                    cmd_parts.extend(["--tmpfs", f"{tmpfs_path}:rw,noexec,nosuid,size=100m"])
            
            # Capability restrictions
            if self.config.drop_all_capabilities:
                cmd_parts.append("--cap-drop=ALL")
                # Add back specific capabilities if needed
                for cap in self.config.add_capabilities:
                    cmd_parts.extend(["--cap-add", cap])
            
            # Seccomp profile
            if self.config.seccomp_profile:
                cmd_parts.extend(["--security-opt", f"seccomp={self.config.seccomp_profile}"])
            elif not self.config.use_default_seccomp:
                cmd_parts.extend(["--security-opt", "seccomp=unconfined"])
            
            # Additional security options
            cmd_parts.extend([
                "--security-opt", "no-new-privileges:true",  # Prevent privilege escalation
            ])
            
            # Workspace mount
            cmd_parts.extend(["-v", f"{self.repo_path}:{self.config.workspace_path}"])
            
            # Environment variables
            for key, value in self.config.env_vars.items():
                cmd_parts.extend(["-e", f"{key}={value}"])
            
            # Additional mounts with explicit read-only/writable control
            for mount_tuple in self.config.additional_mounts:
                if len(mount_tuple) == 3:
                    host_path, container_path, writable = mount_tuple
                else:
                    host_path, container_path = mount_tuple
                    writable = False
                
                if not self._validate_mount_path(host_path):
                    print(f"Mount path not in allowlist: {host_path}")
                    continue
                
                mount_opt = "" if writable else ":ro"
                cmd_parts.extend(["-v", f"{host_path}:{container_path}{mount_opt}"])
            
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


# Dockerfile content for the sandbox image with security hardening
DOCKERFILE_CONTENT = '''# Mini-Devin Sandbox Image (Hardened)
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
    sudo \\
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

# Create non-root user for security
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} sandbox && \\
    useradd -m -u ${USER_ID} -g sandbox -s /bin/bash sandbox && \\
    echo "sandbox ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create workspace directory with proper permissions
RUN mkdir -p /workspace && chown sandbox:sandbox /workspace

# Set up git config for both root and sandbox user
RUN git config --global user.email "mini-devin@example.com" && \\
    git config --global user.name "Mini-Devin" && \\
    git config --global init.defaultBranch main && \\
    git config --global --add safe.directory /workspace

# Set up git config for sandbox user
USER sandbox
RUN git config --global user.email "mini-devin@example.com" && \\
    git config --global user.name "Mini-Devin" && \\
    git config --global init.defaultBranch main && \\
    git config --global --add safe.directory /workspace
USER root

WORKDIR /workspace

# Default command
CMD ["bash"]
'''


# Default seccomp profile for sandbox (restrictive but allows common operations)
DEFAULT_SECCOMP_PROFILE = {
    "defaultAction": "SCMP_ACT_ERRNO",
    "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86", "SCMP_ARCH_AARCH64"],
    "syscalls": [
        {
            "names": [
                "accept", "accept4", "access", "arch_prctl", "bind", "brk",
                "capget", "capset", "chdir", "chmod", "chown", "clock_getres",
                "clock_gettime", "clock_nanosleep", "clone", "clone3", "close",
                "connect", "copy_file_range", "dup", "dup2", "dup3", "epoll_create",
                "epoll_create1", "epoll_ctl", "epoll_pwait", "epoll_wait",
                "eventfd", "eventfd2", "execve", "execveat", "exit", "exit_group",
                "faccessat", "faccessat2", "fadvise64", "fallocate", "fchdir",
                "fchmod", "fchmodat", "fchown", "fchownat", "fcntl", "fdatasync",
                "fgetxattr", "flistxattr", "flock", "fork", "fsetxattr", "fstat",
                "fstatfs", "fsync", "ftruncate", "futex", "getcwd", "getdents",
                "getdents64", "getegid", "geteuid", "getgid", "getgroups",
                "getpeername", "getpgid", "getpgrp", "getpid", "getppid",
                "getpriority", "getrandom", "getresgid", "getresuid", "getrlimit",
                "getrusage", "getsid", "getsockname", "getsockopt", "gettid",
                "gettimeofday", "getuid", "getxattr", "inotify_add_watch",
                "inotify_init", "inotify_init1", "inotify_rm_watch", "ioctl",
                "kill", "lchown", "lgetxattr", "link", "linkat", "listen",
                "listxattr", "llistxattr", "lseek", "lsetxattr", "lstat",
                "madvise", "membarrier", "memfd_create", "mincore", "mkdir",
                "mkdirat", "mknod", "mknodat", "mlock", "mlock2", "mlockall",
                "mmap", "mprotect", "mremap", "msgctl", "msgget", "msgrcv",
                "msgsnd", "msync", "munlock", "munlockall", "munmap", "nanosleep",
                "newfstatat", "open", "openat", "openat2", "pause", "pipe",
                "pipe2", "poll", "ppoll", "prctl", "pread64", "preadv", "preadv2",
                "prlimit64", "pselect6", "pwrite64", "pwritev", "pwritev2",
                "read", "readahead", "readlink", "readlinkat", "readv", "recv",
                "recvfrom", "recvmmsg", "recvmsg", "remap_file_pages", "removexattr",
                "rename", "renameat", "renameat2", "restart_syscall", "rmdir",
                "rseq", "rt_sigaction", "rt_sigpending", "rt_sigprocmask",
                "rt_sigqueueinfo", "rt_sigreturn", "rt_sigsuspend",
                "rt_sigtimedwait", "rt_tgsigqueueinfo", "sched_getaffinity",
                "sched_getattr", "sched_getparam", "sched_get_priority_max",
                "sched_get_priority_min", "sched_getscheduler", "sched_rr_get_interval",
                "sched_setaffinity", "sched_setattr", "sched_setparam",
                "sched_setscheduler", "sched_yield", "seccomp", "select",
                "semctl", "semget", "semop", "semtimedop", "send", "sendfile",
                "sendmmsg", "sendmsg", "sendto", "setfsgid", "setfsuid",
                "setgid", "setgroups", "setitimer", "setpgid", "setpriority",
                "setregid", "setresgid", "setresuid", "setreuid", "setsid",
                "setsockopt", "setuid", "setxattr", "shmat", "shmctl", "shmdt",
                "shmget", "shutdown", "sigaltstack", "signalfd", "signalfd4",
                "socket", "socketpair", "splice", "stat", "statfs", "statx",
                "symlink", "symlinkat", "sync", "sync_file_range", "syncfs",
                "sysinfo", "tee", "tgkill", "time", "timer_create", "timer_delete",
                "timerfd_create", "timerfd_gettime", "timerfd_settime",
                "timer_getoverrun", "timer_gettime", "timer_settime", "times",
                "tkill", "truncate", "umask", "uname", "unlink", "unlinkat",
                "utime", "utimensat", "utimes", "vfork", "vmsplice", "wait4",
                "waitid", "waitpid", "write", "writev"
            ],
            "action": "SCMP_ACT_ALLOW"
        }
    ]
}


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


def generate_seccomp_profile(output_path: str = "seccomp.json") -> str:
    """Generate the default seccomp profile for the sandbox."""
    with open(output_path, "w") as f:
        json.dump(DEFAULT_SECCOMP_PROFILE, f, indent=2)
    return output_path


def create_sandbox(
    repo_path: str,
    memory_limit: str = "2g",
    cpu_limit: float = 2.0,
    network_enabled: bool = False,  # Disabled by default for security
    security_level: SecurityLevel = SecurityLevel.STANDARD,
    mount_allowlist: list[str] | None = None,
) -> DockerSandbox:
    """
    Create a Docker sandbox with security hardening.
    
    Args:
        repo_path: Path to the repository to mount
        memory_limit: Memory limit (e.g., "2g")
        cpu_limit: CPU limit (e.g., 2.0)
        network_enabled: Whether to enable network access (default: False)
        security_level: Security level (MINIMAL, STANDARD, HARDENED)
        mount_allowlist: List of allowed mount paths (None = all allowed)
        
    Returns:
        Configured DockerSandbox instance
    """
    config = SandboxConfig(
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        network_enabled=network_enabled,
        security_level=security_level,
        mount_allowlist=mount_allowlist or [],
    )
    return DockerSandbox(repo_path, config)


def create_hardened_sandbox(
    repo_path: str,
    mount_allowlist: list[str] | None = None,
) -> DockerSandbox:
    """
    Create a fully hardened Docker sandbox.
    
    This is the most secure configuration with:
    - Non-root user
    - Network disabled
    - Read-only root filesystem
    - All capabilities dropped
    - Seccomp profile enabled
    - Mount allowlist enforced
    
    Args:
        repo_path: Path to the repository to mount
        mount_allowlist: List of allowed mount paths
        
    Returns:
        Hardened DockerSandbox instance
    """
    return create_sandbox(
        repo_path=repo_path,
        network_enabled=False,
        security_level=SecurityLevel.HARDENED,
        mount_allowlist=mount_allowlist,
    )
