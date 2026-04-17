"""
Sandbox Module for Plodder

This module provides Docker-based sandboxing for safe code execution:
- Isolated container environment
- Repo mounted as /workspace
- No host filesystem access
- Resource limits (CPU, memory)
- Non-root user execution
- Read-only filesystem with explicit mount allowlists
- Network disabled by default
- Seccomp security profiles
"""

from .docker_sandbox import (
    DockerSandbox,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
    SecurityLevel,
    create_sandbox,
    create_hardened_sandbox,
    generate_dockerfile,
    generate_seccomp_profile,
    build_sandbox_image,
    DOCKERFILE_CONTENT,
    DEFAULT_SECCOMP_PROFILE,
)
from .e2b_sandbox import E2BSandbox
from .factory import create_execution_sandbox, get_sandbox_backend
from .process_sandbox import ProcessSandbox
from .process_execution_sandbox import ProcessExecutionSandbox, use_host_process_terminal_for_tooling
from .runtime_protocol import CommandRuntime, ExecutionResult, normalize_exec_result
from .sandbox import SimpleDockerSandbox, ensure_sandbox_image
from .shell_text import strip_ansi
from .stateful_exec import build_stateful_bash_script, posix_paths_for_docker_workspace, shell_state_dir

__all__ = [
    "DockerSandbox",
    "E2BSandbox",
    "ProcessExecutionSandbox",
    "use_host_process_terminal_for_tooling",
    "ProcessSandbox",
    "CommandRuntime",
    "ExecutionResult",
    "normalize_exec_result",
    "strip_ansi",
    "build_stateful_bash_script",
    "posix_paths_for_docker_workspace",
    "shell_state_dir",
    "SimpleDockerSandbox",
    "ensure_sandbox_image",
    "SandboxConfig",
    "SandboxResult",
    "SandboxStatus",
    "SecurityLevel",
    "create_sandbox",
    "create_execution_sandbox",
    "get_sandbox_backend",
    "create_hardened_sandbox",
    "generate_dockerfile",
    "generate_seccomp_profile",
    "build_sandbox_image",
    "DOCKERFILE_CONTENT",
    "DEFAULT_SECCOMP_PROFILE",
]
