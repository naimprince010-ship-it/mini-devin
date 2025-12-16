"""
Sandbox Security Configuration for Mini-Devin (Phase 6D).

This module provides security settings for the Docker sandbox environment,
including non-root user configuration, resource limits, and network isolation.
"""

import os
from dataclasses import dataclass


@dataclass
class SandboxSecuritySettings:
    """
    Security settings for the Docker sandbox environment.
    
    These settings control how the sandbox container is configured
    for security hardening.
    """
    
    enabled: bool = True
    run_as_non_root: bool = True
    user_id: int = 1000
    group_id: int = 1000
    username: str = "minidevin"
    
    read_only_root: bool = False
    
    cpu_limit: float = 2.0
    memory_limit: str = "4G"
    pid_limit: int = 256
    tmp_size: int = 536870912  # 512MB
    
    nofile_soft: int = 65536
    nofile_hard: int = 65536
    nproc_soft: int = 256
    nproc_hard: int = 512
    
    network_isolation: bool = False
    allowed_hosts: list[str] | None = None
    
    drop_all_capabilities: bool = True
    allowed_capabilities: list[str] | None = None
    no_new_privileges: bool = True
    
    @classmethod
    def from_env(cls) -> "SandboxSecuritySettings":
        """Load sandbox security settings from environment variables."""
        allowed_hosts_str = os.environ.get("ALLOWED_HOSTS", "")
        allowed_hosts = [h.strip() for h in allowed_hosts_str.split(",") if h.strip()] or None
        
        allowed_caps_str = os.environ.get("ALLOWED_CAPABILITIES", "CHOWN,DAC_OVERRIDE,FOWNER,SETGID,SETUID")
        allowed_caps = [c.strip() for c in allowed_caps_str.split(",") if c.strip()] or None
        
        return cls(
            enabled=os.environ.get("SANDBOX_ENABLED", "true").lower() == "true",
            run_as_non_root=os.environ.get("RUN_AS_NON_ROOT", "true").lower() == "true",
            user_id=int(os.environ.get("USER_ID", "1000")),
            group_id=int(os.environ.get("GROUP_ID", "1000")),
            username=os.environ.get("SANDBOX_USER", "minidevin"),
            read_only_root=os.environ.get("READ_ONLY_ROOT", "false").lower() == "true",
            cpu_limit=float(os.environ.get("CPU_LIMIT", "2.0")),
            memory_limit=os.environ.get("MEMORY_LIMIT", "4G"),
            pid_limit=int(os.environ.get("PID_LIMIT", "256")),
            tmp_size=int(os.environ.get("TMP_SIZE", "536870912")),
            nofile_soft=int(os.environ.get("NOFILE_SOFT", "65536")),
            nofile_hard=int(os.environ.get("NOFILE_HARD", "65536")),
            nproc_soft=int(os.environ.get("NPROC_SOFT", "256")),
            nproc_hard=int(os.environ.get("NPROC_HARD", "512")),
            network_isolation=os.environ.get("NETWORK_ISOLATION", "false").lower() == "true",
            allowed_hosts=allowed_hosts,
            drop_all_capabilities=os.environ.get("DROP_ALL_CAPABILITIES", "true").lower() == "true",
            allowed_capabilities=allowed_caps,
            no_new_privileges=os.environ.get("NO_NEW_PRIVILEGES", "true").lower() == "true",
        )
    
    def validate(self) -> list[str]:
        """
        Validate sandbox security settings.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.user_id < 1000:
            errors.append("USER_ID should be >= 1000 for non-system users")
        
        if self.group_id < 1000:
            errors.append("GROUP_ID should be >= 1000 for non-system groups")
        
        if self.cpu_limit <= 0:
            errors.append("CPU_LIMIT must be positive")
        
        if self.pid_limit < 10:
            errors.append("PID_LIMIT must be at least 10")
        
        if self.tmp_size < 1048576:  # 1MB minimum
            errors.append("TMP_SIZE must be at least 1MB (1048576 bytes)")
        
        if self.nofile_soft > self.nofile_hard:
            errors.append("NOFILE_SOFT cannot exceed NOFILE_HARD")
        
        if self.nproc_soft > self.nproc_hard:
            errors.append("NPROC_SOFT cannot exceed NPROC_HARD")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if settings are valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> dict[str, object]:
        """Convert settings to dictionary for logging/serialization."""
        return {
            "enabled": self.enabled,
            "run_as_non_root": self.run_as_non_root,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "username": self.username,
            "read_only_root": self.read_only_root,
            "resource_limits": {
                "cpu_limit": self.cpu_limit,
                "memory_limit": self.memory_limit,
                "pid_limit": self.pid_limit,
                "tmp_size": self.tmp_size,
            },
            "ulimits": {
                "nofile_soft": self.nofile_soft,
                "nofile_hard": self.nofile_hard,
                "nproc_soft": self.nproc_soft,
                "nproc_hard": self.nproc_hard,
            },
            "network": {
                "isolation": self.network_isolation,
                "allowed_hosts": self.allowed_hosts,
            },
            "capabilities": {
                "drop_all": self.drop_all_capabilities,
                "allowed": self.allowed_capabilities,
                "no_new_privileges": self.no_new_privileges,
            },
        }
    
    def get_docker_run_args(self) -> list[str]:
        """
        Generate Docker run arguments for security settings.
        
        Returns:
            List of Docker run arguments
        """
        args = []
        
        if self.run_as_non_root:
            args.extend(["--user", f"{self.user_id}:{self.group_id}"])
        
        if self.read_only_root:
            args.append("--read-only")
        
        args.extend(["--cpus", str(self.cpu_limit)])
        args.extend(["--memory", self.memory_limit])
        args.extend(["--pids-limit", str(self.pid_limit)])
        
        args.extend(["--tmpfs", f"/tmp:size={self.tmp_size},mode=1777"])
        
        args.extend(["--ulimit", f"nofile={self.nofile_soft}:{self.nofile_hard}"])
        args.extend(["--ulimit", f"nproc={self.nproc_soft}:{self.nproc_hard}"])
        
        if self.drop_all_capabilities:
            args.append("--cap-drop=ALL")
            if self.allowed_capabilities:
                for cap in self.allowed_capabilities:
                    args.append(f"--cap-add={cap}")
        
        if self.no_new_privileges:
            args.append("--security-opt=no-new-privileges:true")
        
        return args


_sandbox_settings: SandboxSecuritySettings | None = None


def get_sandbox_settings() -> SandboxSecuritySettings:
    """
    Get the global sandbox security settings instance.
    
    Settings are loaded from environment variables on first access.
    """
    global _sandbox_settings
    if _sandbox_settings is None:
        _sandbox_settings = SandboxSecuritySettings.from_env()
    return _sandbox_settings


def reload_sandbox_settings() -> SandboxSecuritySettings:
    """Reload sandbox security settings from environment variables."""
    global _sandbox_settings
    _sandbox_settings = SandboxSecuritySettings.from_env()
    return _sandbox_settings
