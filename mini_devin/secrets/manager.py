"""
Secrets Manager for Mini-Devin

This module provides secure credential management with:
- Fernet encryption (AES-128-CBC with HMAC)
- Scoped access (global, session, task)
- Environment variable injection
- Audit logging
- Secret redaction in logs/artifacts
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
import secrets as crypto_secrets

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecretScope(str, Enum):
    """Scope of a secret."""
    GLOBAL = "global"  # Available to all sessions
    SESSION = "session"  # Available to a specific session
    TASK = "task"  # Available to a specific task


@dataclass
class Secret:
    """A secret credential."""
    name: str
    scope: SecretScope
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    
    # Scope identifiers
    session_id: str | None = None
    task_id: str | None = None
    
    # Encrypted value (not stored in plain text)
    _encrypted_value: str = ""
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class SecretsManager:
    """
    Manages secrets securely.
    
    Features:
    - Simple encryption using a master key
    - Scoped access control
    - Environment variable injection
    - Audit logging
    - Persistence to encrypted file
    """
    
    def __init__(
        self,
        storage_path: str | None = None,
        master_key: str | None = None,
    ):
        """
        Initialize the secrets manager.
        
        Args:
            storage_path: Path to store encrypted secrets
            master_key: Master key for encryption (generated if not provided)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Generate or use provided master key
        if master_key:
            self._master_key = master_key.encode()
        else:
            self._master_key = self._get_or_create_master_key()
        
        # In-memory secret storage
        self._secrets: dict[str, Secret] = {}
        self._values: dict[str, str] = {}  # name -> decrypted value
        
        # Audit log
        self._audit_log: list[dict[str, Any]] = []
        
        # Load existing secrets
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create the master key."""
        # Check environment variable first
        env_key = os.environ.get("MINI_DEVIN_MASTER_KEY")
        if env_key:
            return env_key.encode()
        
        # Check for key file
        key_file = Path.home() / ".mini-devin" / "master.key"
        if key_file.exists():
            return key_file.read_bytes()
        
        # Generate new key (32 bytes for Fernet key derivation)
        key = crypto_secrets.token_bytes(32)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(key)
        key_file.chmod(0o600)  # Restrict permissions
        
        return key
    
    def _derive_fernet_key(self, salt: bytes | None = None) -> tuple[Fernet, bytes]:
        """
        Derive a Fernet key from the master key using PBKDF2.
        
        Args:
            salt: Optional salt for key derivation (generated if not provided)
            
        Returns:
            Tuple of (Fernet instance, salt used)
        """
        if salt is None:
            salt = crypto_secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # OWASP recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._master_key))
        return Fernet(key), salt
    
    def _encrypt(self, value: str) -> str:
        """
        Encrypt a value using Fernet (AES-128-CBC with HMAC).
        
        The encrypted format is: base64(salt + encrypted_data)
        """
        fernet, salt = self._derive_fernet_key()
        encrypted = fernet.encrypt(value.encode())
        # Prepend salt to encrypted data
        combined = salt + encrypted
        return base64.b64encode(combined).decode()
    
    def _decrypt(self, encrypted: str) -> str:
        """
        Decrypt a Fernet-encrypted value.
        
        Handles both new Fernet format and legacy XOR format for migration.
        """
        try:
            combined = base64.b64decode(encrypted.encode())
            # Extract salt (first 16 bytes) and encrypted data
            salt = combined[:16]
            encrypted_data = combined[16:]
            
            fernet, _ = self._derive_fernet_key(salt)
            return fernet.decrypt(encrypted_data).decode()
        except (InvalidToken, ValueError):
            # Try legacy XOR decryption for backward compatibility
            return self._decrypt_legacy(encrypted)
    
    def _decrypt_legacy(self, encrypted: str) -> str:
        """Decrypt using legacy XOR method (for migration)."""
        key = hashlib.sha256(self._master_key).digest()
        encrypted_bytes = base64.b64decode(encrypted.encode())
        decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(encrypted_bytes))
        return decrypted.decode()
    
    def set(
        self,
        name: str,
        value: str,
        scope: SecretScope = SecretScope.GLOBAL,
        session_id: str | None = None,
        task_id: str | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Secret:
        """
        Set a secret.
        
        Args:
            name: Secret name
            value: Secret value
            scope: Secret scope
            session_id: Session ID (for session/task scope)
            task_id: Task ID (for task scope)
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            The created/updated secret
        """
        # Validate scope
        if scope == SecretScope.SESSION and not session_id:
            raise ValueError("session_id required for session scope")
        if scope == SecretScope.TASK and (not session_id or not task_id):
            raise ValueError("session_id and task_id required for task scope")
        
        # Create secret key
        secret_key = self._make_key(name, scope, session_id, task_id)
        
        # Check if updating existing secret
        is_update = secret_key in self._secrets
        
        # Create or update secret
        secret = Secret(
            name=name,
            scope=scope,
            session_id=session_id,
            task_id=task_id,
            description=description,
            metadata=metadata or {},
            _encrypted_value=self._encrypt(value),
        )
        
        if is_update:
            secret.created_at = self._secrets[secret_key].created_at
        
        self._secrets[secret_key] = secret
        self._values[secret_key] = value
        
        # Audit log
        self._log_access(
            action="set" if not is_update else "update",
            secret_name=name,
            scope=scope,
            session_id=session_id,
            task_id=task_id,
        )
        
        # Persist
        if self.storage_path:
            self._save()
        
        return secret
    
    def get(
        self,
        name: str,
        scope: SecretScope = SecretScope.GLOBAL,
        session_id: str | None = None,
        task_id: str | None = None,
        default: str | None = None,
    ) -> str | None:
        """
        Get a secret value.
        
        Args:
            name: Secret name
            scope: Secret scope
            session_id: Session ID (for session/task scope)
            task_id: Task ID (for task scope)
            default: Default value if not found
            
        Returns:
            The secret value or default
        """
        secret_key = self._make_key(name, scope, session_id, task_id)
        
        if secret_key not in self._values:
            # Try to fall back to broader scope
            if scope == SecretScope.TASK and session_id:
                return self.get(name, SecretScope.SESSION, session_id, default=default)
            elif scope == SecretScope.SESSION:
                return self.get(name, SecretScope.GLOBAL, default=default)
            return default
        
        # Audit log
        self._log_access(
            action="get",
            secret_name=name,
            scope=scope,
            session_id=session_id,
            task_id=task_id,
        )
        
        return self._values[secret_key]
    
    def delete(
        self,
        name: str,
        scope: SecretScope = SecretScope.GLOBAL,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> bool:
        """
        Delete a secret.
        
        Args:
            name: Secret name
            scope: Secret scope
            session_id: Session ID (for session/task scope)
            task_id: Task ID (for task scope)
            
        Returns:
            True if deleted, False if not found
        """
        secret_key = self._make_key(name, scope, session_id, task_id)
        
        if secret_key not in self._secrets:
            return False
        
        del self._secrets[secret_key]
        del self._values[secret_key]
        
        # Audit log
        self._log_access(
            action="delete",
            secret_name=name,
            scope=scope,
            session_id=session_id,
            task_id=task_id,
        )
        
        # Persist
        if self.storage_path:
            self._save()
        
        return True
    
    def list(
        self,
        scope: SecretScope | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> list[Secret]:
        """
        List secrets (without values).
        
        Args:
            scope: Filter by scope
            session_id: Filter by session ID
            task_id: Filter by task ID
            
        Returns:
            List of secrets (without values)
        """
        results = []
        
        for secret in self._secrets.values():
            # Apply filters
            if scope and secret.scope != scope:
                continue
            if session_id and secret.session_id != session_id:
                continue
            if task_id and secret.task_id != task_id:
                continue
            
            results.append(secret)
        
        return results
    
    def inject_env(
        self,
        scope: SecretScope = SecretScope.GLOBAL,
        session_id: str | None = None,
        task_id: str | None = None,
        prefix: str = "",
    ) -> dict[str, str]:
        """
        Get secrets as environment variables.
        
        Args:
            scope: Secret scope
            session_id: Session ID
            task_id: Task ID
            prefix: Prefix for env var names
            
        Returns:
            Dictionary of environment variables
        """
        env_vars = {}
        
        # Get all applicable secrets
        for secret_key, value in self._values.items():
            secret = self._secrets[secret_key]
            
            # Check scope hierarchy
            if scope == SecretScope.TASK:
                if secret.scope == SecretScope.TASK:
                    if secret.session_id != session_id or secret.task_id != task_id:
                        continue
                elif secret.scope == SecretScope.SESSION:
                    if secret.session_id != session_id:
                        continue
                # Global secrets are always included
            elif scope == SecretScope.SESSION:
                if secret.scope == SecretScope.TASK:
                    continue
                if secret.scope == SecretScope.SESSION and secret.session_id != session_id:
                    continue
            else:  # GLOBAL
                if secret.scope != SecretScope.GLOBAL:
                    continue
            
            # Convert name to env var format
            env_name = f"{prefix}{secret.name}".upper().replace("-", "_").replace(".", "_")
            env_vars[env_name] = value
        
        return env_vars
    
    def _make_key(
        self,
        name: str,
        scope: SecretScope,
        session_id: str | None,
        task_id: str | None,
    ) -> str:
        """Create a unique key for a secret."""
        parts = [scope.value, name]
        if session_id:
            parts.append(session_id)
        if task_id:
            parts.append(task_id)
        return ":".join(parts)
    
    def _log_access(
        self,
        action: str,
        secret_name: str,
        scope: SecretScope,
        session_id: str | None,
        task_id: str | None,
    ) -> None:
        """Log a secret access."""
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "secret_name": secret_name,
            "scope": scope.value,
            "session_id": session_id,
            "task_id": task_id,
        })
        
        # Keep only last 1000 entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]
    
    def get_audit_log(
        self,
        limit: int = 100,
        secret_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get audit log entries."""
        entries = self._audit_log
        
        if secret_name:
            entries = [e for e in entries if e["secret_name"] == secret_name]
        
        return entries[-limit:]
    
    def _save(self) -> None:
        """Save secrets to encrypted file."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize secrets (values are already encrypted)
        data = {
            "secrets": {
                key: {
                    "name": s.name,
                    "scope": s.scope.value,
                    "session_id": s.session_id,
                    "task_id": s.task_id,
                    "description": s.description,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "encrypted_value": s._encrypted_value,
                    "metadata": s.metadata,
                }
                for key, s in self._secrets.items()
            }
        }
        
        self.storage_path.write_text(json.dumps(data, indent=2))
        self.storage_path.chmod(0o600)  # Restrict permissions
    
    def _load(self) -> None:
        """Load secrets from encrypted file."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            
            for key, s_data in data.get("secrets", {}).items():
                secret = Secret(
                    name=s_data["name"],
                    scope=SecretScope(s_data["scope"]),
                    session_id=s_data.get("session_id"),
                    task_id=s_data.get("task_id"),
                    description=s_data.get("description", ""),
                    created_at=datetime.fromisoformat(s_data["created_at"]),
                    updated_at=datetime.fromisoformat(s_data["updated_at"]),
                    _encrypted_value=s_data["encrypted_value"],
                    metadata=s_data.get("metadata", {}),
                )
                
                self._secrets[key] = secret
                self._values[key] = self._decrypt(secret._encrypted_value)
                
        except Exception:
            # If loading fails, start fresh
            pass
    
    def clear_session(self, session_id: str) -> int:
        """
        Clear all secrets for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            Number of secrets deleted
        """
        to_delete = []
        
        for key, secret in self._secrets.items():
            if secret.session_id == session_id:
                to_delete.append(key)
        
        for key in to_delete:
            del self._secrets[key]
            del self._values[key]
        
        if to_delete and self.storage_path:
            self._save()
        
        return len(to_delete)
    
    def clear_task(self, session_id: str, task_id: str) -> int:
        """
        Clear all secrets for a task.
        
        Args:
            session_id: The session ID
            task_id: The task ID
            
        Returns:
            Number of secrets deleted
        """
        to_delete = []
        
        for key, secret in self._secrets.items():
            if secret.session_id == session_id and secret.task_id == task_id:
                to_delete.append(key)
        
        for key in to_delete:
            del self._secrets[key]
            del self._values[key]
        
        if to_delete and self.storage_path:
            self._save()
        
        return len(to_delete)


class SecretRedactor:
    """
    Utility for redacting secrets from text content.
    
    Use this to prevent accidental exposure of secrets in:
    - Log files
    - Artifacts
    - Error messages
    - API responses
    """
    
    REDACTED_PLACEHOLDER = "[REDACTED]"
    
    SECRET_PATTERNS = [
        (r'(?i)(api[_-]?key|apikey|api[_-]?token)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', 2),
        (r'(?i)(bearer|token|auth)\s+([a-zA-Z0-9_\-\.]{20,})', 2),
        (r'(?i)(sk-[a-zA-Z0-9]{20,})', 1),
        (r'(?i)(ghp_[a-zA-Z0-9]{36})', 1),
        (r'(?i)(gho_[a-zA-Z0-9]{36})', 1),
        (r'(?i)(github_pat_[a-zA-Z0-9_]{22,})', 1),
        (r'(?i)(AKIA[0-9A-Z]{16})', 1),
        (r'(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', 2),
        (r'(?i)(postgres|mysql|mongodb)://[^:]+:([^@]+)@', 2),
        (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\']{8,})["\']?', 2),
        (r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----', 0),
        (r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', 0),
    ]
    
    def __init__(self, secrets_manager: SecretsManager | None = None):
        self._secrets_manager = secrets_manager
        self._custom_patterns: list[tuple[str, int]] = []
        self._known_secrets: set[str] = set()
    
    def add_pattern(self, pattern: str, group: int = 0) -> None:
        self._custom_patterns.append((pattern, group))
    
    def add_known_secret(self, value: str) -> None:
        if value and len(value) >= 4:
            self._known_secrets.add(value)
    
    def load_from_manager(
        self,
        scope: SecretScope = SecretScope.GLOBAL,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        if not self._secrets_manager:
            return
        
        env_vars = self._secrets_manager.inject_env(
            scope=scope,
            session_id=session_id,
            task_id=task_id,
        )
        
        for value in env_vars.values():
            self.add_known_secret(value)
    
    def redact(self, text: str) -> str:
        if not text:
            return text
        
        result = text
        
        for secret in self._known_secrets:
            if secret in result:
                result = result.replace(secret, self.REDACTED_PLACEHOLDER)
        
        all_patterns = self.SECRET_PATTERNS + self._custom_patterns
        
        for pattern, group in all_patterns:
            try:
                if group == 0:
                    result = re.sub(pattern, self.REDACTED_PLACEHOLDER, result)
                else:
                    def replace_group(match: re.Match) -> str:
                        full = match.group(0)
                        original = match.group(group) if group <= len(match.groups()) else ""
                        return full.replace(original, self.REDACTED_PLACEHOLDER) if original else full
                    
                    result = re.sub(pattern, replace_group, result)
            except re.error:
                continue
        
        return result
    
    def redact_dict(self, data: dict[str, Any], sensitive_keys: set[str] | None = None) -> dict[str, Any]:
        if sensitive_keys is None:
            sensitive_keys = {
                "password", "passwd", "pwd", "secret", "token", "api_key",
                "apikey", "auth", "credential", "private_key", "access_key",
                "secret_key", "bearer", "authorization",
            }
        
        def redact_value(key: str, value: Any) -> Any:
            key_lower = key.lower()
            
            if any(sk in key_lower for sk in sensitive_keys):
                if isinstance(value, str):
                    return self.REDACTED_PLACEHOLDER
                elif isinstance(value, (list, tuple)):
                    return [self.REDACTED_PLACEHOLDER] * len(value)
            
            if isinstance(value, dict):
                return self.redact_dict(value, sensitive_keys)
            elif isinstance(value, list):
                return [redact_value(key, item) for item in value]
            elif isinstance(value, str):
                return self.redact(value)
            
            return value
        
        return {k: redact_value(k, v) for k, v in data.items()}
    
    def redact_file(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            return ""
        
        try:
            content = path.read_text()
            return self.redact(content)
        except Exception:
            return f"[Unable to read file: {file_path}]"
    
    def create_safe_error(self, error: Exception) -> str:
        return self.redact(str(error))
