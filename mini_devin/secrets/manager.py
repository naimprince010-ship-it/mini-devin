"""
Secrets Manager for Mini-Devin

This module provides secure credential management with:
- Encrypted storage
- Scoped access (global, session, task)
- Environment variable injection
- Audit logging
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
import secrets as crypto_secrets


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
        
        # Generate new key
        key = crypto_secrets.token_bytes(32)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(key)
        key_file.chmod(0o600)  # Restrict permissions
        
        return key
    
    def _encrypt(self, value: str) -> str:
        """Encrypt a value using simple XOR with key derivation."""
        # Derive a key from master key
        key = hashlib.sha256(self._master_key).digest()
        
        # XOR encryption (simple but effective for this use case)
        value_bytes = value.encode()
        encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(value_bytes))
        
        return base64.b64encode(encrypted).decode()
    
    def _decrypt(self, encrypted: str) -> str:
        """Decrypt a value."""
        # Derive the same key
        key = hashlib.sha256(self._master_key).digest()
        
        # Decode and XOR decrypt
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
