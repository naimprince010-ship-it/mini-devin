"""
Secrets Module for Mini-Devin

This module provides secure credential management for:
- API keys and tokens
- Database credentials
- Service passwords
- Environment variables
"""

from .manager import SecretsManager, Secret, SecretScope

__all__ = [
    "SecretsManager",
    "Secret",
    "SecretScope",
]
