"""Minimal role/permission model for multi-user deployments (extend as needed)."""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(str, Enum):
    RUN_AGENT = "run_agent"
    SHARE_CONVERSATION = "share_conversation"
    MANAGE_USERS = "manage_users"
    MANAGE_INTEGRATIONS = "manage_integrations"
    MANAGE_API_KEYS = "manage_api_keys"


_ROLE_MATRIX: dict[Role, frozenset[Permission]] = {
    Role.ADMIN: frozenset(Permission),
    Role.MEMBER: frozenset(
        {
            Permission.RUN_AGENT,
            Permission.SHARE_CONVERSATION,
            Permission.MANAGE_API_KEYS,
        },
    ),
    Role.VIEWER: frozenset({Permission.SHARE_CONVERSATION}),
}


def role_allows(role: Role, permission: Permission) -> bool:
    """Return whether ``role`` grants ``permission`` (default matrix)."""
    allowed = _ROLE_MATRIX.get(role)
    if allowed is None:
        return False
    return permission in allowed
