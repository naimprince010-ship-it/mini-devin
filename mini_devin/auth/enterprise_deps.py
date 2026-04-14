"""Map authenticated users to enterprise :class:`~mini_devin.enterprise.rbac.Role` and enforce permissions."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status

from mini_devin.auth.dependencies import get_current_user
from mini_devin.database.models import UserModel
from mini_devin.enterprise.rbac import Permission, Role, role_allows


def user_to_role(user: UserModel) -> Role:
    """Derive a coarse RBAC role until a dedicated ``users.role`` column exists."""
    if getattr(user, "is_admin", False):
        return Role.ADMIN
    return Role.MEMBER


class RequirePermission:
    """FastAPI dependency: require ``permission`` for the current user."""

    def __init__(self, permission: Permission) -> None:
        self.permission = permission

    async def __call__(self, user: UserModel = Depends(get_current_user)) -> UserModel:
        role = user_to_role(user)
        if not role_allows(role, self.permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {self.permission.value}",
            )
        return user
