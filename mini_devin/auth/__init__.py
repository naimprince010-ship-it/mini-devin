"""Authentication module for Mini-Devin."""

from .service import AuthService
from .dependencies import get_current_user, get_current_user_optional, get_api_key_user
from .schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    APIKeyCreate,
    APIKeyResponse,
)

__all__ = [
    "AuthService",
    "get_current_user",
    "get_current_user_optional",
    "get_api_key_user",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "TokenResponse",
    "APIKeyCreate",
    "APIKeyResponse",
]
