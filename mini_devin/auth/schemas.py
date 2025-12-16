"""Pydantic schemas for authentication."""

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=128)
    password: str = Field(..., min_length=8, max_length=128)


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""
    user_id: str
    email: str
    username: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login_at: datetime | None = None


class TokenResponse(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyCreate(BaseModel):
    """Schema for creating an API key."""
    name: str = Field(..., min_length=1, max_length=128)
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """Schema for API key response."""
    api_key_id: str
    name: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    expires_at: datetime | None = None


class APIKeyCreatedResponse(BaseModel):
    """Schema for newly created API key (includes full key)."""
    api_key_id: str
    name: str
    key: str
    key_prefix: str
    created_at: datetime
    expires_at: datetime | None = None


class PasswordChange(BaseModel):
    """Schema for changing password."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
