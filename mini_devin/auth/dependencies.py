"""FastAPI dependencies for authentication."""

from typing import Optional

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from ..database.config import get_session
from ..database.models import UserModel
from .service import AuthService


http_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    api_key: Optional[str] = Depends(api_key_header),
) -> UserModel:
    """Get the current authenticated user (required)."""
    async for db in get_session():
        auth_service = AuthService(db)
        
        if credentials and credentials.credentials:
            payload = auth_service.decode_token(credentials.credentials)
            if payload:
                user_id = payload.get("sub")
                if user_id:
                    user = await auth_service.get_user_by_id(user_id)
                    if user and user.is_active:
                        return user
        
        if api_key:
            user = await auth_service.validate_api_key(api_key)
            if user:
                return user
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[UserModel]:
    """Get the current authenticated user (optional)."""
    async for db in get_session():
        auth_service = AuthService(db)
        
        if credentials and credentials.credentials:
            payload = auth_service.decode_token(credentials.credentials)
            if payload:
                user_id = payload.get("sub")
                if user_id:
                    user = await auth_service.get_user_by_id(user_id)
                    if user and user.is_active:
                        return user
        
        if api_key:
            user = await auth_service.validate_api_key(api_key)
            if user:
                return user
        
        return None


async def get_api_key_user(
    api_key: str = Depends(api_key_header),
) -> UserModel:
    """Get user from API key only."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    
    async for db in get_session():
        auth_service = AuthService(db)
        user = await auth_service.validate_api_key(api_key)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        
        return user


async def get_admin_user(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    """Get the current user and verify they are an admin."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
