"""Authentication routes for Mini-Devin."""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from ..database.config import get_session
from ..database.models import UserModel
from .service import AuthService, ACCESS_TOKEN_EXPIRE_MINUTES
from .dependencies import get_current_user
from .schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    APIKeyCreate,
    APIKeyResponse,
    APIKeyCreatedResponse,
    PasswordChange,
)


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user."""
    async for db in get_session():
        auth_service = AuthService(db)
        
        existing_email = await auth_service.get_user_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        
        existing_username = await auth_service.get_user_by_username(user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken",
            )
        
        user = await auth_service.create_user(
            email=user_data.email,
            username=user_data.username,
            password=user_data.password,
        )
        
        return UserResponse(**user.to_dict())


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """Login and get an access token."""
    async for db in get_session():
        auth_service = AuthService(db)
        
        user = await auth_service.authenticate_user(
            username=credentials.username,
            password=credentials.password,
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token, expires = auth_service.create_access_token(
            user_id=user.id,
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserModel = Depends(get_current_user)):
    """Get the current user's profile."""
    return UserResponse(**current_user.to_dict())


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: UserModel = Depends(get_current_user),
):
    """Change the current user's password."""
    async for db in get_session():
        auth_service = AuthService(db)
        
        success = await auth_service.change_password(
            user_id=current_user.id,
            current_password=password_data.current_password,
            new_password=password_data.new_password,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )
        
        return {"message": "Password changed successfully"}


@router.post("/api-keys", response_model=APIKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: UserModel = Depends(get_current_user),
):
    """Create a new API key."""
    async for db in get_session():
        auth_service = AuthService(db)
        
        api_key, full_key = await auth_service.create_api_key(
            user_id=current_user.id,
            name=key_data.name,
            expires_in_days=key_data.expires_in_days,
        )
        
        return APIKeyCreatedResponse(
            api_key_id=api_key.id,
            name=api_key.name,
            key=full_key,
            key_prefix=api_key.key_prefix,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
        )


@router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(current_user: UserModel = Depends(get_current_user)):
    """List all API keys for the current user."""
    async for db in get_session():
        auth_service = AuthService(db)
        
        api_keys = await auth_service.list_api_keys(current_user.id)
        
        return [
            APIKeyResponse(**key.to_dict())
            for key in api_keys
        ]


@router.delete("/api-keys/{api_key_id}")
async def revoke_api_key(
    api_key_id: str,
    current_user: UserModel = Depends(get_current_user),
):
    """Revoke an API key."""
    async for db in get_session():
        auth_service = AuthService(db)
        
        success = await auth_service.revoke_api_key(current_user.id, api_key_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        
        return {"message": "API key revoked successfully"}
