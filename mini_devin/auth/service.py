"""Authentication service for Mini-Devin."""

import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import jwt, JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import UserModel, APIKeyModel


SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


class AuthService:
    """Service for authentication operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def create_access_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None,
    ) -> tuple[str, datetime]:
        """Create a JWT access token."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt, expire

    def decode_token(self, token: str) -> Optional[dict]:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None

    def generate_api_key(self) -> tuple[str, str, str]:
        """Generate a new API key. Returns (full_key, key_hash, key_prefix)."""
        key = f"md_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_prefix = key[:8]
        return key, key_hash, key_prefix

    def hash_api_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    async def get_user_by_email(self, email: str) -> Optional[UserModel]:
        """Get a user by email."""
        result = await self.db.execute(
            select(UserModel).where(UserModel.email == email)
        )
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[UserModel]:
        """Get a user by username."""
        result = await self.db.execute(
            select(UserModel).where(UserModel.username == username)
        )
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: str) -> Optional[UserModel]:
        """Get a user by ID."""
        result = await self.db.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        username: str,
        password: str,
        is_admin: bool = False,
    ) -> UserModel:
        """Create a new user."""
        import uuid
        
        user = UserModel(
            id=str(uuid.uuid4()),
            email=email,
            username=username,
            hashed_password=self.hash_password(password),
            is_admin=is_admin,
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def authenticate_user(
        self,
        username: str,
        password: str,
    ) -> Optional[UserModel]:
        """Authenticate a user by username and password."""
        user = await self.get_user_by_username(username)
        if not user:
            user = await self.get_user_by_email(username)
        
        if not user:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        if not user.is_active:
            return None
        
        user.last_login_at = datetime.now(timezone.utc)
        await self.db.commit()
        
        return user

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        expires_in_days: Optional[int] = None,
    ) -> tuple[APIKeyModel, str]:
        """Create a new API key. Returns (api_key_model, full_key)."""
        import uuid
        
        full_key, key_hash, key_prefix = self.generate_api_key()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        api_key = APIKeyModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            expires_at=expires_at,
        )
        self.db.add(api_key)
        await self.db.commit()
        await self.db.refresh(api_key)
        
        return api_key, full_key

    async def get_api_key_by_key(self, key: str) -> Optional[APIKeyModel]:
        """Get an API key by the full key."""
        key_hash = self.hash_api_key(key)
        result = await self.db.execute(
            select(APIKeyModel).where(APIKeyModel.key_hash == key_hash)
        )
        return result.scalar_one_or_none()

    async def validate_api_key(self, key: str) -> Optional[UserModel]:
        """Validate an API key and return the associated user."""
        api_key = await self.get_api_key_by_key(key)
        
        if not api_key:
            return None
        
        if not api_key.is_active:
            return None
        
        if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
            return None
        
        api_key.last_used_at = datetime.now(timezone.utc)
        await self.db.commit()
        
        user = await self.get_user_by_id(api_key.user_id)
        if not user or not user.is_active:
            return None
        
        return user

    async def list_api_keys(self, user_id: str) -> list[APIKeyModel]:
        """List all API keys for a user."""
        result = await self.db.execute(
            select(APIKeyModel).where(APIKeyModel.user_id == user_id)
        )
        return list(result.scalars().all())

    async def revoke_api_key(self, user_id: str, api_key_id: str) -> bool:
        """Revoke an API key."""
        result = await self.db.execute(
            select(APIKeyModel).where(
                APIKeyModel.id == api_key_id,
                APIKeyModel.user_id == user_id,
            )
        )
        api_key = result.scalar_one_or_none()
        
        if not api_key:
            return False
        
        api_key.is_active = False
        await self.db.commit()
        return True

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> bool:
        """Change a user's password."""
        user = await self.get_user_by_id(user_id)
        
        if not user:
            return False
        
        if not self.verify_password(current_password, user.hashed_password):
            return False
        
        user.hashed_password = self.hash_password(new_password)
        await self.db.commit()
        return True
