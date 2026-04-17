"""Database configuration and connection management."""

import os
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def _normalize_async_database_url(url: str) -> str:
    """Railway/Heroku often set postgres:// or postgresql:// without a SQLAlchemy async driver."""
    if not url or url.startswith("sqlite"):
        return url
    if "+asyncpg" in url or "+psycopg" in url or "+psycopg2" in url:
        return url
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


def get_database_url() -> str:
    """Get the database URL from environment variables."""
    raw = os.getenv("DATABASE_URL")
    if not raw:
        return "sqlite+aiosqlite:///plodder.db"
    return _normalize_async_database_url(raw.strip())


def get_sync_database_url() -> str:
    """Get the synchronous database URL for Alembic migrations."""
    url = get_database_url()
    if url.startswith("sqlite+aiosqlite"):
        return url.replace("sqlite+aiosqlite", "sqlite")
    return url.replace("+asyncpg", "")


_engine = None
_async_session_maker = None


def _is_sqlite() -> bool:
    """Check if we are using SQLite."""
    return get_database_url().startswith("sqlite")


def _asyncpg_connect_args(url: str) -> dict[str, Any]:
    """asyncpg ignores libpq sslmode= in the URL; add ssl + connect timeout explicitly."""
    raw_timeout = (os.getenv("DATABASE_CONNECT_TIMEOUT") or "15").strip()
    try:
        timeout = float(raw_timeout)
    except ValueError:
        timeout = 15.0
    args: dict[str, Any] = {"timeout": timeout}
    ul = url.lower()
    if (
        "sslmode=require" in ul
        or "sslmode=verify-full" in ul
        or "sslmode=verify-ca" in ul
        or "ssl=true" in ul
        or os.getenv("DATABASE_SSL_REQUIRE", "").lower() in ("1", "true", "yes")
    ):
        args["ssl"] = True
    return args


def get_engine():
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        url = get_database_url()
        echo = os.getenv("DATABASE_ECHO", "false").lower() == "true"
        if _is_sqlite():
            # SQLite does not support connection pooling parameters
            _engine = create_async_engine(
                url,
                echo=echo,
                connect_args={
                    "check_same_thread": False,
                    # Avoid indefinite waits when init_db and create_session overlap (busy handler)
                    "timeout": float(os.getenv("SQLITE_BUSY_TIMEOUT_SEC", "30")),
                },
            )
        else:
            extra: dict[str, Any] = {
                "echo": echo,
                "pool_pre_ping": True,
                "pool_size": 5,
                "max_overflow": 10,
            }
            if "+asyncpg" in url:
                extra["connect_args"] = _asyncpg_connect_args(url)
            _engine = create_async_engine(url, **extra)
    return _engine


def get_session_maker():
    """Get or create the async session maker."""
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_maker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _apply_session_persistence_schema_sync(connection) -> None:
    """Add workspace / conversation columns on existing DBs (SQLite + Postgres)."""
    from sqlalchemy import text

    dialect = connection.dialect.name
    if dialect == "sqlite":
        r = connection.execute(text("PRAGMA table_info(sessions)"))
        cols = {row[1] for row in r.fetchall()}
        if "workspace_id" not in cols:
            connection.execute(
                text("ALTER TABLE sessions ADD COLUMN workspace_id VARCHAR(64)")
            )
            connection.execute(
                text("CREATE UNIQUE INDEX IF NOT EXISTS ix_sessions_workspace_id "
                     "ON sessions (workspace_id) WHERE workspace_id IS NOT NULL")
            )
        if "conversation_json" not in cols:
            connection.execute(
                text("ALTER TABLE sessions ADD COLUMN conversation_json TEXT")
            )
        return

    # PostgreSQL (and other servers): IF NOT EXISTS for columns
    connection.execute(
        text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS workspace_id VARCHAR(64)")
    )
    connection.execute(
        text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS conversation_json TEXT")
    )
    connection.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_sessions_workspace_id "
            "ON sessions (workspace_id) WHERE workspace_id IS NOT NULL"
        )
    )


async def init_db():
    """Initialize the database by creating all tables."""
    from .models import Base as ModelsBase
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
        await conn.run_sync(_apply_session_persistence_schema_sync)


async def close_db():
    """Close the database connection."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
