"""
Async database session management for CORTEXIA.

Uses SQLAlchemy 2.0 async engine with asyncpg driver for PostgreSQL.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from cortexia.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    pool_size=_settings.database_pool_size,
    max_overflow=_settings.database_max_overflow,
    echo=_settings.debug,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session (FastAPI dependency)."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database — create pgvector extension and tables."""
    from sqlalchemy import text

    from cortexia.db.models import Base

    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()


def get_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create a standalone session factory for a given database URL.

    Useful for CLI commands or scripts that don't use the default engine.
    """
    eng = create_async_engine(database_url, pool_pre_ping=True)
    return async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
