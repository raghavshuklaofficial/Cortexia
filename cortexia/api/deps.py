"""
FastAPI dependency injection providers.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import Depends, HTTPException, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from cortexia.config import Settings, get_settings
from cortexia.db.session import get_session


async def get_db(
    session: AsyncSession = Depends(get_session),
) -> AsyncSession:
    """Provide a database session."""
    return session


async def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify API key from X-API-Key header.

    API key is always required. For local development,
    set API_KEY=dev in .env and send X-API-Key: dev.
    """
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )
    return x_api_key


def get_pipeline():
    """Get the Trust Pipeline singleton from app state.

    The pipeline is initialized during app lifespan and stored
    in app.state.pipeline.
    """
    from cortexia.main import app_state

    if app_state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Trust Pipeline not initialized. Server is starting up.",
        )
    return app_state.pipeline


# Type aliases for common dependencies
DbSession = Annotated[AsyncSession, Depends(get_db)]
ApiKey = Annotated[str, Depends(verify_api_key)]
