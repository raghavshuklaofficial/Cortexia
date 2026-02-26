"""
Health, readiness, and system info routes.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from sqlalchemy import func, select, text

from cortexia import __version__
from cortexia.api.deps import DbSession, get_pipeline
from cortexia.api.schemas.models import HealthResponse, ReadinessResponse, SystemInfo
from cortexia.config import get_settings
from cortexia.db.models import FaceEmbedding, Identity

router = APIRouter(tags=["System"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe — is the server running?"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(
    db: DbSession = None,  # type: ignore[assignment]
):
    """Readiness probe — are all dependencies available?"""
    # Check database
    db_status = "unavailable"
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        pass

    # Check Redis
    redis_status = "unavailable"
    try:
        import redis as redis_lib

        settings = get_settings()
        r = redis_lib.from_url(settings.redis_url, socket_timeout=2)
        r.ping()
        redis_status = "connected"
    except Exception:
        pass

    # Check models
    try:
        pipeline = get_pipeline()
        models_loaded = pipeline._initialized
    except Exception:
        models_loaded = False

    overall = "ready" if db_status == "connected" and models_loaded else "not_ready"

    return ReadinessResponse(
        status=overall,
        database=db_status,
        redis=redis_status,
        models_loaded=models_loaded,
    )


@router.get("/system/info", response_model=SystemInfo)
async def system_info(
    db: DbSession = None,  # type: ignore[assignment]
):
    """System configuration and capabilities."""
    settings = get_settings()

    # Count identities and embeddings
    id_count = await db.scalar(
        select(func.count()).select_from(Identity).where(Identity.is_active.is_(True))
    ) or 0
    emb_count = await db.scalar(
        select(func.count()).select_from(FaceEmbedding)
    ) or 0

    # Check GPU
    gpu_available = False
    try:
        import onnxruntime

        gpu_available = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        pass

    return SystemInfo(
        version=__version__,
        detection_backend=settings.model_backend,
        embedding_dim=settings.embedding_dim,
        gpu_available=gpu_available,
        trust_pipeline_enabled=settings.trust_pipeline_enabled,
        antispoof_enabled=settings.antispoof_enabled,
        attributes_enabled=settings.attributes_enabled,
        total_identities=id_count,
        total_embeddings=emb_count,
    )
