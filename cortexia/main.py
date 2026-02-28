"""
CORTEXIA — FastAPI Application Factory.

This is the main entry point for the CORTEXIA API server.
It configures:
  - Lifespan handler (load ML models on startup, release on shutdown)
  - All API routes under /api/v1
  - CORS, authentication, and request logging middleware
  - Automatic OpenAPI documentation at /docs and /redoc
"""

from __future__ import annotations

import contextlib
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cortexia import __version__
from cortexia.config import get_settings
from cortexia.core.trust_pipeline import PipelineConfig, TrustPipeline

logger = structlog.get_logger(__name__)


@dataclass
class AppState:
    """Mutable application state container."""

    pipeline: TrustPipeline | None = None
    start_time: float = 0.0


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    On startup: initialize database, load ML models.
    On shutdown: release resources.
    """
    settings = get_settings()
    app_state.start_time = time.time()

    logger.info(
        "cortexia_starting",
        version=__version__,
        env=settings.app_env,
        backend=settings.model_backend,
    )

    # Initialize database
    from cortexia.db.session import close_db, init_db

    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.warning("database_init_failed", error=str(e))

    # Initialize Trust Pipeline
    config = PipelineConfig.from_settings(settings)
    pipeline = TrustPipeline(config)

    try:
        pipeline.initialize()
        app_state.pipeline = pipeline
        logger.info("trust_pipeline_ready")
    except Exception as e:
        logger.error("pipeline_init_failed", error=str(e))
        # Create pipeline but don't initialize — will retry on first request
        app_state.pipeline = pipeline

    # Load recognition gallery from database
    try:
        import numpy as np

        from cortexia.core.recognizer import StoredIdentity
        from cortexia.db.repositories.identity_repo import IdentityRepository
        from cortexia.db.session import async_session_factory

        async with async_session_factory() as session:
            repo = IdentityRepository(session)
            identities = await repo.get_all_with_embeddings()

            gallery = []
            for ident in identities:
                if not ident.embeddings:
                    continue
                gallery.append(
                    StoredIdentity(
                        identity_id=ident.id,
                        name=ident.name,
                        embeddings=[
                            np.array(e.embedding, dtype=np.float32)
                            for e in ident.embeddings
                        ],
                    )
                )

            if gallery and pipeline.recognizer:
                pipeline.recognizer.load_gallery(gallery)
                logger.info("gallery_loaded", identities=len(gallery))
    except Exception as e:
        logger.warning("gallery_load_failed", error=str(e))

    logger.info("cortexia_ready", uptime_ms=round((time.time() - app_state.start_time) * 1000))

    yield  # ← Application runs here

    # Shutdown
    logger.info("cortexia_shutting_down")
    with contextlib.suppress(Exception):
        await close_db()
    app_state.pipeline = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    # Refuse to start in production without required secrets
    if settings.app_env == "production":
        if not settings.api_key:
            raise RuntimeError("API_KEY must be set in production mode")
        if not settings.secret_key:
            raise RuntimeError("SECRET_KEY must be set in production mode")

    from cortexia.utils.logging import setup_logging

    setup_logging(
        log_level=settings.log_level,
        json_output=settings.app_env == "production",
    )

    is_prod = settings.app_env == "production"

    app = FastAPI(
        title="CORTEXIA",
        description=(
            "Neural Face Intelligence Platform — "
            "Trust Pipeline · Zero-Shot Discovery · Forensic Audit Trail"
        ),
        version=__version__,
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
        lifespan=lifespan,
    )

    # ───── CORS ─────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # ───── Request Logging Middleware ─────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not request.url.path.startswith("/health"):
            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                elapsed_ms=round(elapsed_ms, 2),
            )
        return response

    # ───── Exception Handler ─────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "detail": str(exc) if settings.debug else "An unexpected error occurred.",
            },
        )

    # ───── Register Routes ─────
    from cortexia.api.routes import (
        analytics,
        clusters,
        events,
        forensics,
        health,
        identities,
        recognize,
        streams,
    )

    # Health routes at root level (for Docker healthcheck)
    app.include_router(health.router)

    # API v1 routes
    api_prefix = "/api/v1"
    # Also register health under /api/v1 so dashboard can reach them
    app.include_router(health.router, prefix=api_prefix, include_in_schema=False)
    app.include_router(identities.router, prefix=api_prefix)
    app.include_router(recognize.router, prefix=api_prefix)
    app.include_router(streams.router, prefix=api_prefix)
    app.include_router(forensics.router, prefix=api_prefix)
    app.include_router(clusters.router, prefix=api_prefix)
    app.include_router(analytics.router, prefix=api_prefix)
    app.include_router(events.router, prefix=api_prefix)

    return app


# Application instance
app = create_app()
