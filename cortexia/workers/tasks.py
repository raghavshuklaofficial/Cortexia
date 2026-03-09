"""
Celery background tasks: batch recognition, re-clustering,
data cleanup, gallery warming.
"""

from __future__ import annotations

import asyncio
import base64
import io
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import structlog
from PIL import Image

from cortexia.workers import celery_app

logger = structlog.get_logger(__name__)

# Worker-level singleton to avoid re-loading ML models per task
_pipeline = None


def _get_pipeline():
    """Get or create the worker-level pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from cortexia.config import get_settings
        from cortexia.core.trust_pipeline import PipelineConfig, TrustPipeline

        settings = get_settings()
        _pipeline = TrustPipeline(PipelineConfig.from_settings(settings))
        _pipeline.initialize()
    return _pipeline


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@celery_app.task(bind=True, name="cortexia.batch_recognize", max_retries=2)
def batch_recognize(self, job_id: str, image_payloads: list[dict]) -> dict:
    """
    Process a batch of images for recognition.

    Each payload: {"image_b64": str, "source": str, "metadata": dict}
    Returns aggregated results keyed by index.
    """
    logger.info("batch_recognize_start", job_id=job_id, count=len(image_payloads))

    try:
        results = _run_async(_process_batch(job_id, image_payloads))
        logger.info(
            "batch_recognize_complete",
            job_id=job_id,
            processed=len(results),
        )
        return {"job_id": job_id, "status": "completed", "results": results}
    except Exception as exc:
        logger.error("batch_recognize_failed", job_id=job_id, error=str(exc))
        raise self.retry(exc=exc, countdown=10)


async def _process_batch(job_id: str, payloads: list[dict]) -> list[dict]:
    """Internal async batch processing."""
    from cortexia.db.session import async_session_factory, init_db

    await init_db()

    pipeline = _get_pipeline()

    results = []
    async with async_session_factory() as session:
        for idx, payload in enumerate(payloads):
            try:
                image_bytes = base64.b64decode(payload["image_b64"])
                image = Image.open(io.BytesIO(image_bytes))
                frame = np.array(image)

                if len(frame.shape) == 2:
                    import cv2

                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    import cv2

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    frame = frame[:, :, ::-1]  # RGB → BGR

                analysis = pipeline.process_image(frame)
                source = payload.get("source", f"batch_{job_id}")

                face_results = []
                for face in analysis.faces:
                    face_data = {
                        "bbox": face.face.bbox.to_dict(),
                        "trust_score": face.trust_score,
                    }

                    if face.liveness:
                        face_data["liveness"] = {
                            "verdict": face.liveness.verdict.value,
                            "confidence": face.liveness.confidence,
                        }

                    if face.recognition:
                        face_data["recognition"] = {
                            "identity_id": face.recognition.identity_id,
                            "identity_name": face.recognition.identity_name,
                            "confidence": face.recognition.confidence,
                        }

                    if face.attributes:
                        face_data["attributes"] = {
                            "age": face.attributes.age,
                            "gender": face.attributes.gender,
                            "emotion": face.attributes.emotion.value
                            if face.attributes.emotion
                            else None,
                        }

                    face_results.append(face_data)

                results.append(
                    {
                        "index": idx,
                        "status": "success",
                        "faces_detected": analysis.face_count,
                        "faces": face_results,
                        "processing_time_ms": analysis.total_processing_time_ms,
                    }
                )

            except Exception as e:
                logger.warning(
                    "batch_item_failed", index=idx, error=str(e), job_id=job_id
                )
                results.append(
                    {"index": idx, "status": "error", "error": str(e)}
                )

    return results


@celery_app.task(name="cortexia.recluster_identities")
def recluster_identities(min_cluster_size: int = 5) -> dict:
    """
    Run HDBSCAN clustering on all stored embeddings to discover
    unknown identity groupings.
    """
    logger.info("recluster_start", min_cluster_size=min_cluster_size)

    try:
        result = _run_async(_run_clustering(min_cluster_size))
        logger.info("recluster_complete", **result)
        return result
    except Exception as exc:
        logger.error("recluster_failed", error=str(exc))
        raise


async def _run_clustering(min_cluster_size: int) -> dict:
    """Internal async clustering logic."""
    from cortexia.core.clusterer import IdentityClusterer
    from cortexia.db.repositories.vector_repo import VectorRepository
    from cortexia.db.session import async_session_factory, init_db

    await init_db()

    async with async_session_factory() as session:
        repo = VectorRepository(session)
        embeddings_data = await repo.get_all_embeddings()

        if len(embeddings_data) < min_cluster_size:
            return {
                "status": "skipped",
                "reason": "insufficient_embeddings",
                "count": len(embeddings_data),
            }

        embeddings = np.array([e[1] for e in embeddings_data])

        clusterer = IdentityClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=max(2, min_cluster_size // 2),
        )
        result = clusterer.cluster(embeddings)

        return {
            "status": "completed",
            "total_embeddings": len(embeddings),
            "clusters_found": result.cluster_count,
            "noise_points": result.noise_count,
        }


@celery_app.task(name="cortexia.cleanup_old_events")
def cleanup_old_events(retention_days: int = 90) -> dict:
    """
    Remove recognition events older than the retention period.
    Supports GDPR data minimization requirements.
    """
    logger.info("cleanup_start", retention_days=retention_days)

    try:
        result = _run_async(_run_cleanup(retention_days))
        logger.info("cleanup_complete", **result)
        return result
    except Exception as exc:
        logger.error("cleanup_failed", error=str(exc))
        raise


async def _run_cleanup(retention_days: int) -> dict:
    """Internal async cleanup."""
    from sqlalchemy import delete

    from cortexia.db.models import RecognitionEvent
    from cortexia.db.session import async_session_factory, init_db

    await init_db()

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    async with async_session_factory() as session:
        stmt = delete(RecognitionEvent).where(
            RecognitionEvent.timestamp < cutoff
        )
        result = await session.execute(stmt)
        await session.commit()
        deleted = result.rowcount

    return {
        "status": "completed",
        "cutoff_date": cutoff.isoformat(),
        "events_deleted": deleted,
    }


@celery_app.task(name="cortexia.warm_gallery_cache")
def warm_gallery_cache() -> dict:
    """
    Pre-load the identity gallery into the recognizer cache.
    Useful after deployment or identity changes.
    """
    logger.info("gallery_warm_start")

    try:
        result = _run_async(_warm_gallery())
        logger.info("gallery_warm_complete", **result)
        return result
    except Exception as exc:
        logger.error("gallery_warm_failed", error=str(exc))
        raise


async def _warm_gallery() -> dict:
    """Load all enrolled identities into the recognizer gallery."""
    from cortexia.core.recognizer import StoredIdentity
    from cortexia.db.repositories.identity_repo import IdentityRepository
    from cortexia.db.session import async_session_factory, init_db

    await init_db()

    pipeline = _get_pipeline()

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

        if pipeline.recognizer is not None:
            pipeline.recognizer.load_gallery(gallery)

        total_embeddings = sum(len(si.embeddings) for si in gallery)

    return {
        "status": "completed",
        "identities_loaded": len(gallery),
        "total_embeddings": total_embeddings,
    }


# Celery beat periodic schedule
celery_app.conf.beat_schedule = {
    "recluster-every-6h": {
        "task": "cortexia.recluster_identities",
        "schedule": 21600.0,  # 6 hours
        "args": (5,),
    },
    "cleanup-daily": {
        "task": "cortexia.cleanup_old_events",
        "schedule": 86400.0,  # 24 hours
        "args": (90,),
    },
    "warm-gallery-hourly": {
        "task": "cortexia.warm_gallery_cache",
        "schedule": 3600.0,  # 1 hour
    },
}
