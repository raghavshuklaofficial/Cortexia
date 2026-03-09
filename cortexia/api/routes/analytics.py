"""Analytics routes for the dashboard."""

from __future__ import annotations

from fastapi import APIRouter

from cortexia.api.deps import DbSession, ApiKey
from cortexia.api.schemas.models import (
    DemographicsResponse,
    OverviewStats,
    TimelinePoint,
)
from cortexia.db.repositories.event_repo import EventRepository
from cortexia.db.repositories.identity_repo import IdentityRepository

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview", response_model=OverviewStats)
async def get_overview(
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Get dashboard overview statistics.

    Returns aggregate metrics across all recognition events,
    including total identities, unknown ratio, and average trust scores.
    """
    identity_repo = IdentityRepository(db)
    event_repo = EventRepository(db)

    _, total_identities = await identity_repo.get_all(limit=0)
    stats = await event_repo.get_overview_stats()

    return OverviewStats(
        total_identities=total_identities,
        total_events=stats["total_events"],
        known_events=stats["known_events"],
        unknown_events=stats["unknown_events"],
        spoof_events=stats["spoof_events"],
        unknown_ratio=stats["unknown_ratio"],
        avg_trust_score=stats["avg_trust_score"],
        avg_recognition_confidence=stats["avg_recognition_confidence"],
    )


@router.get("/timeline", response_model=list[TimelinePoint])
async def get_timeline(
    interval: str = "hour",
    days: int = 7,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Get recognition events over time.

    Args:
        interval: "hour" or "day"
        days: Number of days to look back
    """
    event_repo = EventRepository(db)
    timeline = await event_repo.get_timeline(interval=interval, days=days)

    return [
        TimelinePoint(
            period=point["period"],
            total=point["total"],
            known=point["known"],
            spoofs=point["spoofs"],
        )
        for point in timeline
    ]


@router.get("/demographics", response_model=DemographicsResponse)
async def get_demographics(
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Get aggregated demographics from recognition events.

    Returns distributions of age, gender, and emotion across
    all processed faces.
    """
    event_repo = EventRepository(db)
    demographics = await event_repo.get_demographics()

    return DemographicsResponse(
        age_distribution=demographics["age_distribution"],
        gender_distribution=demographics["gender_distribution"],
        emotion_distribution=demographics["emotion_distribution"],
    )
