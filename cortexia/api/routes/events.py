"""Recognition event listing (audit trail)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from cortexia.api.deps import DbSession, ApiKey
from cortexia.api.schemas.models import (
    EventListResponse,
    PaginationMeta,
    RecognitionEventResponse,
)
from cortexia.db.repositories.event_repo import EventRepository

router = APIRouter(prefix="/events", tags=["Events"])


@router.get("", response_model=EventListResponse)
async def list_events(
    skip: int = 0,
    limit: int = 50,
    identity_id: int | None = None,
    source: str | None = None,
    is_known: bool | None = None,
    is_spoof: bool | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """List recognition events with filtering and pagination.

    This is the forensic audit trail — every face recognition action
    is logged here immutably.
    """
    repo = EventRepository(db)
    events, total = await repo.get_events(
        skip=skip,
        limit=limit,
        identity_id=identity_id,
        source=source,
        is_known=is_known,
        is_spoof=is_spoof,
        since=since,
        until=until,
    )

    items = [
        RecognitionEventResponse(
            id=e.id,
            identity_id=e.identity_id,
            identity_name=e.identity_name,
            timestamp=e.timestamp,
            confidence=e.confidence,
            trust_score=e.trust_score,
            is_spoof=e.is_spoof,
            is_known=e.is_known,
            source=e.source,
            attributes=e.attributes_json,
            bounding_box=e.bounding_box_json,
        )
        for e in events
    ]

    return EventListResponse(
        events=items,
        pagination=PaginationMeta(
            total=total, skip=skip, limit=limit, has_more=(skip + limit < total)
        ),
    )
