"""Event repository — audit trail queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func, and_, true as sa_true
from sqlalchemy.ext.asyncio import AsyncSession

from cortexia.db.models import RecognitionEvent


class EventRepository:
    """Data access for recognition event audit logs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log_event(
        self,
        identity_id: int | None,
        identity_name: str | None,
        confidence: float,
        trust_score: float,
        is_spoof: bool,
        is_known: bool,
        source: str = "upload",
        attributes_json: dict | None = None,
        bounding_box_json: dict | None = None,
        frame_hash: str | None = None,
    ) -> RecognitionEvent:
        """Log a recognition event to the audit trail."""
        event = RecognitionEvent(
            identity_id=identity_id if identity_id and identity_id > 0 else None,
            identity_name=identity_name,
            confidence=confidence,
            trust_score=trust_score,
            is_spoof=is_spoof,
            is_known=is_known,
            source=source,
            attributes_json=attributes_json,
            bounding_box_json=bounding_box_json,
            frame_hash=frame_hash,
        )
        self._session.add(event)
        await self._session.flush()
        return event

    async def get_events(
        self,
        skip: int = 0,
        limit: int = 50,
        identity_id: int | None = None,
        source: str | None = None,
        is_known: bool | None = None,
        is_spoof: bool | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[list[RecognitionEvent], int]:
        """Query recognition events with filtering."""
        conditions = []

        if identity_id is not None:
            conditions.append(RecognitionEvent.identity_id == identity_id)
        if source is not None:
            conditions.append(RecognitionEvent.source == source)
        if is_known is not None:
            conditions.append(RecognitionEvent.is_known == is_known)
        if is_spoof is not None:
            conditions.append(RecognitionEvent.is_spoof == is_spoof)
        if since is not None:
            conditions.append(RecognitionEvent.timestamp >= since)
        if until is not None:
            conditions.append(RecognitionEvent.timestamp <= until)

        where_clause = and_(*conditions) if conditions else sa_true()

        # Count
        count_stmt = (
            select(func.count())
            .select_from(RecognitionEvent)
            .where(where_clause)
        )
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar() or 0

        # Fetch
        stmt = (
            select(RecognitionEvent)
            .where(where_clause)
            .order_by(RecognitionEvent.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        events = list(result.scalars().all())

        return events, total

    async def get_overview_stats(self) -> dict:
        """Get aggregate statistics for the analytics dashboard."""
        total_events = await self._session.scalar(
            select(func.count()).select_from(RecognitionEvent)
        )
        known_events = await self._session.scalar(
            select(func.count())
            .select_from(RecognitionEvent)
            .where(RecognitionEvent.is_known.is_(True))
        )
        spoof_events = await self._session.scalar(
            select(func.count())
            .select_from(RecognitionEvent)
            .where(RecognitionEvent.is_spoof.is_(True))
        )
        avg_trust = await self._session.scalar(
            select(func.avg(RecognitionEvent.trust_score))
        )
        avg_confidence = await self._session.scalar(
            select(func.avg(RecognitionEvent.confidence))
            .where(RecognitionEvent.is_known.is_(True))
        )

        return {
            "total_events": total_events or 0,
            "known_events": known_events or 0,
            "unknown_events": (total_events or 0) - (known_events or 0),
            "spoof_events": spoof_events or 0,
            "unknown_ratio": (
                round(1 - (known_events or 0) / total_events, 4)
                if total_events
                else 0.0
            ),
            "avg_trust_score": round(avg_trust or 0.0, 4),
            "avg_recognition_confidence": round(avg_confidence or 0.0, 4),
        }

    async def get_timeline(
        self,
        interval: str = "hour",
        days: int = 7,
    ) -> list[dict]:
        """Get recognition event counts grouped by time interval."""
        since = datetime.now(timezone.utc) - timedelta(days=days)

        if interval == "hour":
            trunc_func = func.date_trunc("hour", RecognitionEvent.timestamp)
        elif interval == "day":
            trunc_func = func.date_trunc("day", RecognitionEvent.timestamp)
        else:
            trunc_func = func.date_trunc("hour", RecognitionEvent.timestamp)

        stmt = (
            select(
                trunc_func.label("period"),
                func.count().label("total"),
                func.count()
                .filter(RecognitionEvent.is_known.is_(True))
                .label("known"),
                func.count()
                .filter(RecognitionEvent.is_spoof.is_(True))
                .label("spoofs"),
            )
            .where(RecognitionEvent.timestamp >= since)
            .group_by("period")
            .order_by("period")
        )

        result = await self._session.execute(stmt)
        return [
            {
                "period": str(row.period),
                "total": row.total,
                "known": row.known,
                "spoofs": row.spoofs,
            }
            for row in result
        ]

    async def get_demographics(self) -> dict:
        """Get aggregated demographics from recognition events."""
        # This queries the JSONB attributes for aggregated stats
        stmt = select(RecognitionEvent.attributes_json).where(
            RecognitionEvent.attributes_json.isnot(None)
        ).limit(1000)
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        ages: list[int] = []
        genders: dict[str, int] = {}
        emotions: dict[str, int] = {}

        for attrs in rows:
            if not isinstance(attrs, dict):
                continue
            if attrs.get("age"):
                ages.append(attrs["age"])
            if attrs.get("gender"):
                g = attrs["gender"]
                genders[g] = genders.get(g, 0) + 1
            if attrs.get("emotion"):
                e = attrs["emotion"]
                emotions[e] = emotions.get(e, 0) + 1

        return {
            "age_distribution": {
                "mean": round(sum(ages) / len(ages), 1) if ages else None,
                "min": min(ages) if ages else None,
                "max": max(ages) if ages else None,
                "count": len(ages),
            },
            "gender_distribution": genders,
            "emotion_distribution": emotions,
        }
