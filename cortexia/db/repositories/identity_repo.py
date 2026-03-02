"""
Identity repository — CRUD operations for enrolled identities.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cortexia.db.models import Identity, FaceEmbedding, RecognitionEvent


class IdentityRepository:
    """Data access layer for Identity entities."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        name: str,
        metadata: dict | None = None,
    ) -> Identity:
        """Create a new identity."""
        identity = Identity(
            name=name,
            metadata_json=metadata or {},
        )
        self._session.add(identity)
        await self._session.flush()
        return identity

    async def get_by_id(self, identity_id: int) -> Identity | None:
        """Get an identity by ID with embeddings loaded."""
        stmt = (
            select(Identity)
            .where(Identity.id == identity_id, Identity.is_active.is_(True))
            .options(selectinload(Identity.embeddings))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 50,
        search: str | None = None,
    ) -> tuple[list[Identity], int]:
        """Get all active identities with pagination.

        Returns:
            Tuple of (identities, total_count)
        """
        base_query = select(Identity).where(Identity.is_active.is_(True))

        if search:
            base_query = base_query.where(Identity.name.ilike(f"%{search}%"))

        # Count
        count_stmt = select(func.count()).select_from(base_query.subquery())
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar() or 0

        # Fetch
        stmt = (
            base_query
            .options(selectinload(Identity.embeddings))
            .order_by(Identity.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        identities = list(result.scalars().all())

        return identities, total

    async def update(
        self,
        identity_id: int,
        name: str | None = None,
        metadata: dict | None = None,
    ) -> Identity | None:
        """Update an identity."""
        identity = await self.get_by_id(identity_id)
        if identity is None:
            return None

        if name is not None:
            identity.name = name
        if metadata is not None:
            identity.metadata_json = metadata

        await self._session.flush()
        return identity

    async def soft_delete(self, identity_id: int) -> bool:
        """Soft delete an identity."""
        identity = await self.get_by_id(identity_id)
        if identity is None:
            return False
        identity.is_active = False
        await self._session.flush()
        return True

    async def hard_delete(self, identity_id: int) -> bool:
        """Hard delete an identity and all associated data (GDPR compliance)."""
        # Query by ID directly (without is_active filter) so soft-deleted
        # identities can still be permanently removed.
        stmt = (
            select(Identity)
            .where(Identity.id == identity_id)
            .options(selectinload(Identity.embeddings))
        )
        result = await self._session.execute(stmt)
        identity = result.scalar_one_or_none()
        if identity is None:
            return False

        # Delete all recognition events
        await self._session.execute(
            delete(RecognitionEvent).where(RecognitionEvent.identity_id == identity_id)
        )
        # Delete identity (cascades to embeddings)
        await self._session.delete(identity)
        await self._session.flush()
        return True

    async def get_last_seen(self, identity_id: int) -> datetime | None:
        """Get the most recent recognition event timestamp for an identity."""
        stmt = (
            select(RecognitionEvent.timestamp)
            .where(RecognitionEvent.identity_id == identity_id)
            .order_by(RecognitionEvent.timestamp.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_with_embeddings(self) -> list[Identity]:
        """Load all active identities with their embeddings for gallery."""
        stmt = (
            select(Identity)
            .where(Identity.is_active.is_(True))
            .options(selectinload(Identity.embeddings))
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
