"""Vector search via pgvector."""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from cortexia.db.models import FaceEmbedding, Identity


class VectorRepository:
    """Handles pgvector similarity search."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_embedding(
        self,
        identity_id: int,
        embedding: list[float],
        source_image_hash: str | None = None,
    ) -> FaceEmbedding:
        """Store a face embedding for an identity.

        Args:
            identity_id: ID of the identity this embedding belongs to
            embedding: 512-d float vector
            source_image_hash: SHA256 hash of source image for dedup

        Returns:
            Created FaceEmbedding record
        """
        face_emb = FaceEmbedding(
            identity_id=identity_id,
            embedding=embedding,
            source_image_hash=source_image_hash,
        )
        self._session.add(face_emb)
        await self._session.flush()
        return face_emb

    async def find_nearest(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[dict]:
        """Find the nearest identity embeddings to a query vector.

        Uses pgvector's cosine distance operator (<=>).

        Args:
            query_embedding: 512-d query vector
            top_k: Number of nearest results to return
            threshold: Maximum cosine distance (optional filter)

        Returns:
            List of dicts with identity_id, identity_name, distance, embedding_id
        """
        # pgvector cosine distance: 1 - cosine_similarity
        # Lower = more similar
        # NOTE: should probably switch to HNSW index when gallery gets large
        query_str = (
            "SELECT fe.id, fe.identity_id, i.name, "
            "fe.embedding <=> :query_vec AS distance "
            "FROM face_embeddings fe "
            "JOIN identities i ON fe.identity_id = i.id "
            "WHERE i.is_active = true "
        )

        if threshold is not None:
            query_str += "AND fe.embedding <=> :query_vec < :threshold "

        query_str += "ORDER BY fe.embedding <=> :query_vec LIMIT :limit"

        # pgvector expects '[x,y,z,...]' format
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        params: dict = {"query_vec": vec_str, "limit": top_k}
        if threshold is not None:
            params["threshold"] = threshold

        result = await self._session.execute(
            text(query_str),
            params,
        )

        matches = []
        for row in result:
            matches.append(
                {
                    "embedding_id": row[0],
                    "identity_id": row[1],
                    "identity_name": row[2],
                    "distance": float(row[3]),
                    "similarity": 1.0 - float(row[3]),
                }
            )
        return matches

    async def get_embeddings_for_identity(
        self, identity_id: int
    ) -> list[FaceEmbedding]:
        """Get all embeddings for an identity."""
        stmt = (
            select(FaceEmbedding)
            .where(FaceEmbedding.identity_id == identity_id)
            .order_by(FaceEmbedding.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete_embeddings_for_identity(self, identity_id: int) -> int:
        """Delete all embeddings for an identity. Returns count deleted."""
        from sqlalchemy import delete as sa_delete

        stmt = sa_delete(FaceEmbedding).where(
            FaceEmbedding.identity_id == identity_id
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount

    async def get_all_embeddings(self) -> list[tuple[int, list[float]]]:
        """Get all embeddings as (identity_id, vector) tuples for gallery loading."""
        stmt = (
            select(FaceEmbedding.identity_id, FaceEmbedding.embedding)
            .join(Identity)
            .where(Identity.is_active.is_(True))
        )
        result = await self._session.execute(stmt)
        return [(row[0], row[1]) for row in result]

    async def check_duplicate(
        self,
        embedding: list[float],
        threshold: float = 0.15,
    ) -> dict | None:
        """Check if a very similar embedding already exists (dedup).

        Args:
            embedding: Query embedding
            threshold: Maximum cosine distance to consider duplicate

        Returns:
            Matching identity info if duplicate found, None otherwise
        """
        matches = await self.find_nearest(embedding, top_k=1, threshold=threshold)
        return matches[0] if matches else None
