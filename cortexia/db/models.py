"""
SQLAlchemy ORM models.

Tables:
  - identities: enrolled people
  - face_embeddings: 512-d vectors (pgvector)
  - recognition_events: immutable audit log
  - clusters: auto-discovered groupings
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Identity(Base):
    """An enrolled person."""

    __tablename__ = "identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=dict)
    privacy_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    embeddings: Mapped[list[FaceEmbedding]] = relationship(
        "FaceEmbedding", back_populates="identity", cascade="all, delete-orphan"
    )
    recognition_events: Mapped[list[RecognitionEvent]] = relationship(
        "RecognitionEvent", back_populates="identity"
    )

    def __repr__(self) -> str:
        return f"<Identity(id={self.id}, name={self.name!r})>"


class FaceEmbedding(Base):
    """512-d face vector stored via pgvector. Each identity can have multiple."""

    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("identities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    source_image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    identity: Mapped[Identity] = relationship("Identity", back_populates="embeddings")

    # Index for fast approximate nearest neighbor search
    __table_args__ = (
        Index(
            "ix_face_embeddings_vector",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<FaceEmbedding(id={self.id}, identity_id={self.identity_id})>"


class RecognitionEvent(Base):
    """Immutable audit log entry for a face recognition event.

    Every recognition (whether matched or unknown) is logged here
    for forensic compliance and analytics.

    Attributes:
        confidence: Platt-calibrated recognition confidence
        trust_score: Composite trust pipeline score
        is_spoof: Whether liveness check flagged as spoof
        source: Where the face came from (upload, webcam, stream)
        attributes_json: Predicted age/gender/emotion
        bounding_box_json: Face location in the frame
    """

    __tablename__ = "recognition_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("identities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    trust_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_spoof: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_known: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    source: Mapped[str] = mapped_column(String(50), default="upload", index=True)
    identity_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    attributes_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    bounding_box_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    frame_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Relationships
    identity: Mapped[Identity | None] = relationship(
        "Identity", back_populates="recognition_events"
    )

    # Composite index for analytics queries
    __table_args__ = (
        Index("ix_recognition_events_time_source", "timestamp", "source"),
    )

    def __repr__(self) -> str:
        return (
            f"<RecognitionEvent(id={self.id}, identity={self.identity_name!r}, "
            f"confidence={self.confidence:.3f})>"
        )


class Cluster(Base):
    """Auto-discovered identity cluster from HDBSCAN.

    Represents a group of face embeddings that likely belong to the
    same unknown person. Can be manually merged into a known identity.

    Attributes:
        centroid_embedding: Average embedding of cluster members
        member_count: Number of faces in this cluster
        merged_into_identity_id: If assigned, which identity this became
    """

    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    centroid_embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, default=0)
    merged_into_identity_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("identities.id", ondelete="SET NULL"), nullable=True
    )
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    members: Mapped[list[ClusterMember]] = relationship(
        "ClusterMember", back_populates="cluster", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Cluster(id={self.id}, members={self.member_count})>"


class ClusterMember(Base):
    """Maps face embeddings to their discovered cluster."""

    __tablename__ = "cluster_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clusters.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding_vector: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    source_event_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("recognition_events.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    cluster: Mapped[Cluster] = relationship(
        "Cluster", back_populates="members"
    )
    source_event: Mapped[RecognitionEvent | None] = relationship(
        "RecognitionEvent"
    )
