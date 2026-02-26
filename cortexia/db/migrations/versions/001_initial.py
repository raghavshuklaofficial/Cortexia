"""initial schema with pgvector

Revision ID: 001_initial
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Identities table
    op.create_table(
        "identities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("metadata_json", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("privacy_score", sa.Float(), server_default="0.0"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_identities_name", "identities", ["name"])
    op.create_index("ix_identities_is_active", "identities", ["is_active"])

    # Face embeddings table with pgvector
    op.create_table(
        "face_embeddings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("identity_id", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.Column("source_image_hash", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["identity_id"], ["identities.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_face_embeddings_identity_id", "face_embeddings", ["identity_id"])

    # IVFFlat index for approximate nearest neighbor search
    op.execute(
        "CREATE INDEX ix_face_embeddings_vector ON face_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # Recognition events table
    op.create_table(
        "recognition_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("identity_id", sa.Integer(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("confidence", sa.Float(), server_default="0.0"),
        sa.Column("trust_score", sa.Float(), server_default="0.0"),
        sa.Column("is_spoof", sa.Boolean(), server_default="false"),
        sa.Column("is_known", sa.Boolean(), server_default="false"),
        sa.Column("source", sa.String(50), server_default="'upload'"),
        sa.Column("identity_name", sa.String(255), nullable=True),
        sa.Column("attributes_json", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("bounding_box_json", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("frame_hash", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["identity_id"], ["identities.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_recognition_events_identity_id", "recognition_events", ["identity_id"])
    op.create_index("ix_recognition_events_timestamp", "recognition_events", ["timestamp"])
    op.create_index("ix_recognition_events_is_spoof", "recognition_events", ["is_spoof"])
    op.create_index("ix_recognition_events_is_known", "recognition_events", ["is_known"])
    op.create_index("ix_recognition_events_source", "recognition_events", ["source"])
    op.create_index(
        "ix_recognition_events_time_source",
        "recognition_events",
        ["timestamp", "source"],
    )

    # Clusters table
    op.create_table(
        "clusters",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("centroid_embedding", Vector(512), nullable=False),
        sa.Column("member_count", sa.Integer(), server_default="0"),
        sa.Column("merged_into_identity_id", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["merged_into_identity_id"], ["identities.id"], ondelete="SET NULL"
        ),
    )

    # Cluster members table
    op.create_table(
        "cluster_members",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("embedding_vector", Vector(512), nullable=False),
        sa.Column("source_event_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["cluster_id"], ["clusters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_event_id"], ["recognition_events.id"], ondelete="SET NULL"
        ),
    )
    op.create_index("ix_cluster_members_cluster_id", "cluster_members", ["cluster_id"])


def downgrade() -> None:
    op.drop_table("cluster_members")
    op.drop_table("clusters")
    op.drop_table("recognition_events")
    op.drop_table("face_embeddings")
    op.drop_table("identities")
    op.execute("DROP EXTENSION IF EXISTS vector")
