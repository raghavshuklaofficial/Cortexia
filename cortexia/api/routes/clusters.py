"""
Cluster routes — zero-shot identity discovery.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from cortexia.api.deps import DbSession, ApiKey, get_pipeline
from cortexia.api.schemas.models import ApiResponse, ClusterAssignRequest, ClusterResponse
from cortexia.db.models import Cluster, ClusterMember
from cortexia.db.repositories.identity_repo import IdentityRepository
from cortexia.db.repositories.vector_repo import VectorRepository

from sqlalchemy import select

router = APIRouter(prefix="/clusters", tags=["Clusters"])


@router.post("/discover", response_model=ApiResponse)
async def discover_clusters(
    min_cluster_size: int = 3,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Trigger HDBSCAN clustering on unidentified face embeddings.

    Scans the recognition_events table for unknown faces, extracts
    their embeddings, and groups them into identity clusters using
    density-based clustering.
    """
    pipeline = get_pipeline()
    clusterer = pipeline.clusterer

    # Get all cluster member embeddings (unknown faces)
    stmt = select(ClusterMember)
    result = await db.execute(stmt)
    members = list(result.scalars().all())

    if len(members) < min_cluster_size:
        return ApiResponse(
            message="Not enough unidentified faces for clustering.",
            data={"embedding_count": len(members), "min_required": min_cluster_size},
        )

    embeddings = np.array([m.embedding_vector for m in members], dtype=np.float32)
    cluster_result = clusterer.cluster(embeddings)

    # Store clusters in database
    created_clusters = 0
    for cluster in cluster_result.clusters:
        if cluster.is_noise:
            continue

        db_cluster = Cluster(
            centroid_embedding=cluster.centroid.tolist(),
            member_count=cluster.member_count,
        )
        db.add(db_cluster)
        created_clusters += 1

    await db.flush()

    return ApiResponse(
        message=f"Discovered {created_clusters} identity clusters.",
        data=cluster_result.to_dict(),
    )


@router.get("", response_model=ApiResponse)
async def list_clusters(
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """List all discovered clusters."""
    stmt = (
        select(Cluster)
        .where(Cluster.merged_into_identity_id.is_(None))
        .order_by(Cluster.member_count.desc())
    )
    result = await db.execute(stmt)
    clusters = list(result.scalars().all())

    return ApiResponse(
        data={
            "clusters": [
                {
                    "id": c.id,
                    "member_count": c.member_count,
                    "created_at": str(c.created_at),
                }
                for c in clusters
            ],
            "total": len(clusters),
        }
    )


@router.post("/{cluster_id}/assign", response_model=ApiResponse)
async def assign_cluster(
    cluster_id: int,
    body: ClusterAssignRequest,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Assign a cluster to an existing identity or create a new one.

    This merges the cluster's face embeddings into the identity's
    gallery, improving future recognition accuracy.
    """
    stmt = select(Cluster).where(Cluster.id == cluster_id)
    result = await db.execute(stmt)
    cluster = result.scalar_one_or_none()

    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found.")

    identity_repo = IdentityRepository(db)
    vector_repo = VectorRepository(db)

    if body.identity_id:
        identity = await identity_repo.get_by_id(body.identity_id)
        if not identity:
            raise HTTPException(status_code=404, detail="Identity not found.")
    elif body.new_identity_name:
        identity = await identity_repo.create(name=body.new_identity_name)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either identity_id or new_identity_name.",
        )

    # Move cluster members' embeddings to the identity
    member_stmt = select(ClusterMember).where(ClusterMember.cluster_id == cluster_id)
    member_result = await db.execute(member_stmt)
    members = list(member_result.scalars().all())

    for member in members:
        await vector_repo.add_embedding(
            identity_id=identity.id,
            embedding=member.embedding_vector,
        )

    # Mark cluster as merged
    cluster.merged_into_identity_id = identity.id
    await db.flush()

    # Refresh gallery
    pipeline = get_pipeline()
    from cortexia.api.routes.identities import _refresh_gallery

    await _refresh_gallery(db, pipeline)

    return ApiResponse(
        message=f"Cluster {cluster_id} merged into identity '{identity.name}'.",
        data={
            "cluster_id": cluster_id,
            "identity_id": identity.id,
            "embeddings_transferred": len(members),
        },
    )
