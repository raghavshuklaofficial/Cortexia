"""
Identity management routes — CRUD for enrolled people.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from cortexia.api.deps import DbSession, ApiKey, get_pipeline
from cortexia.api.schemas.models import (
    ApiResponse,
    IdentityCreate,
    IdentityListResponse,
    IdentityResponse,
    IdentityUpdate,
    PaginationMeta,
)
from cortexia.api.upload_utils import validate_image_upload
from cortexia.db.repositories.identity_repo import IdentityRepository
from cortexia.db.repositories.vector_repo import VectorRepository

router = APIRouter(prefix="/identities", tags=["Identities"])


@router.post("", response_model=ApiResponse, status_code=201)
async def create_identity(
    name: str = Form(...),
    metadata: str | None = Form(default=None),
    images: list[UploadFile] = File(..., description="Face images for enrollment"),
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Create a new identity with face images.

    Upload 1-10 photos of the same person. Each image must contain
    exactly one detectable face. The system extracts 512-d ArcFace
    embeddings and stores them for future recognition.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one face image is required.")
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per enrollment.")

    pipeline = get_pipeline()
    identity_repo = IdentityRepository(db)
    vector_repo = VectorRepository(db)

    # Parse metadata JSON
    import json

    meta_dict = None
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON.")

    # Create identity
    identity = await identity_repo.create(name=name, metadata=meta_dict)

    # Process each image
    embedding_count = 0
    for img_file in images:
        try:
            content = await validate_image_upload(img_file)
        except HTTPException:
            continue

        # Decode image
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            continue

        # Run detection
        result = pipeline.process_image(image)
        if result.face_count == 0:
            continue
        if result.face_count > 1:
            # Take the largest face
            pass

        # Take the first (or largest) face
        face_analysis = result.faces[0]
        if face_analysis.embedding is None:
            continue

        # Store embedding
        img_hash = hashlib.sha256(content).hexdigest()

        # Check for duplicate
        existing = await vector_repo.check_duplicate(
            face_analysis.embedding.tolist(), threshold=0.15
        )
        if existing and existing["identity_id"] != identity.id:
            # This face already belongs to someone else
            continue

        await vector_repo.add_embedding(
            identity_id=identity.id,
            embedding=face_analysis.embedding.tolist(),
            source_image_hash=img_hash,
        )
        embedding_count += 1

    if embedding_count == 0:
        # Rollback — delete the empty identity
        await identity_repo.hard_delete(identity.id)
        raise HTTPException(
            status_code=400,
            detail="No valid faces detected in any of the uploaded images.",
        )

    # Reload gallery
    await _refresh_gallery(db, pipeline)

    return ApiResponse(
        success=True,
        message=f"Identity '{name}' created with {embedding_count} face(s).",
        data={
            "id": identity.id,
            "name": identity.name,
            "face_count": embedding_count,
        },
    )


@router.get("", response_model=IdentityListResponse)
async def list_identities(
    skip: int = 0,
    limit: int = 50,
    search: str | None = None,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """List all enrolled identities with pagination."""
    repo = IdentityRepository(db)
    identities, total = await repo.get_all(skip=skip, limit=limit, search=search)

    items = []
    for ident in identities:
        last_seen = await repo.get_last_seen(ident.id)
        items.append(
            IdentityResponse(
                id=ident.id,
                name=ident.name,
                metadata=ident.metadata_json,
                face_count=len(ident.embeddings),
                privacy_score=ident.privacy_score,
                is_active=ident.is_active,
                last_seen=last_seen,
                created_at=ident.created_at,
                updated_at=ident.updated_at,
            )
        )

    return IdentityListResponse(
        identities=items,
        pagination=PaginationMeta(
            total=total, skip=skip, limit=limit, has_more=(skip + limit < total)
        ),
    )


@router.get("/{identity_id}", response_model=ApiResponse)
async def get_identity(
    identity_id: int,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Get a single identity by ID."""
    repo = IdentityRepository(db)
    identity = await repo.get_by_id(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found.")

    last_seen = await repo.get_last_seen(identity_id)

    return ApiResponse(
        data={
            "id": identity.id,
            "name": identity.name,
            "metadata": identity.metadata_json,
            "face_count": len(identity.embeddings),
            "privacy_score": identity.privacy_score,
            "last_seen": str(last_seen) if last_seen else None,
            "created_at": str(identity.created_at),
            "updated_at": str(identity.updated_at),
        }
    )


@router.put("/{identity_id}", response_model=ApiResponse)
async def update_identity(
    identity_id: int,
    body: IdentityUpdate,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Update an identity's name or metadata."""
    repo = IdentityRepository(db)
    identity = await repo.update(
        identity_id, name=body.name, metadata=body.metadata
    )
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found.")

    return ApiResponse(
        message="Identity updated.",
        data={"id": identity.id, "name": identity.name},
    )


@router.delete("/{identity_id}", response_model=ApiResponse)
async def delete_identity(
    identity_id: int,
    hard: bool = False,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Delete an identity.

    Soft delete by default. Use ?hard=true for GDPR-compliant
    full data erasure (removes embeddings and all events).
    """
    repo = IdentityRepository(db)
    if hard:
        deleted = await repo.hard_delete(identity_id)
    else:
        deleted = await repo.soft_delete(identity_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Identity not found.")

    # Refresh gallery
    pipeline = get_pipeline()
    await _refresh_gallery(db, pipeline)

    return ApiResponse(message="Identity deleted.", data={"id": identity_id, "hard": hard})


@router.post("/{identity_id}/faces", response_model=ApiResponse)
async def add_faces(
    identity_id: int,
    images: list[UploadFile] = File(...),
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Add additional face images to an existing identity."""
    repo = IdentityRepository(db)
    identity = await repo.get_by_id(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found.")

    pipeline = get_pipeline()
    vector_repo = VectorRepository(db)
    added = 0

    for img_file in images:
        try:
            content = await validate_image_upload(img_file)
        except HTTPException:
            continue

        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            continue

        result = pipeline.process_image(image)
        if result.face_count == 0:
            continue

        face_analysis = result.faces[0]
        if face_analysis.embedding is None:
            continue

        img_hash = hashlib.sha256(content).hexdigest()
        await vector_repo.add_embedding(
            identity_id=identity_id,
            embedding=face_analysis.embedding.tolist(),
            source_image_hash=img_hash,
        )
        added += 1

    if added > 0:
        await _refresh_gallery(db, pipeline)

    return ApiResponse(
        message=f"Added {added} face(s) to identity '{identity.name}'.",
        data={"identity_id": identity_id, "faces_added": added},
    )


@router.get("/{identity_id}/timeline", response_model=ApiResponse)
async def get_identity_timeline(
    identity_id: int,
    limit: int = 50,
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Get the recognition event timeline for an identity."""
    from cortexia.db.repositories.event_repo import EventRepository

    repo = IdentityRepository(db)
    identity = await repo.get_by_id(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found.")

    event_repo = EventRepository(db)
    events, total = await event_repo.get_events(
        identity_id=identity_id, limit=limit
    )

    return ApiResponse(
        data={
            "identity_id": identity_id,
            "identity_name": identity.name,
            "total_events": total,
            "events": [
                {
                    "id": e.id,
                    "timestamp": str(e.timestamp),
                    "confidence": e.confidence,
                    "trust_score": e.trust_score,
                    "source": e.source,
                    "is_spoof": e.is_spoof,
                    "attributes": e.attributes_json,
                }
                for e in events
            ],
        }
    )


async def _refresh_gallery(db: DbSession, pipeline) -> None:
    """Reload the recognition gallery from the database."""
    from cortexia.db.repositories.identity_repo import IdentityRepository
    from cortexia.core.recognizer import StoredIdentity

    repo = IdentityRepository(db)
    identities = await repo.get_all_with_embeddings()

    gallery = []
    for ident in identities:
        if not ident.embeddings:
            continue
        gallery.append(
            StoredIdentity(
                identity_id=ident.id,
                name=ident.name,
                embeddings=[np.array(e.embedding, dtype=np.float32) for e in ident.embeddings],
            )
        )

    pipeline.recognizer.load_gallery(gallery)
