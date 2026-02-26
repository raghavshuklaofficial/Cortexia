"""
Recognition routes — upload images for face analysis.
"""

from __future__ import annotations

import hashlib
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from cortexia.api.deps import DbSession, ApiKey, get_pipeline
from cortexia.api.schemas.models import (
    ApiResponse,
    BoundingBoxSchema,
    FaceAnalysisSchema,
    FaceAttributesSchema,
    LivenessSchema,
    RecognitionMatchSchema,
    RecognitionResponse,
)
from cortexia.core.types import FaceAnalysis, LivenessVerdict
from cortexia.db.repositories.event_repo import EventRepository

router = APIRouter(prefix="/recognize", tags=["Recognition"])


def _face_analysis_to_schema(fa: FaceAnalysis) -> FaceAnalysisSchema:
    """Convert internal FaceAnalysis to API schema."""
    liveness = None
    if fa.liveness:
        liveness = LivenessSchema(
            verdict=fa.liveness.verdict.value,
            confidence=fa.liveness.confidence,
            method=fa.liveness.method,
        )

    recognition = None
    if fa.recognition:
        recognition = RecognitionMatchSchema(
            identity_id=fa.recognition.identity_id,
            identity_name=fa.recognition.identity_name,
            distance=fa.recognition.distance,
            confidence=fa.recognition.confidence,
            is_known=fa.recognition.is_known,
        )

    attributes = None
    if fa.attributes:
        attributes = FaceAttributesSchema(
            age=fa.attributes.age,
            gender=fa.attributes.gender,
            gender_confidence=fa.attributes.gender_confidence,
            emotion=fa.attributes.emotion.value if fa.attributes.emotion else None,
            emotion_confidence=fa.attributes.emotion_confidence,
        )

    return FaceAnalysisSchema(
        bbox=BoundingBoxSchema(
            x1=fa.face.bbox.x1,
            y1=fa.face.bbox.y1,
            x2=fa.face.bbox.x2,
            y2=fa.face.bbox.y2,
        ),
        detection_confidence=fa.face.confidence,
        trust_score=fa.trust_score,
        processing_time_ms=fa.processing_time_ms,
        liveness=liveness,
        recognition=recognition,
        attributes=attributes,
        track_id=fa.track_id,
    )


@router.post("", response_model=RecognitionResponse)
async def recognize_image(
    image: UploadFile = File(..., description="Image file to analyze"),
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Run the full Trust Pipeline on an uploaded image.

    The image passes through:
    1. Face Detection (RetinaFace)
    2. Anti-Spoofing Liveness Check
    3. ArcFace Embedding Extraction
    4. Identity Recognition with Platt-calibrated confidence
    5. Attribute Analysis (age, gender, emotion)

    Returns detailed analysis for every detected face.
    """
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image file.")

    # Decode image
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    # Run Trust Pipeline
    pipeline = get_pipeline()
    frame_result = pipeline.process_image(img)

    # Log events to audit trail
    event_repo = EventRepository(db)
    frame_hash = hashlib.sha256(content).hexdigest()[:16]

    for fa in frame_result.faces:
        identity_id = fa.recognition.identity_id if fa.recognition else None
        identity_name = fa.recognition.identity_name if fa.recognition else None

        await event_repo.log_event(
            identity_id=identity_id if identity_id and identity_id > 0 else None,
            identity_name=identity_name,
            confidence=fa.recognition.confidence if fa.recognition else 0.0,
            trust_score=fa.trust_score,
            is_spoof=(
                fa.liveness is not None
                and fa.liveness.verdict == LivenessVerdict.SPOOF
            ),
            is_known=fa.recognition.is_known if fa.recognition else False,
            source="upload",
            attributes_json=fa.attributes.to_dict() if fa.attributes else None,
            bounding_box_json=fa.face.bbox.to_dict(),
            frame_hash=frame_hash,
        )

    # Build response
    face_schemas = [_face_analysis_to_schema(fa) for fa in frame_result.faces]

    return RecognitionResponse(
        faces=face_schemas,
        face_count=frame_result.face_count,
        known_count=frame_result.known_count,
        spoof_count=frame_result.spoof_count,
        total_processing_time_ms=frame_result.total_processing_time_ms,
        frame_dimensions={
            "width": frame_result.frame_width,
            "height": frame_result.frame_height,
        },
    )


@router.post("/batch", response_model=ApiResponse)
async def recognize_batch(
    images: list[UploadFile] = File(..., description="Multiple images"),
    db: DbSession = None,  # type: ignore[assignment]
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Process multiple images in batch.

    For large batches (>5 images), processing is queued as a
    background task. Results are stored in the recognition events table.
    """
    if len(images) > 20:
        raise HTTPException(
            status_code=400, detail="Maximum 20 images per batch request."
        )

    pipeline = get_pipeline()
    event_repo = EventRepository(db)
    total_faces = 0
    total_known = 0
    total_spoofs = 0

    for img_file in images:
        content = await img_file.read()
        if not content:
            continue

        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        result = pipeline.process_image(img)
        total_faces += result.face_count
        total_known += result.known_count
        total_spoofs += result.spoof_count

        frame_hash = hashlib.sha256(content).hexdigest()[:16]
        for fa in result.faces:
            await event_repo.log_event(
                identity_id=(
                    fa.recognition.identity_id
                    if fa.recognition and fa.recognition.identity_id > 0
                    else None
                ),
                identity_name=(
                    fa.recognition.identity_name if fa.recognition else None
                ),
                confidence=fa.recognition.confidence if fa.recognition else 0.0,
                trust_score=fa.trust_score,
                is_spoof=(
                    fa.liveness is not None
                    and fa.liveness.verdict == LivenessVerdict.SPOOF
                ),
                is_known=fa.recognition.is_known if fa.recognition else False,
                source="batch_upload",
                attributes_json=fa.attributes.to_dict() if fa.attributes else None,
                bounding_box_json=fa.face.bbox.to_dict(),
                frame_hash=frame_hash,
            )

    return ApiResponse(
        message=f"Batch processed: {len(images)} images.",
        data={
            "images_processed": len(images),
            "total_faces": total_faces,
            "total_known": total_known,
            "total_spoofs": total_spoofs,
        },
    )
