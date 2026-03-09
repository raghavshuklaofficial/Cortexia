"""Forensics routes — liveness checks and face quality analysis."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File

from cortexia.api.deps import ApiKey, get_pipeline
from cortexia.api.schemas.models import (
    FaceAttributesSchema,
    ForensicAnalysisResponse,
    LivenessSchema,
)
from cortexia.api.upload_utils import validate_image_upload

router = APIRouter(prefix="/forensics", tags=["Forensics"])


@router.post("/liveness", response_model=ForensicAnalysisResponse)
async def check_liveness(
    image: UploadFile = File(..., description="Face image to check"),
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Dedicated liveness detection endpoint.

    Analyzes a face image for anti-spoofing markers:
    - Frequency domain analysis for screen artifacts
    - Color space analysis for print artifacts
    - Texture analysis for micro-texture presence
    - Moiré pattern detection for screen replay
    """
    content = await validate_image_upload(image)

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    pipeline = get_pipeline()
    result = pipeline.process_image(img)

    if result.face_count == 0:
        return ForensicAnalysisResponse(face_detected=False)

    fa = result.faces[0]
    liveness = None
    if fa.liveness:
        liveness = LivenessSchema(
            verdict=fa.liveness.verdict.value,
            confidence=fa.liveness.confidence,
            method=fa.liveness.method,
        )

    return ForensicAnalysisResponse(
        face_detected=True,
        liveness=liveness,
        trust_score=fa.trust_score,
        processing_time_ms=fa.processing_time_ms,
    )


@router.post("/analyze", response_model=ForensicAnalysisResponse)
async def full_forensic_analysis(
    image: UploadFile = File(..., description="Face image for full analysis"),
    api_key: ApiKey = None,  # type: ignore[assignment]
):
    """Full forensic analysis of a face image.

    Combines liveness detection, face quality assessment,
    and attribute prediction into one comprehensive report.
    """
    content = await validate_image_upload(image)

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    pipeline = get_pipeline()
    result = pipeline.process_image(img)

    if result.face_count == 0:
        return ForensicAnalysisResponse(face_detected=False)

    fa = result.faces[0]

    liveness = None
    if fa.liveness:
        liveness = LivenessSchema(
            verdict=fa.liveness.verdict.value,
            confidence=fa.liveness.confidence,
            method=fa.liveness.method,
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

    # Face quality score based on detection confidence and image clarity
    face_quality = fa.face.confidence * 0.5
    if fa.face.aligned_face is not None:
        gray = cv2.cvtColor(fa.face.aligned_face, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 500.0)
        face_quality += sharpness * 0.5

    return ForensicAnalysisResponse(
        face_detected=True,
        liveness=liveness,
        face_quality_score=round(face_quality, 4),
        attributes=attributes,
        trust_score=fa.trust_score,
        processing_time_ms=fa.processing_time_ms,
    )
