"""
Pydantic schemas for all CORTEXIA API request/response models.

These schemas define the contract between the API and clients.
All responses use consistent envelope structure.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ─── Common ──────────────────────────────────────────────────


class PaginationMeta(BaseModel):
    """Pagination metadata included in list responses."""

    total: int
    skip: int
    limit: int
    has_more: bool


class ApiResponse(BaseModel):
    """Standard API response envelope."""

    success: bool = True
    message: str = "OK"
    data: dict | list | None = None
    meta: dict | None = None


# ─── Identity Schemas ────────────────────────────────────────


class IdentityCreate(BaseModel):
    """Request schema for creating a new identity."""

    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    metadata: dict | None = Field(default=None, description="Custom metadata (department, etc.)")


class IdentityUpdate(BaseModel):
    """Request schema for updating an identity."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    metadata: dict | None = None


class IdentityResponse(BaseModel):
    """Response schema for a single identity."""

    id: int
    name: str
    metadata: dict | None = None
    face_count: int = 0
    privacy_score: float = 0.0
    is_active: bool = True
    last_seen: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IdentityListResponse(BaseModel):
    """Response schema for identity listing."""

    identities: list[IdentityResponse]
    pagination: PaginationMeta


# ─── Recognition Schemas ─────────────────────────────────────


class BoundingBoxSchema(BaseModel):
    """Bounding box coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int


class LivenessSchema(BaseModel):
    """Liveness detection result."""

    verdict: str  # "live", "spoof", "uncertain"
    confidence: float
    method: str


class RecognitionMatchSchema(BaseModel):
    """Recognition match result."""

    identity_id: int
    identity_name: str
    distance: float
    confidence: float
    is_known: bool


class FaceAttributesSchema(BaseModel):
    """Predicted face attributes."""

    age: int | None = None
    gender: str | None = None
    gender_confidence: float | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None


class FaceAnalysisSchema(BaseModel):
    """Complete analysis result for a single face."""

    bbox: BoundingBoxSchema
    detection_confidence: float
    trust_score: float
    processing_time_ms: float
    liveness: LivenessSchema | None = None
    recognition: RecognitionMatchSchema | None = None
    attributes: FaceAttributesSchema | None = None
    track_id: int | None = None


class RecognitionResponse(BaseModel):
    """Response schema for image recognition."""

    faces: list[FaceAnalysisSchema]
    face_count: int
    known_count: int
    spoof_count: int
    total_processing_time_ms: float
    frame_dimensions: dict[str, int]


# ─── Event Schemas ───────────────────────────────────────────


class RecognitionEventResponse(BaseModel):
    """Response schema for a recognition event."""

    id: int
    identity_id: int | None
    identity_name: str | None
    timestamp: datetime
    confidence: float
    trust_score: float
    is_spoof: bool
    is_known: bool
    source: str
    attributes: dict | None = None
    bounding_box: dict | None = None

    model_config = {"from_attributes": True}


class EventListResponse(BaseModel):
    """Response schema for event listing."""

    events: list[RecognitionEventResponse]
    pagination: PaginationMeta


# ─── Analytics Schemas ───────────────────────────────────────


class OverviewStats(BaseModel):
    """Dashboard overview statistics."""

    total_identities: int
    total_events: int
    known_events: int
    unknown_events: int
    spoof_events: int
    unknown_ratio: float
    avg_trust_score: float
    avg_recognition_confidence: float


class TimelinePoint(BaseModel):
    """A single point on the timeline chart."""

    period: str
    total: int
    known: int
    spoofs: int


class DemographicsResponse(BaseModel):
    """Aggregated demographics data."""

    age_distribution: dict
    gender_distribution: dict[str, int]
    emotion_distribution: dict[str, int]


# ─── Cluster Schemas ─────────────────────────────────────────


class ClusterResponse(BaseModel):
    """Response schema for a discovered cluster."""

    id: int
    member_count: int
    merged_into_identity_id: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ClusterAssignRequest(BaseModel):
    """Request to assign a cluster to an identity."""

    identity_id: int | None = None
    new_identity_name: str | None = None


# ─── Forensics Schemas ───────────────────────────────────────


class ForensicAnalysisResponse(BaseModel):
    """Full forensic analysis result."""

    face_detected: bool
    liveness: LivenessSchema | None = None
    face_quality_score: float = 0.0
    attributes: FaceAttributesSchema | None = None
    trust_score: float = 0.0
    processing_time_ms: float = 0.0


# ─── System Schemas ──────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str
    database: str
    redis: str
    models_loaded: bool


class SystemInfo(BaseModel):
    """System configuration and capabilities."""

    version: str
    detection_backend: str
    embedding_dim: int
    gpu_available: bool
    trust_pipeline_enabled: bool
    antispoof_enabled: bool
    attributes_enabled: bool
    total_identities: int
    total_embeddings: int



