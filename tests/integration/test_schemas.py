"""
Integration tests for Pydantic schema validation.
"""

from __future__ import annotations

import pytest


class TestSchemaValidation:
    """Test Pydantic schema serialization."""

    def test_identity_create_schema(self):
        from cortexia.api.schemas.models import IdentityResponse

        resp = IdentityResponse(
            id=1,
            name="Alice",
            metadata={},
            face_count=3,
            privacy_score=1.0,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            last_seen=None,
        )
        assert resp.name == "Alice"
        assert resp.face_count == 3

    def test_recognition_response_schema(self):
        from cortexia.api.schemas.models import (
            BoundingBoxSchema,
            RecognitionResponse,
            FaceAnalysisSchema,
        )

        face = FaceAnalysisSchema(
            bbox=BoundingBoxSchema(x1=10, y1=20, x2=100, y2=150),
            detection_confidence=0.95,
            trust_score=0.82,
            processing_time_ms=12.3,
            liveness=None,
            recognition=None,
            attributes=None,
        )
        resp = RecognitionResponse(
            faces=[face],
            face_count=1,
            known_count=0,
            spoof_count=0,
            total_processing_time_ms=45.2,
            frame_dimensions={"width": 640, "height": 480},
        )
        assert resp.face_count == 1
        assert resp.faces[0].detection_confidence == 0.95

    def test_overview_stats_schema(self):
        from cortexia.api.schemas.models import OverviewStats

        stats = OverviewStats(
            total_identities=10,
            total_events=500,
            known_events=400,
            unknown_events=100,
            spoof_events=2,
            unknown_ratio=0.2,
            avg_trust_score=0.85,
            avg_recognition_confidence=0.90,
        )
        assert stats.total_identities == 10
        assert stats.avg_trust_score == 0.85

    def test_api_response_envelope(self):
        from cortexia.api.schemas.models import ApiResponse

        resp = ApiResponse(success=True, data={"key": "value"})
        assert resp.success is True
        assert resp.data["key"] == "value"

    def test_health_response(self):
        from cortexia.api.schemas.models import HealthResponse

        hr = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.5,
        )
        assert hr.status == "healthy"
        assert hr.uptime_seconds > 0
