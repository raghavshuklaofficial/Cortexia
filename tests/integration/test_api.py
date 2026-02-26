"""
Integration tests for API endpoints.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock


class TestHealthEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from cortexia.main import app

        return TestClient(app, raise_server_exceptions=False)

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "uptime_seconds" in body

    def test_unknown_route_returns_404(self, client):
        resp = client.get("/api/v1/does_not_exist")
        assert resp.status_code == 404


class TestSchemaModels:
    def test_identity_response(self):
        from cortexia.api.schemas.models import IdentityResponse

        ir = IdentityResponse(
            id=1,
            name="Test User",
            metadata={"role": "admin"},
            face_count=5,
            privacy_score=0.95,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            last_seen=None,
        )
        assert ir.id == 1
        assert ir.metadata["role"] == "admin"

    def test_overview_stats(self):
        from cortexia.api.schemas.models import OverviewStats

        stats = OverviewStats(
            total_identities=100,
            total_events=10000,
            known_events=8000,
            unknown_events=2000,
            spoof_events=50,
            unknown_ratio=0.2,
            avg_trust_score=0.88,
            avg_recognition_confidence=0.91,
        )
        assert stats.total_identities == 100
