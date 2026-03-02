"""
Unit tests for cortexia.config — settings and validation.
"""

import os
from unittest.mock import patch

import pytest


class TestSettings:
    def test_default_values(self):
        from cortexia.config import Settings

        s = Settings(_env_file=None)
        assert s.app_name == "cortexia"
        assert s.debug is False
        assert s.embedding_dim == 512

    def test_cors_parsing(self):
        from cortexia.config import Settings

        s = Settings(_env_file=None, cors_origins='["http://localhost:3000"]')
        assert isinstance(s.cors_origins, list)
        assert "http://localhost:3000" in s.cors_origins

    def test_cors_single_string(self):
        from cortexia.config import Settings

        s = Settings(_env_file=None, cors_origins="http://localhost:3000")
        assert isinstance(s.cors_origins, list)
        assert "http://localhost:3000" in s.cors_origins

    def test_database_url_default(self):
        from cortexia.config import Settings

        s = Settings(_env_file=None)
        assert "postgresql+asyncpg" in s.database_url

    def test_trust_pipeline_weights(self):
        from cortexia.config import Settings

        s = Settings(_env_file=None)
        total = s.trust_weight_detection + s.trust_weight_liveness + s.trust_weight_recognition
        assert total == pytest.approx(1.0)
