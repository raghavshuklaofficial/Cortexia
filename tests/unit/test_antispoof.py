"""
Unit tests for cortexia.core.models.antispoof — liveness detection.
"""

import numpy as np
import pytest

from cortexia.core.models.antispoof import AntiSpoofDetector
from cortexia.core.types import LivenessVerdict


class TestAntiSpoofDetector:
    @pytest.fixture
    def detector(self):
        return AntiSpoofDetector()

    def test_detect_returns_liveness_result(self, detector, sample_face_crop):
        result = detector.detect(sample_face_crop)
        assert result.verdict in (
            LivenessVerdict.LIVE,
            LivenessVerdict.SPOOF,
            LivenessVerdict.UNCERTAIN,
        )
        assert 0 <= result.confidence <= 1

    def test_detect_has_method(self, detector, sample_face_crop):
        result = detector.detect(sample_face_crop)
        assert isinstance(result.method, str)
        assert len(result.method) > 0

    def test_confidence_bounded(self, detector, sample_face_crop):
        result = detector.detect(sample_face_crop)
        assert 0.0 <= result.confidence <= 1.0

    def test_uniform_color_image(self, detector):
        """A completely uniform image should likely be flagged."""
        uniform = np.full((112, 112, 3), 128, dtype=np.uint8)
        result = detector.detect(uniform)
        # Uniform images have very low texture — likely spoof or uncertain
        assert result.verdict in (LivenessVerdict.SPOOF, LivenessVerdict.UNCERTAIN)

    def test_high_frequency_noise(self, detector):
        """An image with high noise should have different freq analysis."""
        noisy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = detector.detect(noisy)
        # Random noise should still produce a valid result
        assert result.verdict in (
            LivenessVerdict.LIVE,
            LivenessVerdict.SPOOF,
            LivenessVerdict.UNCERTAIN,
        )

    def test_deterministic_same_input(self, detector, sample_face_crop):
        """Same input should give same output."""
        r1 = detector.detect(sample_face_crop)
        r2 = detector.detect(sample_face_crop)
        assert r1.verdict == r2.verdict
        assert r1.confidence == pytest.approx(r2.confidence, abs=1e-6)
