"""
Tests for core data types and geometry helpers.
"""

import numpy as np
import pytest

from cortexia.core.types import (
    BoundingBox,
    DetectedFace,
    FaceAnalysis,
    FaceLandmarks,
    FrameAnalysis,
    LivenessResult,
    LivenessVerdict,
    RecognitionMatch,
)


class TestBoundingBox:
    def test_width_height(self):
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=170)
        assert bbox.width == 100
        assert bbox.height == 150

    def test_area(self):
        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=80)
        assert bbox.area == 4000

    def test_center(self):
        bbox = BoundingBox(x1=100, y1=200, x2=200, y2=400)
        cx, cy = bbox.center
        assert cx == 150.0
        assert cy == 300.0

    def test_iou_identical(self):
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.iou(bbox) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_partial(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        assert a.iou(b) == pytest.approx(2500 / 17500, rel=1e-3)

    def test_iou_contained(self):
        outer = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        inner = BoundingBox(x1=25, y1=25, x2=75, y2=75)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 2500 - 2500 = 10000
        assert outer.iou(inner) == pytest.approx(2500 / 10000, rel=1e-3)


class TestFaceLandmarks:
    def test_landmarks_creation(self):
        lm = FaceLandmarks(
            left_eye=(30.0, 40.0),
            right_eye=(70.0, 40.0),
            nose=(50.0, 60.0),
            mouth_left=(35.0, 80.0),
            mouth_right=(65.0, 80.0),
        )
        assert lm.left_eye == (30.0, 40.0)
        assert lm.nose == (50.0, 60.0)


class TestDetectedFace:
    def test_creation(self, sample_face_crop):
        face = DetectedFace(
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
            confidence=0.95,
            aligned_face=sample_face_crop,
        )
        assert face.confidence == 0.95
        assert face.aligned_face is not None


class TestLivenessResult:
    def test_live_verdict(self):
        lr = LivenessResult(
            verdict=LivenessVerdict.LIVE,
            confidence=0.92,
            method="all_passed",
        )
        assert lr.verdict == LivenessVerdict.LIVE
        assert lr.confidence == 0.92

    def test_spoof_verdict(self):
        lr = LivenessResult(
            verdict=LivenessVerdict.SPOOF,
            confidence=0.85,
            method="fft_anomaly+texture_anomaly",
        )
        assert lr.verdict == LivenessVerdict.SPOOF


class TestFrameAnalysis:
    def test_empty_frame(self):
        fa = FrameAnalysis(
            faces=[],
            total_processing_time_ms=5.0,
        )
        assert fa.face_count == 0
        assert len(fa.faces) == 0
