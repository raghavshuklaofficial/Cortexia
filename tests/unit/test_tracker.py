"""
Tracker tests (IOU-based SORT).
"""

import numpy as np
import pytest

from cortexia.core.tracker import FaceTracker
from cortexia.core.types import BoundingBox, DetectedFace


def _make_detection(x1, y1, x2, y2, conf=0.95):
    """Helper to create a detection with a random embedding."""
    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return DetectedFace(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=conf,
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
    ), emb


class TestFaceTracker:
    def test_new_track_creation(self):
        tracker = FaceTracker()
        det, emb = _make_detection(100, 100, 200, 200)
        active, needs_recog = tracker.update([det], [emb])

        assert len(active) >= 1
        assert len(needs_recog) >= 1  # first detection needs recognition

    def test_track_continuation(self):
        tracker = FaceTracker()

        # Frame 1
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        det1 = DetectedFace(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.95,
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        )
        tracker.update([det1], [emb])

        # Frame 2 — slightly moved
        det2 = DetectedFace(
            bbox=BoundingBox(x1=105, y1=105, x2=205, y2=205),
            confidence=0.95,
            aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        )
        emb2 = emb + np.random.randn(512).astype(np.float32) * 0.01
        emb2 /= np.linalg.norm(emb2)
        active, _ = tracker.update([det2], [emb2])

        # Should still be 1 track (continued, not new)
        assert len(active) == 1

    def test_track_disappearance(self):
        tracker = FaceTracker(max_age=2)

        det, emb = _make_detection(100, 100, 200, 200)
        tracker.update([det], [emb])

        # Several empty frames
        tracker.update([])
        tracker.update([])
        active, _ = tracker.update([])

        # Track should be gone after max_age
        assert len(active) == 0

    def test_multiple_faces(self):
        tracker = FaceTracker()

        det1, emb1 = _make_detection(50, 50, 150, 150)
        det2, emb2 = _make_detection(300, 300, 400, 400)
        active, needs_recog = tracker.update([det1, det2], [emb1, emb2])

        assert len(active) == 2
