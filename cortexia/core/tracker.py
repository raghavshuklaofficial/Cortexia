"""
Multi-face tracker for video streams.

Uses a combination of IoU (Intersection over Union) and embedding
similarity to maintain consistent track IDs across frames. This
avoids re-running the full recognition pipeline every frame —
only new tracks or periodically refreshed tracks get full analysis.

Based on a simplified SORT (Simple Online and Realtime Tracking)
algorithm adapted for face tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

import structlog

from cortexia.core.types import BoundingBox, DetectedFace, FaceAnalysis

logger = structlog.get_logger(__name__)


@dataclass
class Track:
    """A tracked face across video frames."""

    track_id: int
    bbox: BoundingBox
    embedding: NDArray[np.float32] | None = None
    last_analysis: FaceAnalysis | None = None
    age: int = 0  # Frames since last detection
    hits: int = 1  # Total successful associations
    frames_since_recognition: int = 0  # For periodic re-recognition

    def predict_next_bbox(self) -> BoundingBox:
        """Simple prediction: assume face stays in same position."""
        return self.bbox


class FaceTracker:
    """Multi-face tracker for video streams.

    Maintains persistent track IDs across frames to:
    1. Avoid redundant recognition on every frame
    2. Provide smooth bounding box tracking
    3. Aggregate recognition results over time per track

    A track is created when a new face appears and destroyed
    when it hasn't been detected for max_age frames.
    """

    def __init__(
        self,
        max_age: int = 30,
        iou_threshold: float = 0.3,
        embedding_threshold: float = 0.45,
        recognition_interval: int = 5,
    ) -> None:
        """Initialize the tracker.

        Args:
            max_age: Delete track after this many frames without detection
            iou_threshold: Minimum IoU for spatial association
            embedding_threshold: Minimum cosine sim for embedding association
            recognition_interval: Re-recognize every N frames per track
        """
        self._max_age = max_age
        self._iou_threshold = iou_threshold
        self._embedding_threshold = embedding_threshold
        self._recognition_interval = recognition_interval
        self._tracks: list[Track] = []
        self._next_id = 1

        logger.info(
            "face_tracker_initialized",
            max_age=max_age,
            iou_threshold=iou_threshold,
            recognition_interval=recognition_interval,
        )

    @property
    def active_tracks(self) -> list[Track]:
        """Currently active tracks."""
        return [t for t in self._tracks if t.age <= self._max_age]

    def update(
        self,
        detections: list[DetectedFace],
        embeddings: list[NDArray[np.float32]] | None = None,
    ) -> tuple[list[Track], list[int]]:
        """Update tracks with new frame detections.

        Args:
            detections: Faces detected in current frame
            embeddings: Optional embeddings for each detection

        Returns:
            Tuple of (all active tracks, indices of detections needing recognition).
            Detections needing recognition are new tracks or tracks due for refresh.
        """
        start = time.perf_counter()

        # Age all existing tracks
        for track in self._tracks:
            track.age += 1
            track.frames_since_recognition += 1

        if not detections:
            # Remove expired tracks
            self._tracks = [t for t in self._tracks if t.age <= self._max_age]
            return self.active_tracks, []

        # Associate detections with existing tracks using IoU
        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()
        needs_recognition: list[int] = []

        # Build cost matrix (IoU)
        for det_idx, det in enumerate(detections):
            best_iou = 0.0
            best_track_idx = -1

            for track_idx, track in enumerate(self._tracks):
                if track_idx in matched_track_indices:
                    continue
                if track.age > self._max_age:
                    continue

                iou = det.bbox.iou(track.bbox)

                # Also check embedding similarity if available
                if (
                    embeddings is not None
                    and embeddings[det_idx] is not None
                    and track.embedding is not None
                ):
                    emb_sim = float(
                        np.dot(
                            embeddings[det_idx].astype(np.float32),
                            track.embedding.astype(np.float32),
                        )
                    )
                    # Weighted combination of IoU and embedding similarity
                    combined = iou * 0.4 + emb_sim * 0.6
                else:
                    combined = iou

                if combined > best_iou:
                    best_iou = combined
                    best_track_idx = track_idx

            if best_iou >= self._iou_threshold and best_track_idx >= 0:
                # Match found — update existing track
                track = self._tracks[best_track_idx]
                track.bbox = det.bbox
                track.age = 0
                track.hits += 1
                if embeddings is not None and embeddings[det_idx] is not None:
                    track.embedding = embeddings[det_idx]

                matched_track_indices.add(best_track_idx)
                matched_det_indices.add(det_idx)

                # Check if track needs re-recognition
                if track.frames_since_recognition >= self._recognition_interval:
                    needs_recognition.append(det_idx)
                    track.frames_since_recognition = 0

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue

            new_track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                embedding=(
                    embeddings[det_idx]
                    if embeddings is not None and det_idx < len(embeddings)
                    else None
                ),
            )
            self._tracks.append(new_track)
            self._next_id += 1
            needs_recognition.append(det_idx)

        # Remove expired tracks
        self._tracks = [t for t in self._tracks if t.age <= self._max_age]

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "tracker_update",
            active_tracks=len(self.active_tracks),
            new_detections=len(needs_recognition),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return self.active_tracks, needs_recognition

    def get_track_for_bbox(self, bbox: BoundingBox) -> Track | None:
        """Find the track associated with a bounding box."""
        best_iou = 0.0
        best_track = None
        for track in self.active_tracks:
            iou = bbox.iou(track.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        return best_track if best_iou > 0.1 else None

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1
