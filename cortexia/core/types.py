"""Data types shared across the CORTEXIA core engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class LivenessVerdict(str, Enum):
    """Result of anti-spoofing liveness check."""

    LIVE = "live"
    SPOOF = "spoof"
    UNCERTAIN = "uncertain"


class EmotionLabel(str, Enum):
    """Recognized facial emotions."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box for a detected face."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_dict(self) -> dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    def iou(self, other: BoundingBox) -> float:
        """Compute Intersection over Union with another box."""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class FaceLandmarks:
    """5-point facial landmarks (left_eye, right_eye, nose, mouth_left, mouth_right)."""

    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]

    def as_array(self) -> NDArray[np.float32]:
        return np.array(
            [self.left_eye, self.right_eye, self.nose, self.mouth_left, self.mouth_right],
            dtype=np.float32,
        )


@dataclass
class DetectedFace:
    """A face detected in an image with its metadata."""

    bbox: BoundingBox
    confidence: float
    landmarks: FaceLandmarks | None = None
    aligned_face: NDArray[np.uint8] | None = None  # 112x112 aligned crop

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "confidence": round(self.confidence, 4),
            "has_landmarks": self.landmarks is not None,
        }


@dataclass
class LivenessResult:
    """Result from the anti-spoofing liveness detector."""

    verdict: LivenessVerdict
    confidence: float
    method: str = "texture_analysis"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "method": self.method,
        }


@dataclass
class FaceAttributes:
    """Predicted demographic and emotional attributes of a face."""

    age: int | None = None
    gender: str | None = None
    gender_confidence: float | None = None
    emotion: EmotionLabel | None = None
    emotion_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "gender": self.gender,
            "gender_confidence": round(self.gender_confidence, 4) if self.gender_confidence else None,
            "emotion": self.emotion.value if self.emotion else None,
            "emotion_confidence": (
                round(self.emotion_confidence, 4) if self.emotion_confidence else None
            ),
        }


@dataclass
class RecognitionMatch:
    """A single identity match candidate."""

    identity_id: int
    identity_name: str
    distance: float
    confidence: float  # Platt-scaled probability
    is_known: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "distance": round(self.distance, 6),
            "confidence": round(self.confidence, 4),
            "is_known": self.is_known,
        }


@dataclass
class FaceAnalysis:
    """Complete analysis of a single face through the Trust Pipeline."""

    face: DetectedFace
    embedding: NDArray[np.float32] | None = None
    liveness: LivenessResult | None = None
    recognition: RecognitionMatch | None = None
    attributes: FaceAttributes | None = None
    trust_score: float = 0.0
    processing_time_ms: float = 0.0
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "face": self.face.to_dict(),
            "trust_score": round(self.trust_score, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }
        if self.liveness:
            result["liveness"] = self.liveness.to_dict()
        if self.recognition:
            result["recognition"] = self.recognition.to_dict()
        if self.attributes:
            result["attributes"] = self.attributes.to_dict()
        if self.track_id is not None:
            result["track_id"] = self.track_id
        return result


@dataclass
class FrameAnalysis:
    """Complete analysis of all faces in a single frame."""

    faces: list[FaceAnalysis] = field(default_factory=list)
    frame_index: int = 0
    total_processing_time_ms: float = 0.0
    frame_width: int = 0
    frame_height: int = 0

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def known_count(self) -> int:
        return sum(
            1 for f in self.faces if f.recognition and f.recognition.is_known
        )

    @property
    def spoof_count(self) -> int:
        return sum(
            1
            for f in self.faces
            if f.liveness and f.liveness.verdict == LivenessVerdict.SPOOF
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "faces": [f.to_dict() for f in self.faces],
            "face_count": self.face_count,
            "known_count": self.known_count,
            "spoof_count": self.spoof_count,
            "frame_index": self.frame_index,
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "frame_dimensions": {
                "width": self.frame_width,
                "height": self.frame_height,
            },
        }
