"""
Face detection backends for CORTEXIA.

Supports multiple detection backends via a factory pattern:
  - RetinaFace (via InsightFace) — high accuracy, GPU-capable
  - MediaPipe — lightweight CPU fallback

Each detector returns a list of DetectedFace objects with aligned crops
ready for embedding extraction.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

import structlog

from cortexia.core.types import BoundingBox, DetectedFace, FaceLandmarks

logger = structlog.get_logger(__name__)

# Standard aligned face size for ArcFace embedding models
ALIGNED_FACE_SIZE = (112, 112)


class FaceDetectorProtocol(Protocol):
    """Protocol that all face detectors must satisfy."""

    def detect(
        self, image: NDArray[np.uint8], threshold: float = 0.5
    ) -> list[DetectedFace]: ...

    @property
    def name(self) -> str: ...


class BaseFaceDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(
        self, image: NDArray[np.uint8], threshold: float = 0.5
    ) -> list[DetectedFace]:
        """Detect faces in a BGR image.

        Args:
            image: BGR uint8 image (H, W, 3)
            threshold: Minimum detection confidence

        Returns:
            List of detected faces with aligned crops
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this detector backend."""

    @staticmethod
    def align_face(
        image: NDArray[np.uint8],
        landmarks: FaceLandmarks,
        output_size: tuple[int, int] = ALIGNED_FACE_SIZE,
    ) -> NDArray[np.uint8]:
        """Align a face crop using 5-point landmarks via similarity transform.

        Uses the standard ArcFace alignment reference points.
        """
        # ArcFace standard reference landmarks for 112x112
        ref_pts = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        src_pts = landmarks.as_array()

        # Estimate similarity transform (no shear)
        tform = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
        if tform[0] is None:
            # Fallback: simple crop and resize
            bbox = cv2.boundingRect(src_pts.astype(np.int32))
            x, y, w, h = bbox
            x, y = max(0, x), max(0, y)
            crop = image[y : y + h, x : x + w]
            if crop.size == 0:
                return cv2.resize(image, output_size)
            return cv2.resize(crop, output_size)

        aligned = cv2.warpAffine(
            image,
            tform[0],
            output_size,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned


class RetinaFaceDetector(BaseFaceDetector):
    """High-accuracy face detector using InsightFace's RetinaFace model.

    This is the primary detector for CORTEXIA, offering:
    - Sub-pixel facial landmark detection (5-point)
    - Multi-scale face detection
    - GPU acceleration when available
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1) -> None:
        """Initialize InsightFace model.

        Args:
            model_name: InsightFace model pack name
            ctx_id: -1 for CPU, 0+ for GPU device ID
        """
        import insightface  # type: ignore[import-untyped]

        self._app = insightface.app.FaceAnalysis(
            name=model_name,
            allowed_modules=["detection"],
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if ctx_id >= 0
                else ["CPUExecutionProvider"]
            ),
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("retinaface_detector_initialized", model=model_name, gpu=ctx_id >= 0)

    @property
    def name(self) -> str:
        return "retinaface"

    def detect(
        self, image: NDArray[np.uint8], threshold: float = 0.5
    ) -> list[DetectedFace]:
        """Detect faces using RetinaFace."""
        start = time.perf_counter()
        raw_faces = self._app.get(image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[DetectedFace] = []
        for face in raw_faces:
            score = float(face.det_score)
            if score < threshold:
                continue

            bbox_arr = face.bbox.astype(int)
            bbox = BoundingBox(
                x1=max(0, int(bbox_arr[0])),
                y1=max(0, int(bbox_arr[1])),
                x2=min(image.shape[1], int(bbox_arr[2])),
                y2=min(image.shape[0], int(bbox_arr[3])),
            )

            landmarks = None
            aligned = None
            if face.kps is not None:
                kps = face.kps
                landmarks = FaceLandmarks(
                    left_eye=(float(kps[0][0]), float(kps[0][1])),
                    right_eye=(float(kps[1][0]), float(kps[1][1])),
                    nose=(float(kps[2][0]), float(kps[2][1])),
                    mouth_left=(float(kps[3][0]), float(kps[3][1])),
                    mouth_right=(float(kps[4][0]), float(kps[4][1])),
                )
                aligned = self.align_face(image, landmarks)

            results.append(
                DetectedFace(
                    bbox=bbox,
                    confidence=score,
                    landmarks=landmarks,
                    aligned_face=aligned,
                )
            )

        logger.debug(
            "retinaface_detection_complete",
            faces_found=len(results),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return results


class MediaPipeDetector(BaseFaceDetector):
    """Lightweight CPU-friendly face detector using MediaPipe.

    Best for:
    - Resource-constrained environments
    - When GPU is unavailable
    - Quick prototyping / testing
    """

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        import mediapipe as mp  # type: ignore[import-untyped]

        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=min_detection_confidence,
        )
        logger.info("mediapipe_detector_initialized")

    @property
    def name(self) -> str:
        return "mediapipe"

    def detect(
        self, image: NDArray[np.uint8], threshold: float = 0.5
    ) -> list[DetectedFace]:
        """Detect faces using MediaPipe."""
        start = time.perf_counter()
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = self._detector.process(rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[DetectedFace] = []
        if not mp_results.detections:
            return results

        for det in mp_results.detections:
            score = det.score[0]
            if score < threshold:
                continue

            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width) * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

            # Extract keypoints for landmarks
            kps = det.location_data.relative_keypoints
            landmarks = None
            aligned = None
            if len(kps) >= 5:
                landmarks = FaceLandmarks(
                    left_eye=(kps[0].x * w, kps[0].y * h),
                    right_eye=(kps[1].x * w, kps[1].y * h),
                    nose=(kps[2].x * w, kps[2].y * h),
                    mouth_left=(kps[3].x * w, kps[3].y * h),
                    mouth_right=(kps[4].x * w, kps[4].y * h),
                )
                aligned = self.align_face(image, landmarks)

            if aligned is None:
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    aligned = cv2.resize(crop, ALIGNED_FACE_SIZE)

            results.append(
                DetectedFace(
                    bbox=bbox,
                    confidence=float(score),
                    landmarks=landmarks,
                    aligned_face=aligned,
                )
            )

        logger.debug(
            "mediapipe_detection_complete",
            faces_found=len(results),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return results


def create_detector(
    backend: str = "retinaface", gpu: bool = False
) -> BaseFaceDetector:
    """Factory function to create a face detector.

    Args:
        backend: "retinaface" or "mediapipe"
        gpu: Whether to use GPU acceleration (RetinaFace only)

    Returns:
        Configured face detector instance
    """
    if backend == "retinaface":
        return RetinaFaceDetector(ctx_id=0 if gpu else -1)
    elif backend == "mediapipe":
        return MediaPipeDetector()
    else:
        raise ValueError(f"Unknown detector backend: {backend!r}. Use 'retinaface' or 'mediapipe'.")
