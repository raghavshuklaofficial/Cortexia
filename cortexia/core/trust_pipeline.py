"""
CORTEXIA Trust Pipeline — The Core Intelligence Chain.

The Trust Pipeline is what makes CORTEXIA unique. Every face passes
through a multi-stage verification chain:

    Detection → Alignment → Liveness Check → Embedding → Recognition → Attributes

Each stage produces a result and feeds the next. Faces failing liveness
check are flagged (and optionally blocked from recognition). A composite
trust_score aggregates confidence across all stages.

The pipeline is configurable — stages can be enabled/disabled independently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import structlog

from cortexia.config import Settings
from cortexia.core.clusterer import IdentityClusterer
from cortexia.core.detector import BaseFaceDetector, create_detector
from cortexia.core.embedder import FaceEmbedder
from cortexia.core.models.antispoof import AntiSpoofDetector
from cortexia.core.models.attributes import FaceAttributePredictor
from cortexia.core.recognizer import FaceRecognizer
from cortexia.core.types import (
    DetectedFace,
    FaceAnalysis,
    FrameAnalysis,
    LivenessVerdict,
)

logger = structlog.get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the Trust Pipeline stages."""

    detection_enabled: bool = True
    liveness_enabled: bool = True
    recognition_enabled: bool = True
    attributes_enabled: bool = True
    clustering_enabled: bool = True

    detection_backend: str = "retinaface"
    detection_threshold: float = 0.5
    gpu_enabled: bool = False

    antispoof_threshold: float = 0.7
    recognition_threshold: float = 0.45
    unknown_threshold: float = 0.35

    max_faces_per_frame: int = 20

    # Trust score weights (must sum to 1.0)
    trust_weight_detection: float = 0.20
    trust_weight_liveness: float = 0.40
    trust_weight_recognition: float = 0.40

    @classmethod
    def from_settings(cls, settings: Settings) -> PipelineConfig:
        """Create pipeline config from application settings."""
        return cls(
            liveness_enabled=settings.antispoof_enabled,
            recognition_enabled=True,
            attributes_enabled=settings.attributes_enabled,
            detection_backend=settings.model_backend,
            detection_threshold=settings.detection_threshold,
            antispoof_threshold=settings.antispoof_threshold,
            recognition_threshold=settings.recognition_threshold,
            unknown_threshold=settings.unknown_threshold,
            max_faces_per_frame=settings.max_faces_per_frame,
            trust_weight_detection=settings.trust_weight_detection,
            trust_weight_liveness=settings.trust_weight_liveness,
            trust_weight_recognition=settings.trust_weight_recognition,
        )


class TrustPipeline:
    """The CORTEXIA Trust Pipeline.

    Orchestrates the full face intelligence chain:
    1. DETECT — Find faces with bounding boxes and landmarks
    2. ALIGN — Warp faces to standard 112x112 using landmarks
    3. VERIFY — Anti-spoofing liveness check
    4. EMBED — Extract 512-d ArcFace embeddings
    5. RECOGNIZE — Match against enrolled identities
    6. ANALYZE — Predict age, gender, emotion

    Each face receives a trust_score computed as:
        trust = w1*detection_conf + w2*liveness_conf + w3*recognition_conf
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize all pipeline components.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self._config = config or PipelineConfig()
        self._initialized = False

        # Components (lazy-initialized)
        self._detector: BaseFaceDetector | None = None
        self._embedder: FaceEmbedder | None = None
        self._antispoof: AntiSpoofDetector | None = None
        self._recognizer: FaceRecognizer | None = None
        self._attributes: FaceAttributePredictor | None = None
        self._clusterer: IdentityClusterer | None = None

    def initialize(self) -> None:
        """Initialize all ML components. Call once at startup."""
        if self._initialized:
            return

        logger.info("trust_pipeline_initializing")
        start = time.perf_counter()

        # Stage 1 & 2: Detector (includes alignment)
        gpu = self._config.gpu_enabled
        ctx_id = 0 if gpu else -1
        self._detector = create_detector(self._config.detection_backend, gpu=gpu)

        # Stage 3: Anti-spoofing
        if self._config.liveness_enabled:
            self._antispoof = AntiSpoofDetector(
                threshold=self._config.antispoof_threshold,
            )

        # Stage 4: Embedder
        self._embedder = FaceEmbedder(ctx_id=ctx_id)

        # Stage 5: Recognizer
        self._recognizer = FaceRecognizer(
            recognition_threshold=self._config.recognition_threshold,
            unknown_threshold=self._config.unknown_threshold,
        )

        # Stage 6: Attributes
        if self._config.attributes_enabled:
            self._attributes = FaceAttributePredictor(ctx_id=ctx_id)

        # Clusterer
        if self._config.clustering_enabled:
            self._clusterer = IdentityClusterer()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._initialized = True
        logger.info(
            "trust_pipeline_ready",
            elapsed_ms=round(elapsed_ms, 2),
            liveness=self._config.liveness_enabled,
            attributes=self._config.attributes_enabled,
            backend=self._config.detection_backend,
        )

    @property
    def recognizer(self) -> FaceRecognizer:
        """Access the recognizer for gallery management."""
        if self._recognizer is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._recognizer

    @property
    def embedder(self) -> FaceEmbedder:
        """Access the embedder."""
        if self._embedder is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._embedder

    @property
    def clusterer(self) -> IdentityClusterer:
        """Access the identity clusterer."""
        if self._clusterer is None:
            raise RuntimeError("Pipeline not initialized or clustering disabled.")
        return self._clusterer

    def process_image(self, image: NDArray[np.uint8]) -> FrameAnalysis:
        """Run the full Trust Pipeline on a single image.

        Args:
            image: BGR uint8 image

        Returns:
            FrameAnalysis with results for all detected faces
        """
        if not self._initialized:
            self.initialize()

        start = time.perf_counter()
        h, w = image.shape[:2]

        # ════════════ Stage 1: DETECT ════════════
        assert self._detector is not None
        detected_faces = self._detector.detect(
            image, threshold=self._config.detection_threshold
        )

        # Limit faces per frame
        if len(detected_faces) > self._config.max_faces_per_frame:
            detected_faces.sort(key=lambda f: f.confidence, reverse=True)
            detected_faces = detected_faces[: self._config.max_faces_per_frame]

        # Process each face through remaining pipeline stages
        face_analyses: list[FaceAnalysis] = []

        for face in detected_faces:
            analysis = self._process_single_face(image, face)
            face_analyses.append(analysis)

        total_ms = (time.perf_counter() - start) * 1000

        frame_result = FrameAnalysis(
            faces=face_analyses,
            total_processing_time_ms=total_ms,
            frame_width=w,
            frame_height=h,
        )

        logger.info(
            "frame_processed",
            faces=frame_result.face_count,
            known=frame_result.known_count,
            spoofs=frame_result.spoof_count,
            elapsed_ms=round(total_ms, 2),
        )
        return frame_result

    def _process_single_face(
        self,
        image: NDArray[np.uint8],
        face: DetectedFace,
    ) -> FaceAnalysis:
        """Run pipeline stages 2-6 on a single detected face."""
        face_start = time.perf_counter()

        analysis = FaceAnalysis(face=face)
        detection_conf = face.confidence

        # ════════════ Stage 2: ALIGN (done during detection) ════════════
        aligned = face.aligned_face
        if aligned is None:
            # Fallback: crop from bounding box
            import cv2

            bb = face.bbox
            crop = image[bb.y1 : bb.y2, bb.x1 : bb.x2]
            if crop.size > 0:
                aligned = cv2.resize(crop, (112, 112))
            else:
                analysis.trust_score = 0.0
                return analysis

        # ════════════ Stage 3: LIVENESS CHECK ════════════
        liveness_conf = 1.0
        if self._antispoof is not None and self._config.liveness_enabled:
            liveness_result = self._antispoof.detect(aligned)
            analysis.liveness = liveness_result
            liveness_conf = liveness_result.confidence

        # ════════════ Stage 4: EMBED ════════════
        assert self._embedder is not None
        embedding = self._embedder.extract(aligned)
        analysis.embedding = embedding

        # ════════════ Stage 5: RECOGNIZE ════════════
        recognition_conf = 0.0
        assert self._recognizer is not None
        match = self._recognizer.recognize(embedding)
        analysis.recognition = match
        recognition_conf = match.confidence

        # ════════════ Stage 6: ATTRIBUTES ════════════
        if self._attributes is not None and self._config.attributes_enabled:
            attrs = self._attributes.predict(aligned)
            analysis.attributes = attrs

        # ════════════ COMPUTE TRUST SCORE ════════════
        # Weighted combination of all confidence signals (from config)
        trust = (
            self._config.trust_weight_detection * detection_conf
            + self._config.trust_weight_liveness * liveness_conf
            + self._config.trust_weight_recognition * recognition_conf
        )

        # Penalize spoofs heavily
        if (
            analysis.liveness is not None
            and analysis.liveness.verdict == LivenessVerdict.SPOOF
        ):
            trust *= 0.3

        analysis.trust_score = min(1.0, max(0.0, trust))
        analysis.processing_time_ms = (time.perf_counter() - face_start) * 1000

        return analysis

    def process_face_crop(self, face_crop: NDArray[np.uint8]) -> FaceAnalysis:
        """Run the pipeline on a pre-cropped face image.

        Skips detection — useful for uploaded face crops or API endpoints
        that receive already-cropped faces.

        Args:
            face_crop: BGR face crop image

        Returns:
            FaceAnalysis with all available results
        """
        if not self._initialized:
            self.initialize()

        import cv2

        from cortexia.core.types import BoundingBox

        h, w = face_crop.shape[:2]
        face = DetectedFace(
            bbox=BoundingBox(x1=0, y1=0, x2=w, y2=h),
            confidence=1.0,
            aligned_face=cv2.resize(face_crop, (112, 112)),
        )

        return self._process_single_face(face_crop, face)
