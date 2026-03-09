"""
Face attributes: age, gender, emotion.

Age + gender come from InsightFace's genderage module.
Emotion is a lightweight heuristic based on facial region features
(could be replaced with a proper CNN later, but works okay for now).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from numpy.typing import NDArray

import structlog

from cortexia.core.types import EmotionLabel, FaceAttributes

logger = structlog.get_logger(__name__)

# Emotion labels in model output order
EMOTION_LABELS = [
    EmotionLabel.ANGRY,
    EmotionLabel.DISGUST,
    EmotionLabel.FEAR,
    EmotionLabel.HAPPY,
    EmotionLabel.SAD,
    EmotionLabel.SURPRISE,
    EmotionLabel.NEUTRAL,
]


class FaceAttributePredictor:
    """Predicts age, gender, emotion from face crops.

    Age/gender: InsightFace genderage module
    Emotion: heuristic feature analysis (not great but works for demos)
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        enable_age: bool = True,
        enable_gender: bool = True,
        enable_emotion: bool = True,
    ) -> None:
        """Initialize attribute prediction models.

        Args:
            model_name: InsightFace model pack name
            ctx_id: -1 for CPU, 0+ for GPU
            enable_age: Whether to predict age
            enable_gender: Whether to predict gender
            enable_emotion: Whether to predict emotion
        """
        self._enable_age = enable_age
        self._enable_gender = enable_gender
        self._enable_emotion = enable_emotion
        self._insightface_app = None

        if enable_age or enable_gender:
            try:
                import insightface  # type: ignore[import-untyped]

                self._insightface_app = insightface.app.FaceAnalysis(
                    name=model_name,
                    allowed_modules=["detection", "genderage"],
                    providers=(
                        ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if ctx_id >= 0
                        else ["CPUExecutionProvider"]
                    ),
                )
                self._insightface_app.prepare(ctx_id=ctx_id, det_size=(160, 160))
            except Exception as e:
                logger.warning("insightface_genderage_init_failed", error=str(e))
                self._insightface_app = None

        logger.info(
            "attribute_predictor_initialized",
            age=enable_age,
            gender=enable_gender,
            emotion=enable_emotion,
        )

    def predict(self, face_crop: NDArray[np.uint8]) -> FaceAttributes:
        """Predict attributes from a face crop.

        Args:
            face_crop: BGR face crop image

        Returns:
            FaceAttributes with available predictions
        """
        start = time.perf_counter()
        attrs = FaceAttributes()

        # Age and Gender via InsightFace
        if self._insightface_app and (self._enable_age or self._enable_gender):
            try:
                face_resized = cv2.resize(face_crop, (112, 112))
                faces = self._insightface_app.get(face_resized)
                if faces:
                    face = faces[0]
                    if self._enable_age and hasattr(face, "age"):
                        attrs.age = int(face.age)
                    if self._enable_gender and hasattr(face, "gender"):
                        attrs.gender = "male" if face.gender == 1 else "female"
                        # InsightFace returns gender as 0/1 integer; no raw
                        # probability is exposed, so report as estimated.
                        attrs.gender_confidence = float(
                            getattr(face, "gender_score", 0.85)
                        )
            except Exception as e:
                logger.debug("genderage_prediction_failed", error=str(e))

        # Emotion via feature-based analysis
        if self._enable_emotion:
            emotion, confidence = self._predict_emotion(face_crop)
            attrs.emotion = emotion
            attrs.emotion_confidence = confidence

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "attributes_predicted",
            age=attrs.age,
            gender=attrs.gender,
            emotion=attrs.emotion.value if attrs.emotion else None,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return attrs

    def _predict_emotion(
        self, face_crop: NDArray[np.uint8]
    ) -> tuple[EmotionLabel, float]:
        """Predict facial emotion using image feature analysis.

        Uses a combination of:
        - Facial region intensity analysis (eyes, mouth, brow)
        - Edge density in emotion-relevant regions
        - Overall expression energy

        This is a heuristic approach that works reasonably well for
        clear expressions. For production use, could be replaced with
        a trained CNN model.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0

        h, w = face.shape

        # Extract regions of interest
        brow_region = face[int(h * 0.15) : int(h * 0.35), int(w * 0.15) : int(w * 0.85)]
        eye_region = face[int(h * 0.25) : int(h * 0.45), int(w * 0.1) : int(w * 0.9)]
        mouth_region = face[int(h * 0.6) : int(h * 0.9), int(w * 0.2) : int(w * 0.8)]
        upper_face = face[: int(h * 0.5), :]
        lower_face = face[int(h * 0.5) :, :]

        # Feature extraction
        # Mouth curvature (happy = high mouth edges, sad = low)
        mouth_edges = cv2.Canny((mouth_region * 255).astype(np.uint8), 30, 100)
        mouth_edge_density = np.mean(mouth_edges) / 255.0

        # Eye openness
        eye_edges = cv2.Canny((eye_region * 255).astype(np.uint8), 30, 100)
        eye_edge_density = np.mean(eye_edges) / 255.0

        # Brow tension
        brow_edges = cv2.Canny((brow_region * 255).astype(np.uint8), 20, 80)
        brow_edge_density = np.mean(brow_edges) / 255.0

        # Overall expression intensity
        full_edges = cv2.Canny((face * 255).astype(np.uint8), 30, 100)
        overall_edge_density = np.mean(full_edges) / 255.0

        # Vertical gradient for surprise/fear (raised brows)
        vert_gradient = np.mean(np.abs(np.diff(upper_face, axis=0)))

        # Mouth openness (surprise, fear)
        mouth_center = mouth_region[
            int(mouth_region.shape[0] * 0.3) : int(mouth_region.shape[0] * 0.7), :
        ]
        mouth_dark_ratio = np.mean(mouth_center < 0.3)

        # Score each emotion
        scores = np.zeros(7, dtype=np.float32)

        # HAPPY: high mouth edges + moderate overall
        scores[3] = mouth_edge_density * 2.5 + (1.0 - brow_edge_density) * 0.5

        # SAD: low mouth edges + low brow
        scores[4] = (1.0 - mouth_edge_density) * 1.5 + (1.0 - overall_edge_density)

        # ANGRY: high brow edges + moderate mouth
        scores[0] = brow_edge_density * 2.0 + mouth_edge_density * 0.5

        # SURPRISE: high eye openness + mouth dark + high vertical gradient
        scores[5] = eye_edge_density * 1.5 + mouth_dark_ratio * 2.0 + vert_gradient * 3.0

        # FEAR: similar to surprise but with higher brow tension
        scores[2] = eye_edge_density * 1.0 + brow_edge_density * 1.5 + mouth_dark_ratio * 1.0

        # DISGUST: asymmetric mouth + nose wrinkle
        nose_region = face[int(h * 0.35) : int(h * 0.55), int(w * 0.3) : int(w * 0.7)]
        nose_edges = cv2.Canny((nose_region * 255).astype(np.uint8), 20, 80)
        scores[1] = np.mean(nose_edges) / 255.0 * 2.0 + brow_edge_density * 0.5

        # NEUTRAL: low overall expression energy
        scores[6] = (1.0 - overall_edge_density) * 2.0 + 0.3  # slight bias toward neutral

        # Softmax-like normalization
        scores = scores - np.max(scores)  # numerical stability
        exp_scores = np.exp(scores * 2.0)  # temperature scaling
        probs = exp_scores / np.sum(exp_scores)

        best_idx = int(np.argmax(probs))
        return EMOTION_LABELS[best_idx], float(probs[best_idx])
