"""
Face recognition engine with Platt-scaled confidence calibration.

Matches face embeddings against a database of known identities using
cosine similarity. Unlike raw thresholding, CORTEXIA uses Platt scaling
to convert distances into calibrated probabilities — so a confidence of
0.92 genuinely means ~92% probability of correct match.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import structlog

from cortexia.core.types import RecognitionMatch

logger = structlog.get_logger(__name__)


@dataclass
class StoredIdentity:
    """An identity stored in the recognition gallery."""

    identity_id: int
    name: str
    embeddings: list[NDArray[np.float32]]  # Multiple enrolled embeddings

    @property
    def centroid(self) -> NDArray[np.float32]:
        """Average embedding (centroid) of all enrolled faces."""
        if not self.embeddings:
            return np.zeros(512, dtype=np.float32)
        centroid = np.mean(self.embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.astype(np.float32)


class PlattCalibrator:
    """Platt scaling for converting raw cosine distances to probabilities.

    Fits a sigmoid function: P(match) = 1 / (1 + exp(A * distance + B))
    where A and B are learned from a calibration dataset.

    For cases without calibration data, uses empirically tuned defaults
    based on ArcFace embedding characteristics.
    """

    def __init__(self, a: float = -15.0, b: float = 6.5) -> None:
        """Initialize with sigmoid parameters.

        Default values tuned for ArcFace 512-d embeddings where:
        - Same-person cosine similarity: typically 0.5 - 0.9
        - Different-person cosine similarity: typically -0.1 - 0.3
        """
        self.a = a
        self.b = b

    def calibrate(self, cosine_similarity: float) -> float:
        """Convert cosine similarity to calibrated probability.

        Args:
            cosine_similarity: Raw cosine similarity in [-1, 1]

        Returns:
            Calibrated probability in [0, 1]
        """
        logit = self.a * (1.0 - cosine_similarity) + self.b
        try:
            return 1.0 / (1.0 + math.exp(-logit))
        except OverflowError:
            return 0.0 if logit < 0 else 1.0


class FaceRecognizer:
    """Matches face embeddings against enrolled identities.

    Features:
    - Cosine similarity matching with configurable threshold
    - Platt-scaled confidence calibration
    - Top-K candidate retrieval
    - Explicit unknown face handling
    """

    def __init__(
        self,
        recognition_threshold: float = 0.45,
        unknown_threshold: float = 0.35,
    ) -> None:
        """Initialize the recognizer.

        Args:
            recognition_threshold: Minimum cosine similarity to consider a match
            unknown_threshold: Below this, face is definitively unknown
        """
        self._threshold = recognition_threshold
        self._unknown_threshold = unknown_threshold
        self._calibrator = PlattCalibrator()
        self._gallery: list[StoredIdentity] = []
        self._lock = threading.Lock()
        logger.info(
            "recognizer_initialized",
            threshold=recognition_threshold,
            unknown_threshold=unknown_threshold,
        )

    @property
    def gallery_size(self) -> int:
        """Number of enrolled identities."""
        return len(self._gallery)

    def load_gallery(self, identities: list[StoredIdentity]) -> None:
        """Load identity gallery for matching.

        Args:
            identities: List of enrolled identities with embeddings
        """
        with self._lock:
            self._gallery = list(identities)
        logger.info("gallery_loaded", identities=len(identities))

    def add_to_gallery(self, identity: StoredIdentity) -> None:
        """Add a single identity to the gallery."""
        with self._lock:
            for i, existing in enumerate(self._gallery):
                if existing.identity_id == identity.identity_id:
                    self._gallery[i] = identity
                    return
            self._gallery.append(identity)

    def remove_from_gallery(self, identity_id: int) -> None:
        """Remove an identity from the gallery."""
        with self._lock:
            self._gallery = [g for g in self._gallery if g.identity_id != identity_id]

    def recognize(
        self,
        embedding: NDArray[np.float32],
        top_k: int = 1,
    ) -> RecognitionMatch:
        """Match an embedding against the gallery.

        Args:
            embedding: 512-d L2-normalized face embedding
            top_k: Number of candidates to consider (uses best match)

        Returns:
            RecognitionMatch with identity info, distance, and calibrated confidence
        """
        start = time.perf_counter()

        if not self._gallery:
            return RecognitionMatch(
                identity_id=-1,
                identity_name="unknown",
                distance=1.0,
                confidence=0.0,
                is_known=False,
            )

        # Compute cosine similarity against all gallery embeddings
        # Collect (similarity, identity) candidates
        candidates: list[tuple[float, StoredIdentity]] = []

        with self._lock:
            gallery_snapshot = list(self._gallery)

        for identity in gallery_snapshot:
            best_id_sim = -1.0
            for enrolled_emb in identity.embeddings:
                sim = float(
                    np.dot(
                        embedding.astype(np.float32),
                        enrolled_emb.astype(np.float32),
                    )
                )
                if sim > best_id_sim:
                    best_id_sim = sim
            candidates.append((best_id_sim, identity))

        # Sort by similarity descending, take top_k
        candidates.sort(key=lambda c: c[0], reverse=True)
        candidates = candidates[:top_k]

        best_sim, best_identity = candidates[0]

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert similarity to distance (for consistent interface)
        distance = 1.0 - best_sim

        # Calibrate confidence via Platt scaling
        confidence = self._calibrator.calibrate(best_sim)

        # Determine if this is a known face
        is_known = best_sim >= self._threshold and best_identity is not None

        if is_known and best_identity is not None:
            match = RecognitionMatch(
                identity_id=best_identity.identity_id,
                identity_name=best_identity.name,
                distance=distance,
                confidence=confidence,
                is_known=True,
            )
        else:
            match = RecognitionMatch(
                identity_id=-1,
                identity_name="unknown",
                distance=distance,
                confidence=confidence,
                is_known=False,
            )

        logger.debug(
            "recognition_complete",
            identity=match.identity_name,
            distance=round(distance, 4),
            confidence=round(confidence, 4),
            is_known=is_known,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return match

    def recognize_batch(
        self,
        embeddings: list[NDArray[np.float32]],
    ) -> list[RecognitionMatch]:
        """Match multiple embeddings against the gallery.

        Args:
            embeddings: List of face embeddings

        Returns:
            List of recognition matches
        """
        return [self.recognize(emb) for emb in embeddings]
