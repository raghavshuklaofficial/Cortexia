"""
ArcFace embedding extraction (InsightFace).

512-d L2-normalized vectors. Two faces of the same person typically
have cosine similarity > 0.45. Replaces the old 128-d face_recognition
library approach with something that actually works well.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from numpy.typing import NDArray

import structlog

logger = structlog.get_logger(__name__)

# Standard input size for ArcFace models
FACE_INPUT_SIZE = (112, 112)


class FaceEmbedder:
    """ArcFace embedding extraction. Returns 512-d L2-normalized vectors."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        quantize: bool = False,
    ) -> None:
        """Initialize the ArcFace embedding model.

        Args:
            model_name: InsightFace model pack name
            ctx_id: -1 for CPU, 0+ for GPU device ID
            quantize: If True, return float16 embeddings (halves storage)
        """
        import insightface  # type: ignore[import-untyped]

        # Load recognition model directly to avoid FaceAnalysis assertion
        # that requires detection model to be present.
        self._app = insightface.app.FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if ctx_id >= 0
                else ["CPUExecutionProvider"]
            ),
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # Extract the recognition model directly for standalone embedding
        self._rec_model = None
        for task, model in self._app.models.items():
            if task == "recognition":
                self._rec_model = model
                break

        self._quantize = quantize
        self._embedding_dim = 512
        logger.info(
            "face_embedder_initialized",
            model=model_name,
            gpu=ctx_id >= 0,
            quantize=quantize,
            embedding_dim=self._embedding_dim,
        )

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        return self._embedding_dim

    def extract(self, aligned_face: NDArray[np.uint8]) -> NDArray:
        """Extract embedding from a single aligned face crop.

        Args:
            aligned_face: BGR uint8 image, ideally 112x112 from alignment

        Returns:
            L2-normalized 512-d embedding (float32, or float16 if quantize=True)
        """
        start = time.perf_counter()

        # Ensure correct input size
        if aligned_face.shape[:2] != FACE_INPUT_SIZE:
            aligned_face = cv2.resize(aligned_face, FACE_INPUT_SIZE)

        # Use InsightFace's recognition model directly
        # (this avoids running detection again on an already-cropped face)
        if self._rec_model is not None:
            embedding = self._rec_model.get_feat(aligned_face).flatten()
        else:
            # Fallback: run full pipeline on the aligned crop
            faces = self._app.get(aligned_face)
            if not faces:
                logger.warning("no_face_in_aligned_crop")
                return np.zeros(self._embedding_dim, dtype=np.float32)
            embedding = faces[0].normed_embedding

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("embedding_extracted", elapsed_ms=round(elapsed_ms, 2))

        if self._quantize:
            return embedding.astype(np.float16)
        return embedding.astype(np.float32)

    def extract_batch(
        self, aligned_faces: list[NDArray[np.uint8]]
    ) -> list[NDArray[np.float32]]:
        """Extract embeddings for multiple faces. Just loops for now.
        TODO: actual batch inference would be faster
        """
        if not aligned_faces:
            return []

        start = time.perf_counter()
        embeddings = [self.extract(face) for face in aligned_faces]
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "batch_embeddings_extracted",
            count=len(embeddings),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return embeddings

    @staticmethod
    def cosine_similarity(
        emb1: NDArray[np.float32], emb2: NDArray[np.float32]
    ) -> float:
        """Compute cosine similarity between two L2-normalized embeddings.

        Since embeddings are L2-normalized, cosine similarity = dot product.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Similarity score in [-1, 1], where > 0.45 typically indicates same person
        """
        return float(np.dot(emb1.astype(np.float32), emb2.astype(np.float32)))
