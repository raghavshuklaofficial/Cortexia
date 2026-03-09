"""
HDBSCAN-based identity discovery.

Groups unknown face embeddings into clusters without needing to know
how many people there are. Useful for answering "who are the unknown
faces that keep showing up?" without any enrollment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FaceCluster:
    """A discovered cluster of face embeddings."""

    cluster_id: int
    member_indices: list[int]  # Indices into the input embedding array
    centroid: NDArray[np.float32]
    member_count: int
    is_noise: bool = False  # HDBSCAN label -1 = noise

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "member_count": self.member_count,
            "is_noise": self.is_noise,
            "centroid_norm": float(np.linalg.norm(self.centroid)),
        }


@dataclass
class ClusteringResult:
    """Result from the HDBSCAN clustering operation."""

    clusters: list[FaceCluster] = field(default_factory=list)
    noise_count: int = 0
    total_processed: int = 0
    processing_time_ms: float = 0.0

    @property
    def cluster_count(self) -> int:
        """Number of real clusters (excluding noise)."""
        return sum(1 for c in self.clusters if not c.is_noise)

    def to_dict(self) -> dict:
        return {
            "cluster_count": self.cluster_count,
            "noise_count": self.noise_count,
            "total_processed": self.total_processed,
            "clusters": [c.to_dict() for c in self.clusters if not c.is_noise],
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class IdentityClusterer:
    """HDBSCAN clustering on face embeddings.

    Groups faces by identity without knowing how many distinct people exist.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
    ) -> None:
        """Initialize the clusterer.

        Args:
            min_cluster_size: Minimum faces to form a cluster (identity)
            min_samples: Core point density parameter
            metric: Distance metric (euclidean works well for L2-normalized embeddings)
            cluster_selection_epsilon: Merge clusters within this distance
        """
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples
        self._metric = metric
        self._epsilon = cluster_selection_epsilon
        logger.info(
            "identity_clusterer_initialized",
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )

    def cluster(
        self,
        embeddings: NDArray[np.float32],
    ) -> ClusteringResult:
        """Cluster face embeddings into identity groups.

        Args:
            embeddings: (N, 512) array of face embeddings

        Returns:
            ClusteringResult with discovered clusters
        """
        import hdbscan  # type: ignore[import-untyped]

        start = time.perf_counter()

        if len(embeddings) < self._min_cluster_size:
            logger.info("insufficient_embeddings_for_clustering", count=len(embeddings))
            return ClusteringResult(total_processed=len(embeddings))

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            metric=self._metric,
            cluster_selection_epsilon=self._epsilon,
            core_dist_n_jobs=-1,  # Parallelize
        )
        labels = clusterer.fit_predict(embeddings.astype(np.float64))

        # Build cluster objects
        unique_labels = set(labels)
        clusters: list[FaceCluster] = []
        noise_count = 0

        for label in sorted(unique_labels):
            member_mask = labels == label
            member_indices = list(np.where(member_mask)[0])

            if label == -1:
                # Noise points
                noise_count = len(member_indices)
                clusters.append(
                    FaceCluster(
                        cluster_id=-1,
                        member_indices=member_indices,
                        centroid=np.zeros(embeddings.shape[1], dtype=np.float32),
                        member_count=noise_count,
                        is_noise=True,
                    )
                )
                continue

            # Compute cluster centroid (L2-normalized mean)
            member_embeddings = embeddings[member_mask]
            centroid = np.mean(member_embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            clusters.append(
                FaceCluster(
                    cluster_id=int(label),
                    member_indices=member_indices,
                    centroid=centroid.astype(np.float32),
                    member_count=len(member_indices),
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = ClusteringResult(
            clusters=clusters,
            noise_count=noise_count,
            total_processed=len(embeddings),
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            "clustering_complete",
            clusters=result.cluster_count,
            noise=noise_count,
            total=len(embeddings),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return result

    def assign_to_cluster(
        self,
        embedding: NDArray[np.float32],
        clusters: list[FaceCluster],
        threshold: float = 0.45,
    ) -> int | None:
        """Assign a new embedding to the nearest existing cluster.

        Args:
            embedding: New face embedding to assign
            clusters: Existing clusters to match against
            threshold: Minimum cosine similarity to assign

        Returns:
            Cluster ID if assigned, None if no match
        """
        best_sim = -1.0
        best_cluster_id = None

        for cluster in clusters:
            if cluster.is_noise:
                continue
            sim = float(
                np.dot(
                    embedding.astype(np.float32),
                    cluster.centroid.astype(np.float32),
                )
            )
            if sim > best_sim:
                best_sim = sim
                best_cluster_id = cluster.cluster_id

        if best_sim >= threshold and best_cluster_id is not None:
            return best_cluster_id
        return None
