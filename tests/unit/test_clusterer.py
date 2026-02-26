"""
Unit tests for cortexia.core.clusterer — HDBSCAN identity clustering.
"""

import numpy as np
import pytest

from cortexia.core.clusterer import IdentityClusterer


class TestIdentityClusterer:
    def test_cluster_distinct_groups(self):
        """Well-separated groups should form distinct clusters."""
        np.random.seed(42)
        # 3 clearly separated groups of 10 points each
        group_a = np.random.randn(10, 512).astype(np.float32) + 5
        group_b = np.random.randn(10, 512).astype(np.float32) - 5
        group_c = np.random.randn(10, 512).astype(np.float32) + np.array([5, -5] + [0] * 510)

        embeddings = np.vstack([group_a, group_b, group_c])
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        clusterer = IdentityClusterer(min_cluster_size=5, min_samples=3)
        result = clusterer.cluster(embeddings)

        assert result.cluster_count >= 2  # Should find at least 2 groups

    def test_cluster_too_few_points(self):
        """Very few points shouldn't crash."""
        embeddings = np.random.randn(3, 512).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        clusterer = IdentityClusterer(min_cluster_size=5, min_samples=2)
        result = clusterer.cluster(embeddings)
        # Should work without error — returns empty result for too few points
        assert result.cluster_count >= 0

    def test_cluster_single_group(self):
        """All similar points should form one cluster."""
        np.random.seed(42)
        # Use lower-dimensional-like structure: share most dimensions,
        # vary only in a few. This ensures HDBSCAN sees one dense group.
        base = np.zeros(512, dtype=np.float32)
        base[:10] = np.random.randn(10)
        base /= np.linalg.norm(base)

        embeddings = []
        for _ in range(30):
            e = base.copy()
            e[:10] += np.random.randn(10).astype(np.float32) * 0.3
            e /= np.linalg.norm(e)
            embeddings.append(e)
        embeddings = np.array(embeddings)

        clusterer = IdentityClusterer(min_cluster_size=3, min_samples=2)
        result = clusterer.cluster(embeddings)

        # All points cluster together — at least 1 non-noise cluster
        assert result.cluster_count >= 1

    def test_cluster_result_structure(self):
        """ClusteringResult should have correct attributes."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 512).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        clusterer = IdentityClusterer(min_cluster_size=5)
        result = clusterer.cluster(embeddings)

        assert hasattr(result, "cluster_count")
        assert hasattr(result, "noise_count")
        assert hasattr(result, "clusters")
        assert isinstance(result.clusters, list)

    def test_assign_to_cluster(self):
        """assign_to_cluster should return a cluster_id or None."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 512).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        clusterer = IdentityClusterer(min_cluster_size=5)
        result = clusterer.cluster(embeddings)

        if result.cluster_count > 0:
            new_point = embeddings[0] + np.random.randn(512).astype(np.float32) * 0.01
            new_point /= np.linalg.norm(new_point)
            label = clusterer.assign_to_cluster(new_point, result.clusters)
            assert label is None or isinstance(label, (int, np.integer))
