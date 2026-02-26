"""
Unit tests for cortexia.core.recognizer — identity matching & Platt calibration.
"""

import numpy as np
import pytest

from cortexia.core.recognizer import FaceRecognizer, PlattCalibrator, StoredIdentity


class TestPlattCalibrator:
    def test_default_params(self):
        cal = PlattCalibrator()
        # High similarity → high probability
        prob = cal.calibrate(0.95)
        assert prob > 0.9

    def test_low_similarity(self):
        cal = PlattCalibrator()
        # Low similarity → low probability
        prob = cal.calibrate(0.2)
        assert prob < 0.3

    def test_monotonicity(self):
        """Calibrated probability should increase with similarity."""
        cal = PlattCalibrator()
        sims = [0.1, 0.3, 0.5, 0.7, 0.9]
        probs = [cal.calibrate(s) for s in sims]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_bounded_output(self):
        """Output should always be in [0, 1]."""
        cal = PlattCalibrator()
        for sim in np.linspace(-1, 1, 100):
            prob = cal.calibrate(float(sim))
            assert 0 <= prob <= 1

    def test_custom_params(self):
        cal = PlattCalibrator(a=-10.0, b=5.0)
        prob = cal.calibrate(0.8)
        assert 0 < prob < 1


class TestFaceRecognizer:
    def test_empty_gallery(self, sample_embedding):
        rec = FaceRecognizer()
        match = rec.recognize(sample_embedding)
        assert match.is_known is False
        assert match.identity_name == "unknown"
        assert match.confidence == 0.0

    def test_enroll_and_recognize(self, sample_embedding):
        rec = FaceRecognizer()
        identity = StoredIdentity(
            identity_id=1, name="Alice", embeddings=[sample_embedding]
        )
        rec.add_to_gallery(identity)

        match = rec.recognize(sample_embedding)
        assert match is not None
        assert match.identity_id == 1
        assert match.identity_name == "Alice"
        assert match.confidence > 0.9
        assert match.is_known is True

    def test_recognize_different_person(self, sample_embedding):
        rec = FaceRecognizer()
        identity = StoredIdentity(
            identity_id=1, name="Alice", embeddings=[sample_embedding]
        )
        rec.add_to_gallery(identity)

        # Create a very different embedding
        other = -sample_embedding  # opposite direction
        match = rec.recognize(other)
        # Should be unknown with low confidence
        assert match.is_known is False

    def test_multiple_identities(self):
        rec = FaceRecognizer()

        emb_a = np.random.randn(512).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)
        emb_b = np.random.randn(512).astype(np.float32)
        emb_b /= np.linalg.norm(emb_b)

        rec.add_to_gallery(StoredIdentity(identity_id=1, name="Alice", embeddings=[emb_a]))
        rec.add_to_gallery(StoredIdentity(identity_id=2, name="Bob", embeddings=[emb_b]))

        # Should recognize each correctly
        match_a = rec.recognize(emb_a)
        assert match_a is not None
        assert match_a.identity_id == 1

        match_b = rec.recognize(emb_b)
        assert match_b is not None
        assert match_b.identity_id == 2

    def test_clear_gallery(self, sample_embedding):
        rec = FaceRecognizer()
        identity = StoredIdentity(
            identity_id=1, name="Alice", embeddings=[sample_embedding]
        )
        rec.add_to_gallery(identity)
        rec.load_gallery([])  # Clear gallery
        match = rec.recognize(sample_embedding)
        assert match.is_known is False
        assert match.identity_name == "unknown"

    def test_cosine_similarity_self(self, sample_embedding):
        from cortexia.core.embedder import FaceEmbedder

        sim = FaceEmbedder.cosine_similarity(sample_embedding, sample_embedding)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        from cortexia.core.embedder import FaceEmbedder

        a = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(512, dtype=np.float32)
        b[1] = 1.0
        sim = FaceEmbedder.cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-5)
