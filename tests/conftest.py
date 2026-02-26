"""
Test fixtures and configuration for CORTEXIA test suite.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio

# ─── Async event loop ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create a session-scoped event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─── Synthetic face data ─────────────────────────────────────────────────


@pytest.fixture
def sample_bgr_frame() -> np.ndarray:
    """A synthetic 640x480 BGR frame with a face-like bright region."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a face-like ellipse region
    center = (320, 200)
    for y in range(150, 280):
        for x in range(270, 370):
            dist = ((x - center[0]) / 50) ** 2 + ((y - center[1]) / 60) ** 2
            if dist < 1.0:
                frame[y, x] = [180, 180, 200]  # skin-tone BGR
    return frame


@pytest.fixture
def sample_face_crop() -> np.ndarray:
    """A 112x112 aligned face crop (synthetic)."""
    crop = np.random.randint(100, 200, (112, 112, 3), dtype=np.uint8)
    return crop


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """A random L2-normalized 512-d embedding."""
    vec = np.random.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def sample_embedding_batch() -> np.ndarray:
    """Batch of 10 random embeddings."""
    vecs = np.random.randn(10, 512).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def identity_id() -> str:
    """A random UUID for test identity."""
    return str(uuid.uuid4())


# ─── Mock pipeline components ────────────────────────────────────────────


@pytest.fixture
def mock_detector():
    """Mock face detector returning a single detection."""
    from cortexia.core.types import BoundingBox, DetectedFace

    det = MagicMock()
    det.detect.return_value = [
        DetectedFace(
            bbox=BoundingBox(x1=100, y1=80, x2=200, y2=220),
            confidence=0.98,
            aligned_face=np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
        )
    ]
    return det


@pytest.fixture
def mock_embedder(sample_embedding):
    """Mock embedder returning a fixed embedding."""
    emb = MagicMock()
    emb.extract.return_value = sample_embedding
    emb.extract_batch.return_value = sample_embedding.reshape(1, -1)
    return emb
