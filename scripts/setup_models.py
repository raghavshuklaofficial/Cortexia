#!/usr/bin/env python3
"""
CORTEXIA — Model Setup Script.

Downloads and caches the InsightFace buffalo_l model pack
required by the Trust Pipeline (RetinaFace detector + ArcFace embedder).

Usage:
    python scripts/setup_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("  CORTEXIA — Model Setup")
    print("=" * 60)
    print()

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("ERROR: insightface is not installed.")
        print("Run: pip install insightface onnxruntime")
        sys.exit(1)

    print("[1/3] Downloading buffalo_l model pack...")
    print("      This includes RetinaFace (detection) + ArcFace (recognition)")
    print("      First run may take 2-5 minutes depending on connection.")
    print()

    try:
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("      ✓ buffalo_l downloaded and verified")
    except Exception as e:
        print(f"      ✗ Download failed: {e}")
        sys.exit(1)

    print()
    print("[2/3] Verifying model components...")

    models_found = []
    if hasattr(app, "models"):
        for model in app.models.values():
            name = getattr(model, "taskname", type(model).__name__)
            models_found.append(name)
            print(f"      ✓ {name}")
    else:
        print("      ✓ Models loaded (legacy API)")

    print()
    print("[3/3] Testing inference...")

    import numpy as np

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a face-like region
    test_frame[150:300, 250:400] = 180

    faces = app.get(test_frame)
    print(f"      ✓ Inference OK (detected {len(faces)} faces in test image)")

    print()
    print("=" * 60)
    print("  Setup complete! Models cached in ~/.insightface/")
    print("=" * 60)


if __name__ == "__main__":
    main()
