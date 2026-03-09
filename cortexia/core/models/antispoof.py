"""
Anti-spoofing / liveness detection.

Heuristic approach combining FFT frequency analysis, YCrCb color
checks, LBP texture analysis, and moiré pattern detection. No neural
network needed -- everything is computed algorithmically.

Handles: printed photos, screen replay attacks, paper masks.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from numpy.typing import NDArray

import structlog

from cortexia.core.types import LivenessResult, LivenessVerdict

logger = structlog.get_logger(__name__)


class AntiSpoofDetector:
    """Multi-method liveness detector.

    Combines FFT frequency analysis, YCrCb color checks, LBP texture,
    and moiré pattern detection. Each method votes independently, then
    we take a weighted ensemble score.

    TODO: train a small CNN for this instead of heuristics -- the
    frequency analysis is good but the color check has too many false
    positives on certain skin tones
    """

    def __init__(self, threshold: float = 0.7) -> None:
        """Initialize the anti-spoof detector.

        Args:
            threshold: Minimum ensemble confidence to declare LIVE
        """
        self._threshold = threshold
        self._uncertain_threshold = threshold * 0.7
        logger.info(
            "antispoof_detector_initialized",
            threshold=threshold,
        )

    def detect(self, face_crop: NDArray[np.uint8]) -> LivenessResult:
        """Run anti-spoofing analysis on a face crop.

        Args:
            face_crop: BGR face crop (any size, will be resized internally)

        Returns:
            LivenessResult with verdict, confidence, and method description
        """
        start = time.perf_counter()

        # Resize to standard analysis size
        analysis_size = (128, 128)
        face = cv2.resize(face_crop, analysis_size)

        # Run all analysis methods
        fft_score = self._frequency_analysis(face)
        color_score = self._color_analysis(face)
        texture_score = self._texture_analysis(face)
        moire_score = self._moire_detection(face)

        # Weighted ensemble
        weights = [0.30, 0.20, 0.30, 0.20]
        scores = [fft_score, color_score, texture_score, moire_score]
        ensemble_score = sum(w * s for w, s in zip(weights, scores))

        # Determine verdict
        if ensemble_score >= self._threshold:
            verdict = LivenessVerdict.LIVE
        elif ensemble_score >= self._uncertain_threshold:
            verdict = LivenessVerdict.UNCERTAIN
        else:
            verdict = LivenessVerdict.SPOOF

        elapsed_ms = (time.perf_counter() - start) * 1000

        methods = []
        if fft_score < 0.5:
            methods.append("fft_anomaly")
        if color_score < 0.5:
            methods.append("color_anomaly")
        if texture_score < 0.5:
            methods.append("texture_anomaly")
        if moire_score < 0.5:
            methods.append("moire_detected")

        result = LivenessResult(
            verdict=verdict,
            confidence=round(ensemble_score, 4),
            method="+".join(methods) if methods else "all_passed",
        )

        logger.debug(
            "antispoof_analysis",
            verdict=verdict.value,
            confidence=round(ensemble_score, 4),
            fft=round(fft_score, 3),
            color=round(color_score, 3),
            texture=round(texture_score, 3),
            moire=round(moire_score, 3),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return result

    def _frequency_analysis(self, face: NDArray[np.uint8]) -> float:
        """FFT-based check for screen/print artifacts."""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        # Analyze high-frequency energy ratio
        h, w = magnitude.shape
        center = (h // 2, w // 2)
        radius = min(h, w) // 4

        # Create masks for low and high frequency regions
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        low_freq_energy = np.sum(magnitude[dist <= radius])
        high_freq_energy = np.sum(magnitude[dist > radius])
        total = low_freq_energy + high_freq_energy

        if total == 0:
            return 0.5

        # Live faces: balanced frequency distribution
        # Spoofs: concentrated low-freq or anomalous high-freq peaks
        ratio = high_freq_energy / total

        # Empirical range: live faces ~ 0.35-0.55, spoofs ~ 0.15-0.30 or > 0.60
        if 0.25 <= ratio <= 0.60:
            return min(1.0, 0.5 + (0.5 - abs(ratio - 0.42)) * 2)
        else:
            return max(0.0, 1.0 - abs(ratio - 0.42) * 3)

    def _color_analysis(self, face: NDArray[np.uint8]) -> float:
        """Check YCrCb chroma distribution for print/screen artifacts."""
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)

        # Analyze Cr and Cb channels (chroma)
        cr = ycrcb[:, :, 1].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)

        # Skin color typically: Cr in [133, 173], Cb in [77, 127]
        cr_mean = np.mean(cr)
        cb_mean = np.mean(cb)
        cr_std = np.std(cr)
        cb_std = np.std(cb)

        # Live faces have moderate chroma variation
        # Printed: less variation, screens: shifted distribution
        score = 0.5

        # Check if chroma means are in expected skin range
        if 130 <= cr_mean <= 180 and 75 <= cb_mean <= 135:
            score += 0.2

        # Check for natural variation (not too uniform, not too noisy)
        if 8 <= cr_std <= 30 and 5 <= cb_std <= 25:
            score += 0.2

        # Check for color channel correlation (natural in live skin)
        if cr_std > 0 and cb_std > 0:
            correlation = np.corrcoef(cr.flatten(), cb.flatten())[0, 1]
            if -0.5 <= correlation <= 0.5:
                score += 0.1

        return min(1.0, max(0.0, score))

    def _texture_analysis(self, face: NDArray[np.uint8]) -> float:
        """LBP variance check. Live faces have richer micro-texture."""
        # FIXME: this needs better calibration for different camera qualities
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Compute LBP-like texture measure using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()

        # Compute gradient magnitude for texture richness
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_mean = np.mean(gradient_mag)

        # Live faces: moderate to high texture (lap_var ~ 100-800)
        # Printed/screen: lower texture (lap_var < 50) or very high (> 1000 for moiré)
        score = 0.5

        if 30 <= lap_var <= 1000:
            score += 0.3 * min(1.0, lap_var / 200)

        if 10 <= grad_mean <= 80:
            score += 0.2

        return min(1.0, max(0.0, score))

    def _moire_detection(self, face: NDArray[np.uint8]) -> float:
        """Detect Moiré patterns characteristic of screen replay attacks.

        When a camera captures another screen, interference between the
        camera sensor grid and the display pixel grid creates visible
        Moiré patterns — periodic diagonal or wave-like artifacts.
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Detect periodic patterns via autocorrelation
        f_transform = np.fft.fft2(gray)
        power_spectrum = np.abs(f_transform) ** 2

        # Autocorrelation via inverse FFT of power spectrum
        autocorr = np.real(np.fft.ifft2(power_spectrum))
        autocorr = autocorr / autocorr[0, 0]  # Normalize

        # Check for periodic peaks (excluding the center)
        h, w = autocorr.shape
        center_mask = np.ones_like(autocorr, dtype=bool)
        ch, cw = h // 2, w // 2
        r = max(3, min(h, w) // 10)
        y, x = np.ogrid[:h, :w]
        center_region = ((y - 0) ** 2 + (x - 0) ** 2) <= r**2
        center_mask[center_region] = False

        # Also mask corners (which are equivalent to center in FFT)
        for cy, cx in [(0, w), (h, 0), (h, w)]:
            corner_region = ((y - cy) ** 2 + (x - cx) ** 2) <= r**2
            center_mask[corner_region] = False

        # High autocorrelation peaks outside center = periodic pattern = likely Moiré
        max_peak = np.max(autocorr[center_mask]) if np.any(center_mask) else 0

        # Live faces: max_peak typically < 0.15
        # Screen captures: max_peak > 0.25 due to Moiré
        if max_peak < 0.10:
            return 1.0  # No Moiré detected — likely live
        elif max_peak < 0.20:
            return 0.7
        elif max_peak < 0.30:
            return 0.4
        else:
            return 0.1  # Strong Moiré — likely screen
