"""
Image Quality Metrics Module for Pixel Enhancement System.

Provides quantitative comparison between original and enhanced images.
Reference: Friend's MetricsCalculator (Global Enhancement Module).

Metrics:
    - PSNR  (Peak Signal-to-Noise Ratio)
    - SSIM  (Structural Similarity Index)
    - Entropy (Information content)
    - Colorfulness (Hasler & Süsstrunk metric)
    - Difference Map (color-coded heatmap of changes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class QualityMetrics:
    """Stores computed quality metrics for a single image pair."""

    psnr: float
    ssim: float
    entropy_original: float
    entropy_enhanced: float
    colorfulness_original: float
    colorfulness_enhanced: float


class MetricsCalculator:
    """Computes image quality metrics between original and enhanced images."""

    @staticmethod
    def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
        """Peak Signal-to-Noise Ratio (dB). Higher = less distortion."""
        mse = np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2)
        if mse < 1e-10:
            return float("inf")
        return float(10.0 * np.log10(255.0 ** 2 / mse))

    @staticmethod
    def compute_ssim(
        original: np.ndarray,
        enhanced: np.ndarray,
        *,
        k1: float = 0.01,
        k2: float = 0.03,
        L: int = 255,
    ) -> float:
        """Structural Similarity Index (-1..1). 1 = identical structure."""
        gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float64)

        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2

        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

        numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / denominator
        return float(np.mean(ssim_map))

    @staticmethod
    def compute_entropy(image: np.ndarray) -> float:
        """Shannon entropy of grayscale histogram. Higher = more information."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    @staticmethod
    def compute_colorfulness(image: np.ndarray) -> float:
        """Hasler & Susstrunk colorfulness metric. Higher = more vivid colors."""
        b, g, r = image[:, :, 0].astype(np.float64), \
                   image[:, :, 1].astype(np.float64), \
                   image[:, :, 2].astype(np.float64)

        rg = r - g
        yb = 0.5 * (r + g) - b

        std_rg = float(np.std(rg))
        std_yb = float(np.std(yb))
        mean_rg = float(np.mean(rg))
        mean_yb = float(np.mean(yb))

        std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)

        return float(std_root + 0.3 * mean_root)

    @staticmethod
    def compute_difference_map(
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> np.ndarray:
        """Color-coded heatmap showing where pixel changes occurred.

        Returns BGR uint8 heatmap: blue=no change, red=max change.
        """
        gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float64)
        diff = np.abs(gray2 - gray1)

        max_diff = diff.max()
        if max_diff > 0:
            diff = (diff / max_diff * 255.0).astype(np.uint8)
        else:
            diff = diff.astype(np.uint8)

        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
        return heatmap

    def compute_all(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> QualityMetrics:
        """Compute all metrics at once."""
        return QualityMetrics(
            psnr=self.compute_psnr(original, enhanced),
            ssim=self.compute_ssim(original, enhanced),
            entropy_original=self.compute_entropy(original),
            entropy_enhanced=self.compute_entropy(enhanced),
            colorfulness_original=self.compute_colorfulness(original),
            colorfulness_enhanced=self.compute_colorfulness(enhanced),
        )
