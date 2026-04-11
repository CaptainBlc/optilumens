"""
Image Profiler — GFPGAN-based Enhancement System.
Analyzes input images to guide adaptive restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ProfileResult:
    brightness: float
    contrast: float
    blur_score: float
    noise_level: float
    edge_density: float
    skin_ratio: float
    has_skin: bool
    is_low_light: bool
    is_overexposed: bool
    is_noisy: bool
    is_blurry: bool
    width: int
    height: int

    def summary(self) -> str:
        flags = []
        if self.is_low_light:   flags.append("LOW_LIGHT")
        if self.is_overexposed: flags.append("OVEREXPOSED")
        if self.is_noisy:       flags.append("NOISY")
        if self.is_blurry:      flags.append("BLURRY")
        if self.has_skin:       flags.append("FACE/SKIN")
        return " | ".join(flags) if flags else "NORMAL"


class ImageProfiler:
    SKIN_THRESHOLD  = 0.05
    LOW_LIGHT_THR   = 0.30
    OVEREXP_THR     = 0.75
    NOISE_THR       = 15.0
    BLUR_THR        = 100.0

    def profile(self, image: np.ndarray) -> Optional[ProfileResult]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None
        if image.ndim != 3 or image.shape[2] != 3:
            return None

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gf   = gray.astype(np.float64) / 255.0

        brightness  = float(np.mean(gf))
        contrast    = float(np.std(gf))
        blur_score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges       = cv2.Canny(gray, 100, 200)
        edge_density = float(np.sum(edges > 0) / max(edges.size, 1))
        noise_level  = self._estimate_noise(gray)
        skin_ratio, has_skin = self._detect_skin(image)

        return ProfileResult(
            brightness   = round(brightness, 4),
            contrast     = round(contrast, 4),
            blur_score   = round(blur_score, 2),
            noise_level  = round(noise_level, 2),
            edge_density = round(edge_density, 4),
            skin_ratio   = round(skin_ratio, 4),
            has_skin     = has_skin,
            is_low_light = brightness < self.LOW_LIGHT_THR,
            is_overexposed = brightness > self.OVEREXP_THR,
            is_noisy     = noise_level > self.NOISE_THR,
            is_blurry    = blur_score < self.BLUR_THR,
            width=w, height=h,
        )

    def _estimate_noise(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        k = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float64)
        s = np.sum(np.abs(cv2.filter2D(gray.astype(np.float64), -1, k)))
        return float(s * np.sqrt(0.5 * np.pi) / (6.0 * max(w - 2, 1) * max(h - 2, 1)))

    def _detect_skin(self, image: np.ndarray):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask  = cv2.inRange(ycrcb,
                            np.array([0, 133, 77], np.uint8),
                            np.array([255, 173, 127], np.uint8))
        total = image.shape[0] * image.shape[1]
        ratio = float(np.count_nonzero(mask) / max(total, 1))
        return ratio, ratio > self.SKIN_THRESHOLD
