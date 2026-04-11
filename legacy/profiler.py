"""
Image Profiler Module for Pixel Enhancement System.

Analyzes input image statistics to guide adaptive pixel enhancement.
Reference: Friend's ImageProfiler (Global Enhancement Module).

Outputs a ProfileResult dataclass with all measured characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ProfileResult:
    """Complete statistical profile of an input image."""

    # Basic statistics (0..1 normalized)
    brightness: float
    contrast: float

    # Sharpness / blur (Laplacian variance; higher = sharper)
    blur_score: float

    # Edge density (ratio of edge pixels)
    edge_density: float

    # Noise estimate (standard deviation of high-frequency component)
    noise_level: float

    # Skin detection
    skin_ratio: float
    has_skin: bool

    # Scene classification flags
    is_low_light: bool
    is_overexposed: bool
    is_noisy: bool
    is_blurry: bool

    # Image dimensions
    width: int
    height: int


class ImageProfiler:
    """Analyzes an image and returns a ProfileResult with all statistics."""

    SKIN_THRESHOLD: float = 0.05
    LOW_LIGHT_THRESHOLD: float = 0.3
    OVEREXPOSED_THRESHOLD: float = 0.75
    NOISE_THRESHOLD: float = 15.0
    BLUR_THRESHOLD: float = 100.0

    def profile(self, image: np.ndarray) -> Optional[ProfileResult]:
        """Analyze image and return a complete profile.

        Parameters
        ----------
        image : np.ndarray
            BGR, uint8, (H, W, 3) input image.

        Returns
        -------
        ProfileResult or None if input is invalid.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None

        if image.ndim != 3 or image.shape[2] != 3:
            return None

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float64) / 255.0

        brightness = float(np.mean(gray_f))
        contrast = float(np.std(gray_f))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.sum(edges > 0) / max(edges.size, 1))

        noise_level = self._estimate_noise(gray)

        skin_ratio, has_skin = self._detect_skin(image)

        is_low_light = brightness < self.LOW_LIGHT_THRESHOLD
        is_overexposed = brightness > self.OVEREXPOSED_THRESHOLD
        is_noisy = noise_level > self.NOISE_THRESHOLD
        is_blurry = blur_score < self.BLUR_THRESHOLD

        return ProfileResult(
            brightness=round(brightness, 4),
            contrast=round(contrast, 4),
            blur_score=round(blur_score, 2),
            edge_density=round(edge_density, 4),
            noise_level=round(noise_level, 2),
            skin_ratio=round(skin_ratio, 4),
            has_skin=has_skin,
            is_low_light=is_low_light,
            is_overexposed=is_overexposed,
            is_noisy=is_noisy,
            is_blurry=is_blurry,
            width=w,
            height=h,
        )

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level using Laplacian-based method (Immerkær 1996).

        Returns estimated standard deviation of noise.
        """
        h, w = gray.shape
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1],
        ], dtype=np.float64)

        sigma = np.sum(np.abs(cv2.filter2D(gray.astype(np.float64), -1, kernel)))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
        return float(sigma)

    def _detect_skin(self, image: np.ndarray) -> tuple:
        """Detect skin-tone pixels using YCrCb color space.

        Returns (skin_ratio, has_skin).
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        skin_pixels = np.count_nonzero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        ratio = skin_pixels / total_pixels if total_pixels > 0 else 0.0

        return float(ratio), ratio > self.SKIN_THRESHOLD


def profile_image(img: np.ndarray) -> Optional[dict]:
    """Legacy wrapper: returns a dict for backward compatibility.

    Existing code calls profile_image(img) and expects a dict.
    """
    profiler = ImageProfiler()
    result = profiler.profile(img)
    if result is None:
        return None
    return {
        "brightness": result.brightness,
        "contrast": result.contrast,
        "blur": result.blur_score,
        "edge_density": result.edge_density,
        "noise_level": result.noise_level,
        "skin_ratio": result.skin_ratio,
        "has_skin": result.has_skin,
        "is_low_light": result.is_low_light,
        "is_overexposed": result.is_overexposed,
        "is_noisy": result.is_noisy,
        "is_blurry": result.is_blurry,
        "width": result.width,
        "height": result.height,
    }
