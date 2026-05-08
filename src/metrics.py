"""
Image Quality Metrics — GFPGAN Enhancement System.
PSNR, SSIM, Entropy, Colorfulness, Difference Map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class QualityMetrics:
    psnr: float
    ssim: float
    entropy_before: float
    entropy_after: float
    colorfulness_before: float
    colorfulness_after: float
    delta_e: float
    sat_shift: float


class MetricsCalculator:

    @staticmethod
    def psnr(a: np.ndarray, b: np.ndarray) -> float:
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        return float("inf") if mse < 1e-10 else float(10.0 * np.log10(255.0 ** 2 / mse))

    @staticmethod
    def ssim(a: np.ndarray, b: np.ndarray, k1=0.01, k2=0.03, L=255) -> float:
        g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
        c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
        mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
        m1, m2, m12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - m1
        s2 = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - m2
        s12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - m12
        num = (2 * m12 + c1) * (2 * s12 + c2)
        den = (m1 + m2 + c1) * (s1 + s2 + c2)
        return float(np.mean(num / den))

    @staticmethod
    def entropy(img: np.ndarray) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        h = h / h.sum(); h = h[h > 0]
        return float(-np.sum(h * np.log2(h)))

    @staticmethod
    def colorfulness(img: np.ndarray) -> float:
        b, g, r = [img[:,:,i].astype(np.float64) for i in range(3)]
        rg, yb = r - g, 0.5 * (r + g) - b
        return float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
                     + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))

    @staticmethod
    def difference_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
        diff = np.abs(g2 - g1)
        mx = diff.max()
        diff = (diff / mx * 255).astype(np.uint8) if mx > 0 else diff.astype(np.uint8)
        return cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)

    @staticmethod
    def delta_e(a: np.ndarray, b: np.ndarray) -> float:
        """Mean LAB ΔE (CIE76) — fast perceptual colour drift proxy."""
        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
        d = lab_a - lab_b
        de = np.sqrt(np.sum(d * d, axis=2))
        return float(de.mean())

    @staticmethod
    def sat_shift(a: np.ndarray, b: np.ndarray) -> float:
        """Mean |ΔS| in HSV (0..255)."""
        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        ha = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        hb = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        Sa = ha[:, :, 1].astype(np.float32)
        Sb = hb[:, :, 1].astype(np.float32)
        return float(np.mean(np.abs(Sb - Sa)))

    def compute_all(self, original: np.ndarray, restored: np.ndarray) -> QualityMetrics:
        return QualityMetrics(
            psnr              = self.psnr(original, restored),
            ssim              = self.ssim(original, restored),
            entropy_before    = self.entropy(original),
            entropy_after     = self.entropy(restored),
            colorfulness_before = self.colorfulness(original),
            colorfulness_after  = self.colorfulness(restored),
            delta_e           = self.delta_e(original, restored),
            sat_shift         = self.sat_shift(original, restored),
        )
