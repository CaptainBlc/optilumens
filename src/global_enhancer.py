"""
GlobalEnhancer — whole-image quality lift for the enhancement pipeline.

CMPE 491 Senior Design Project.

Implements the 'Global Enhancement Layer' (Step 5) that was previously
a placeholder in pipeline.py.  Applies a stacked pipeline of perceptual
quality improvements to transform webcam frames into something that looks
like it was captured by a quality camera:

    1. White Balance  — Gray-World AWB removes greenish/cold webcam casts
    2. Shadow Lift    — adaptive gamma correction for low-light scenes
    3. Denoising      — fast bilateral filter before any sharpening
    4. CLAHE          — adaptive local contrast, no blown highlights
    5. HDR Tone Map   — Reinhard operator rolls off highlights smoothly
    6. Sharpening     — multi-scale unsharp mask recovers micro-detail
    7. Vibrance       — targeted saturation boost in desaturated areas
    8. Film Grade     — subtle S-curve + warm split-tone colour grade
    9. Temporal Avg   — exponentially-weighted frame blend (live mode)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import cv2
import numpy as np


@dataclass
class GlobalEnhancerConfig:
    """Intensity knobs — 0.0 = off, 1.0 = maximum effect."""
    white_balance:   float = 0.80
    shadow_lift:     float = 0.75
    denoise:         float = 0.55
    clahe_strength:  float = 0.75
    hdr_tone:        float = 0.65
    sharpen:         float = 0.70
    vibrance:        float = 0.60
    film_look:       float = 0.55
    temporal_frames: int   = 3


@dataclass
class GlobalEnhanceResult:
    image:         Optional[np.ndarray] = None
    log:           List[str]            = field(default_factory=list)
    success:       bool                 = False
    steps_applied: List[str]            = field(default_factory=list)


def _gray_world_wb(img: np.ndarray, strength: float) -> np.ndarray:
    img_f = img.astype(np.float32)
    means = img_f.mean(axis=(0, 1))
    if means.min() < 1.0:
        return img
    gray = means.mean()
    scale = np.clip(gray / means, 0.70, 1.80)
    corrected = np.clip(img_f * scale, 0, 255).astype(np.uint8)
    return cv2.addWeighted(corrected, strength, img, 1.0 - strength, 0)


def _adaptive_gamma_lift(img: np.ndarray, brightness: float, strength: float) -> np.ndarray:
    if brightness >= 0.55:
        return img
    darkness = 0.55 - brightness
    gamma    = max(0.38, 1.0 - darkness * 0.95)
    inv_g    = 1.0 / gamma
    lut      = np.array(
        [min(255, int((i / 255.0) ** inv_g * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    corrected = cv2.LUT(img, lut)
    alpha     = min(0.95, strength * darkness * 2.2)
    return cv2.addWeighted(corrected, alpha, img, 1.0 - alpha, 0)


def _fast_bilateral_denoise(img: np.ndarray, noise_level: float, strength: float) -> np.ndarray:
    if noise_level < 4.0 and strength < 0.4:
        return img
    nf = min(1.0, noise_level / 25.0)
    sc = int(15 + strength * nf * 65)
    ss = int(8  + strength * nf * 15)
    denoised = cv2.bilateralFilter(img, d=7, sigmaColor=sc, sigmaSpace=ss)
    alpha = min(0.88, strength * nf + 0.15)
    return cv2.addWeighted(denoised, alpha, img, 1.0 - alpha, 0)


def _clahe_enhancement(img: np.ndarray, strength: float) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clip  = 1.5 + strength * 3.5
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    L_eq  = clahe.apply(L)
    L_out = cv2.addWeighted(L_eq, strength, L, 1.0 - strength, 0)
    return cv2.cvtColor(cv2.merge((L_out, A, B)), cv2.COLOR_LAB2BGR)


def _reinhard_hdr_tonemap(img: np.ndarray, strength: float) -> np.ndarray:
    img_f  = img.astype(np.float32) / 255.0
    lum    = (0.2126 * img_f[:, :, 2]
              + 0.7152 * img_f[:, :, 1]
              + 0.0722 * img_f[:, :, 0])
    lum    = np.maximum(lum, 1e-6)
    L_w    = 1.8
    lum_out = lum * (1.0 + lum / (L_w * L_w)) / (1.0 + lum)
    ratio   = (lum_out / lum)[:, :, np.newaxis]
    tone    = np.clip(img_f * ratio, 0.0, 1.0)
    tone    = np.power(tone, 1.0 / 1.9)
    result  = (tone * 255).astype(np.uint8)
    return cv2.addWeighted(result, strength, img, 1.0 - strength, 0)


def _advanced_sharpen(img: np.ndarray, strength: float) -> np.ndarray:
    img_f   = img.astype(np.float32)
    b_fine  = cv2.GaussianBlur(img_f, (0, 0), sigmaX=0.8)
    b_mid   = cv2.GaussianBlur(img_f, (0, 0), sigmaX=2.2)
    d_fine  = img_f - b_fine
    d_mid   = img_f - b_mid
    out     = img_f + strength * 1.2 * d_fine + strength * 0.5 * d_mid
    return np.clip(out, 0, 255).astype(np.uint8)


def _vibrance_boost(img: np.ndarray, strength: float) -> np.ndarray:
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    S_f  = S.astype(np.float32)
    weight = 1.0 - S_f / 255.0
    S_new  = np.clip(S_f + strength * 78.0 * weight, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([H, S_new, V]), cv2.COLOR_HSV2BGR)


def _film_grade(img: np.ndarray, strength: float) -> np.ndarray:
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    x    = np.arange(256, dtype=np.float32) / 255.0
    k    = strength * 0.13
    y    = np.clip(x + k * np.sin(np.pi * x), 0.0, 1.0)
    lut  = (y * 255).astype(np.uint8)
    L_g  = cv2.LUT(L, lut)
    sw   = (255 - L.astype(np.float32)) / 255.0
    A_g  = np.clip(A.astype(np.float32) + strength * 2.8 * sw, 0, 255).astype(np.uint8)
    B_g  = np.clip(B.astype(np.float32) + strength * 2.2 * sw, 0, 255).astype(np.uint8)
    graded = cv2.cvtColor(cv2.merge((L_g, A_g, B_g)), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(graded, strength, img, 1.0 - strength, 0)


class GlobalEnhancer:
    """Full-image quality enhancer — the Global Enhancement Layer."""

    def __init__(self, config: Optional[GlobalEnhancerConfig] = None) -> None:
        self.cfg        = config or GlobalEnhancerConfig()
        self._frame_buf: Deque[np.ndarray] = deque(maxlen=self.cfg.temporal_frames)

    def reset(self) -> None:
        self._frame_buf.clear()

    def enhance(
        self,
        img: np.ndarray,
        profile=None,
        use_temporal: bool = False,
    ) -> GlobalEnhanceResult:
        res = GlobalEnhanceResult()
        log: List[str] = []

        if img is None or img.size == 0:
            log.append("[ERROR] GlobalEnhancer: empty input")
            res.log = log
            return res

        brightness  = float(getattr(profile, "brightness",  0.50))
        noise_level = float(getattr(profile, "noise_level", 12.0))

        log.append("GlobalEnhancer.enhance()")
        out   = img.copy()
        steps: List[str] = []
        cfg   = self.cfg

        if cfg.white_balance > 0.01:
            out = _gray_world_wb(out, cfg.white_balance)
            steps.append("wb")
            log.append("  white_balance   = {:.2f}".format(cfg.white_balance))

        if cfg.shadow_lift > 0.01:
            out = _adaptive_gamma_lift(out, brightness, cfg.shadow_lift)
            steps.append("gamma")
            log.append("  shadow_lift     = {:.2f}  brightness={:.3f}".format(
                cfg.shadow_lift, brightness))

        if cfg.denoise > 0.01:
            out = _fast_bilateral_denoise(out, noise_level, cfg.denoise)
            steps.append("denoise")
            log.append("  bilateral       = {:.2f}  noise={:.1f}".format(
                cfg.denoise, noise_level))

        if cfg.clahe_strength > 0.01:
            out = _clahe_enhancement(out, cfg.clahe_strength)
            steps.append("clahe")
            log.append("  clahe_clip      = {:.1f}".format(
                1.5 + cfg.clahe_strength * 3.5))

        if cfg.hdr_tone > 0.01:
            out = _reinhard_hdr_tonemap(out, cfg.hdr_tone)
            steps.append("hdr")
            log.append("  reinhard_hdr    = {:.2f}".format(cfg.hdr_tone))

        if cfg.sharpen > 0.01:
            out = _advanced_sharpen(out, cfg.sharpen)
            steps.append("sharpen")
            log.append("  multi_sharpen   = {:.2f}".format(cfg.sharpen))

        if cfg.vibrance > 0.01:
            out = _vibrance_boost(out, cfg.vibrance)
            steps.append("vibrance")
            log.append("  vibrance        = {:.2f}".format(cfg.vibrance))

        if cfg.film_look > 0.01:
            out = _film_grade(out, cfg.film_look)
            steps.append("film")
            log.append("  film_grade      = {:.2f}".format(cfg.film_look))

        if use_temporal and cfg.temporal_frames > 1:
            self._frame_buf.append(out.astype(np.float32))
            n = len(self._frame_buf)
            if n > 1:
                w = np.exp(np.linspace(-1.0, 0.0, n))
                w /= w.sum()
                avg = np.zeros_like(out, dtype=np.float32)
                for wi, f in zip(w, self._frame_buf):
                    avg += wi * f
                out = np.clip(avg, 0, 255).astype(np.uint8)
                steps.append("temporal")
                log.append("  temporal_avg    n={}".format(n))

        res.image         = out
        res.log           = log
        res.success       = True
        res.steps_applied = steps
        log.append("  Applied: [{}]".format(", ".join(steps)))
        return res
