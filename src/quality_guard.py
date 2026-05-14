"""
QualityGuard — anti-hallucination safety net for AI-driven layers.

CMPE 491 Senior Design Project.

The High-Level Design report explicitly criticises systems where AI
"invents non-existent textures, creating inconsistent and untrustworthy
details". GFPGAN and Real-ESRGAN are GANs and *can* hallucinate when the
input is degraded outside their training distribution. This module
enforces a deterministic guard around every AI step:

    1. Compute SSIM (structure) and L1 (pixel drift) between the
       layer's input and output.
    2. Convert into a 0-100 trust score:
            100 = output identical to input (no drift)
              0 = output completely different (full hallucination)
    3. Compare to a threshold:
            score >= accept_threshold → keep AI output
            score >= warn_threshold   → blend AI with original to soften
            score <  warn_threshold   → reject AI output, return original

The whole guard is *post-hoc*: nothing is fed into the network, it just
audits the outputs. This keeps it model-agnostic — it works the same way
for GFPGAN, Real-ESRGAN, RegionEnhancer, or any future module.

Public API
----------
    g = QualityGuard(accept_threshold=70.0, warn_threshold=50.0)
    rep = g.evaluate(before, after, label="GFPGAN")
    rep.score          # 0..100
    rep.action         # "accept" | "blend" | "reject"
    rep.output         # the image to actually use downstream
    rep.log            # one human-readable line
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


# ── Result ───────────────────────────────────────────────────────────

@dataclass
class GuardReport:
    """Outcome of one QualityGuard evaluation."""
    label:       str   = ""
    score:       float = 100.0      # 0..100, higher = closer to original
    action:      str   = "accept"   # "accept" | "blend" | "reject"
    ssim:        float = 1.0
    pixel_drift: float = 0.0        # mean |delta| / 255
    delta_e:     float = 0.0        # mean LAB ΔE (proxy)
    sat_shift:   float = 0.0        # mean ΔS / 255 (HSV)
    output:      Optional[np.ndarray] = None
    log:         List[str] = field(default_factory=list)


# ── SSIM (lightweight — no scikit-image dependency) ──────────────────

def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


_GK_11 = _gaussian_kernel(11, 1.5)


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean SSIM (luminance only) on grayscale arrays in 0..255 uint8.
    Implementation: Wang et al. 2004. Window 11×11 Gaussian, σ=1.5.
    """
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_AREA)

    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)

    K1, K2, L = 0.01, 0.03, 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Separable convolution with the precomputed Gaussian
    def _blur(img):
        return cv2.sepFilter2D(img, -1, _GK_11, _GK_11)

    mu_a   = _blur(a_f)
    mu_b   = _blur(b_f)
    mu_a2  = mu_a * mu_a
    mu_b2  = mu_b * mu_b
    mu_ab  = mu_a * mu_b

    sigma_a2 = _blur(a_f * a_f) - mu_a2
    sigma_b2 = _blur(b_f * b_f) - mu_b2
    sigma_ab = _blur(a_f * b_f) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(ssim_map.mean())


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _downscale(img: np.ndarray, max_side: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _mean_delta_e_cie76(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """Fast perceptual colour difference proxy (CIE76) on downscaled LAB."""
    a = _downscale(a_bgr, 256)
    b = _downscale(b_bgr, 256)
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = lab_a - lab_b
    de = np.sqrt(np.sum(d * d, axis=2))
    return float(de.mean())


def _mean_sat_shift(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """Mean saturation change in HSV (0..255 scale)."""
    a = _downscale(a_bgr, 256)
    b = _downscale(b_bgr, 256)
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    Sa = hsv_a[:, :, 1].astype(np.float32)
    Sb = hsv_b[:, :, 1].astype(np.float32)
    return float(np.mean(np.abs(Sb - Sa)))


# ── Guard ────────────────────────────────────────────────────────────

class QualityGuard:
    """
    Decides what to do with an AI layer's output.

    Parameters
    ----------
    accept_threshold : float
        Trust score >= this → keep AI output untouched.
    warn_threshold : float
        Trust score in [warn, accept) → blend AI with original at
        `blend_ratio` fraction toward the original. Below `warn` → reject
        AI output entirely and pass the original through.
    blend_ratio : float
        How much of the *original* image to mix back when blending.
        0.5 = 50/50, 0.3 = mostly AI but pulled toward original.
    """

    def __init__(
        self,
        accept_threshold: float = 70.0,
        warn_threshold:   float = 50.0,
        blend_ratio:      float = 0.4,
    ) -> None:
        self.accept = float(accept_threshold)
        self.warn   = float(warn_threshold)
        self.blend  = float(np.clip(blend_ratio, 0.0, 1.0))

    # ── public API ───────────────────────────────────────────────────

    def evaluate(
        self,
        before: np.ndarray,
        after:  np.ndarray,
        label:  str = "AI",
    ) -> GuardReport:
        """Compare `before` and `after`; pick an action."""
        rep = GuardReport(label=label, output=after)

        if before is None or after is None or before.size == 0 or after.size == 0:
            rep.score = 0.0; rep.action = "reject"
            rep.output = before
            rep.log.append("[GUARD/{}] invalid inputs — reject".format(label))
            return rep

        # Match shapes (AI may up-/down-sample)
        if after.shape[:2] != before.shape[:2]:
            after_cmp = cv2.resize(after, (before.shape[1], before.shape[0]),
                                   interpolation=cv2.INTER_AREA)
        else:
            after_cmp = after

        # SSIM (luminance) + pixel drift
        ssim = _ssim_gray(_to_gray(before), _to_gray(after_cmp))
        drift = float(np.mean(np.abs(
            after_cmp.astype(np.int16) - before.astype(np.int16)))) / 255.0
        delta_e = _mean_delta_e_cie76(before, after_cmp)
        sat_shift = _mean_sat_shift(before, after_cmp)

        # Trust score: structural + colour safety.
        # - SSIM is structure (luma)
        # - drift catches overall pixel movement
        # - ΔE catches perceptual colour shifts (purple/green casts)
        # - sat_shift catches saturation overshoot
        ssim01 = max(0.0, min(1.0, ssim))
        drift_pen = 1.0 - min(1.0, drift * 5.0)
        # ΔE roughly: <3 barely visible, 5-10 noticeable, >15 strong cast.
        de_pen = 1.0 - min(1.0, max(0.0, (delta_e - 3.0) / 12.0))
        sat_pen = 1.0 - min(1.0, sat_shift / 90.0)
        score = 100.0 * (
            0.50 * ssim01 +
            0.25 * drift_pen +
            0.15 * de_pen +
            0.10 * sat_pen
        )

        # Structural floor: GANs can keep mean brightness (low drift) while
        # destroying local structure. Do not accept purely on drift/ΔE.
        if ssim01 < 0.80:
            score = min(score, self.accept - 1e-3)

        # Hard cap: catastrophic colour drift is always at least a BLEND.
        # A single channel shifting 18% on average is the kind of damage
        # we never want to silently accept, even if SSIM is forgiving.
        if drift > 0.18 and score >= self.accept:
            score = self.warn  # forces the blend branch below

        score = float(max(0.0, min(100.0, score)))

        rep.ssim = float(ssim01)
        rep.pixel_drift = drift
        rep.delta_e = float(delta_e)
        rep.sat_shift = float(sat_shift)
        rep.score = score

        if score >= self.accept:
            rep.action = "accept"
            rep.output = after
            rep.log.append(
                "[GUARD/{}] ACCEPT  trust={:.1f}  ssim={:.3f}  drift={:.3f}  ΔE={:.1f}  ΔS={:.0f}".format(
                    label, score, ssim01, drift, delta_e, sat_shift))
        elif score >= self.warn:
            # Blend: pull AI output toward original
            blended = cv2.addWeighted(
                after_cmp, 1.0 - self.blend, before, self.blend, 0)
            rep.action = "blend"
            rep.output = blended
            rep.log.append(
                "[GUARD/{}] BLEND   trust={:.1f}  ssim={:.3f}  drift={:.3f}  ΔE={:.1f}  ΔS={:.0f}  "
                "(mix={:.0f}% original)".format(
                    label, score, ssim01, drift, delta_e, sat_shift, self.blend * 100))
        else:
            rep.action = "reject"
            rep.output = before
            rep.log.append(
                "[GUARD/{}] REJECT  trust={:.1f}  ssim={:.3f}  drift={:.3f}  ΔE={:.1f}  ΔS={:.0f}  "
                "— possible hallucination, keeping original".format(
                    label, score, ssim01, drift, delta_e, sat_shift))

        return rep
