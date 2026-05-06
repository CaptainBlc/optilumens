"""
Global Enhancer — scene-wide tonal corrections.

CMPE 491 Senior Design Project — Furkan-Develop layer.

Runs after GFPGAN face restoration and before the semantic layer so that
downstream face parsing operates on a properly-exposed image.

Operations (cheap, single-pass):
    1. Auto-gamma   — lift mid-tones if mean luminance is low
    2. CLAHE on L   — local contrast on the LAB lightness channel
    3. Saturation   — gentle vibrance bump

Conservative by design — heavy lifting (white-balance, tint, colour cast)
will be added in a follow-up.  Goal here: rescue dim webcam frames so
features become visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class GlobalConfig:
    auto_gamma:        bool  = True
    target_luminance:  float = 0.55      # only lift if mean Y < this
    max_gamma:         float = 1.8       # cap on darkening compensation
    clahe_clip:        float = 2.5       # CLAHE clipLimit on L channel
    clahe_grid:        int   = 8         # tile size
    saturation_gain:   float = 1.08      # 1.0 = unchanged


@dataclass
class GlobalEnhanceResult:
    image:    Optional[np.ndarray] = None
    gamma:    float = 1.0
    luminance_before: float = 0.0
    luminance_after:  float = 0.0
    log:      List[str] = None
    success:  bool = False

    def __post_init__(self):
        if self.log is None:
            self.log = []


class GlobalEnhancer:
    """Scene-wide auto-tone corrections."""

    def __init__(self, config: Optional[GlobalConfig] = None) -> None:
        self.cfg = config if config is not None else GlobalConfig()

    def enhance(self, img: np.ndarray) -> GlobalEnhanceResult:
        cfg = self.cfg
        result = GlobalEnhanceResult()

        if img is None or img.size == 0:
            result.log.append("[ERROR] Invalid input")
            return result

        out = img.copy()
        y0 = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()) / 255.0
        result.luminance_before = y0
        result.log.append("GlobalEnhancer.enhance()")
        result.log.append("  luminance(before) = {:.3f}".format(y0))

        # 1) Auto gamma — only when scene is dim
        if cfg.auto_gamma and y0 < cfg.target_luminance:
            # gamma < 1 brightens; map current y → target via y^gamma = target
            gamma = max(1.0 / cfg.max_gamma,
                        np.log(max(y0, 0.01)) / np.log(max(cfg.target_luminance, 0.01)))
            inv = 1.0 / max(gamma, 1e-3)
            lut = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)],
                           dtype=np.uint8)
            out = cv2.LUT(out, lut)
            result.gamma = float(gamma)
            result.log.append("  gamma = {:.3f}  (lifted dim scene)".format(gamma))
        else:
            result.log.append("  gamma skipped (scene bright enough)")

        # 2) CLAHE on L channel of LAB — local contrast
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                                tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        result.log.append("  CLAHE clip={:.1f} grid={}x{}".format(
            cfg.clahe_clip, cfg.clahe_grid, cfg.clahe_grid))

        # 3) Saturation
        if abs(cfg.saturation_gain - 1.0) > 1e-3:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * cfg.saturation_gain, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            result.log.append("  saturation x{:.2f}".format(cfg.saturation_gain))

        y1 = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()) / 255.0
        result.luminance_after = y1
        result.log.append("  luminance(after)  = {:.3f}".format(y1))

        result.image = out
        result.success = True
        return result
