"""
Region Enhancer — Semantic Layer (Stage 2)

CMPE 491 Senior Design Project.

Consumes a SemanticResult from semantic_parser and applies region-specific
processing with feathered alpha-blending so transitions stay invisible.

Operations per region:
    skin    → bilateral smoothing (preserves edges, reduces blemishes)
    eyes    → unsharp mask + mild brightness lift
    lips    → saturation + warmth boost
    brows   → local contrast
    nose    → mild sharpening
    hair    → optional denoise (off by default — slow)

Mask pipeline:  morphological open → Gaussian feather → soft-blend
The opening step kills sub-pixel noise from Stage 1's inverse-affine
warp; the Gaussian feather avoids visible seams between treated and
untreated areas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from semantic_parser import SemanticResult, FaceParseResult


# ── Configuration ─────────────────────────────────────────────────────

@dataclass
class RegionConfig:
    """Per-region intensity (0 = no effect, 1 = max effect)."""
    skin_smooth:   float = 0.40
    eye_sharpen:   float = 0.50
    eye_brighten:  float = 0.15
    lip_vibrance:  float = 0.35
    lip_warmth:    float = 0.20
    brow_contrast: float = 0.30
    nose_sharpen:  float = 0.20
    hair_denoise:  float = 0.00      # 0 = skip (NLM is slow)

    # Mask processing
    morph_open_px: int   = 3         # kernel radius for noise cleanup
    feather_px:    int   = 9         # Gaussian sigma for soft transitions


# ── Result ────────────────────────────────────────────────────────────

@dataclass
class RegionEnhanceResult:
    image: Optional[np.ndarray] = None
    applied: Dict[str, int] = field(default_factory=dict)   # name -> pixel count
    log: List[str] = field(default_factory=list)
    success: bool = False


# ── Mask helpers ──────────────────────────────────────────────────────

def _clean_mask(mask: np.ndarray, open_px: int) -> np.ndarray:
    """Morphological opening to remove tiny isolated label noise."""
    if open_px <= 0:
        return mask
    k = max(3, 2 * open_px + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur a 0/255 mask into a float32 [0..1] alpha map."""
    if sigma <= 0:
        return mask.astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return soft.astype(np.float32) / 255.0


def _blend(base: np.ndarray, processed: np.ndarray,
           alpha: np.ndarray) -> np.ndarray:
    """Per-pixel α blend.  alpha shape: (H, W) float32 in [0..1]."""
    a3 = alpha[..., None]
    return (base.astype(np.float32) * (1.0 - a3)
            + processed.astype(np.float32) * a3).astype(np.uint8)


def _union_mask(faces: List[FaceParseResult],
                names: List[str], shape: Tuple[int, int]) -> np.ndarray:
    """Union of all matching region masks across all detected faces."""
    out = np.zeros(shape, dtype=np.uint8)
    for face in faces:
        for n in names:
            r = face.regions.get(n)
            if r is not None:
                np.maximum(out, r.mask, out=out)
    return out


# ── Per-region operations (image-level processed versions) ────────────
# Each returns a *full-image* processed copy. Blending is done outside.

def _proc_skin_smooth(img: np.ndarray, amount: float) -> np.ndarray:
    """Bilateral filter — edge-preserving smoothing for skin."""
    # d=9 is a reasonable budget; sigmas scale with amount.
    sc = 25 + 60 * amount      # color sigma
    ss = 25 + 60 * amount      # spatial sigma
    return cv2.bilateralFilter(img, d=9, sigmaColor=sc, sigmaSpace=ss)


def _proc_unsharp(img: np.ndarray, amount: float, sigma: float = 1.4) -> np.ndarray:
    """Unsharp mask — adds high-frequency detail."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _proc_brighten(img: np.ndarray, amount: float) -> np.ndarray:
    """Lift V channel in HSV (preserves color)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + amount * 0.30), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _proc_vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    """Boost saturation (lips)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + amount * 0.50), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _proc_warmth(img: np.ndarray, amount: float) -> np.ndarray:
    """Shift toward warm tones: + red, − blue (lips)."""
    out = img.astype(np.float32)
    out[..., 2] = np.clip(out[..., 2] + amount * 12, 0, 255)   # R
    out[..., 0] = np.clip(out[..., 0] - amount *  8, 0, 255)   # B
    return out.astype(np.uint8)


def _proc_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """Linear contrast around mid-grey (brows)."""
    return cv2.convertScaleAbs(img, alpha=1.0 + amount * 0.35, beta=-amount * 8)


def _proc_denoise(img: np.ndarray, amount: float) -> np.ndarray:
    """NLM denoise — used only for hair, slow."""
    h = 3 + amount * 7
    return cv2.fastNlMeansDenoisingColored(img, None, h=h, hColor=h,
                                           templateWindowSize=7,
                                           searchWindowSize=15)


# ── Main class ────────────────────────────────────────────────────────

class RegionEnhancer:
    """
    Apply per-region enhancements on top of a SemanticResult.

    Usage:
        cfg = RegionConfig(skin_smooth=0.5, lip_vibrance=0.4)
        out = RegionEnhancer(cfg).apply(img, semantic_result)
        cv2.imshow("enhanced", out.image)

    Order of operations (largest area first → details on top):
        skin → hair → nose → brows → eyes → lips
    """

    def __init__(self, config: Optional[RegionConfig] = None) -> None:
        self.cfg = config if config is not None else RegionConfig()
        self._log: List[str] = []

    # ── public API ────────────────────────────────────────────────────

    def apply(self, img: np.ndarray,
              semantic: SemanticResult) -> RegionEnhanceResult:
        """Run the full per-region enhancement pass."""
        self._log = []
        result = RegionEnhanceResult(applied={}, log=self._log)

        if img is None or img.size == 0:
            self._log.append("[ERROR] Invalid input image")
            return result
        if not semantic.success or not semantic.faces:
            self._log.append("RegionEnhancer: no faces — passthrough")
            result.image = img.copy()
            result.success = True
            return result

        H, W = img.shape[:2]
        out = img.copy()
        cfg = self.cfg

        self._log.append("RegionEnhancer.apply()")
        self._log.append(
            "  feather={}px  morph_open={}px  faces={}".format(
                cfg.feather_px, cfg.morph_open_px, len(semantic.faces)))

        # Layer order matters: broad regions first, fine details on top
        layers: List[Tuple[str, List[str], float, callable]] = [
            ("skin",  ["skin"],            cfg.skin_smooth,   _proc_skin_smooth),
            ("hair",  ["hair"],            cfg.hair_denoise,  _proc_denoise),
            ("nose",  ["nose"],            cfg.nose_sharpen,  _proc_unsharp),
            ("brows", ["l_brow", "r_brow"], cfg.brow_contrast, _proc_contrast),
        ]

        # ── Single-op layers ──
        for name, region_names, amount, op in layers:
            if amount <= 0.001:
                continue
            mask = _union_mask(semantic.faces, region_names, (H, W))
            mask = _clean_mask(mask, cfg.morph_open_px)
            count = int((mask > 0).sum())
            if count == 0:
                continue
            alpha = _feather(mask, cfg.feather_px)
            processed = op(out, amount)
            out = _blend(out, processed, alpha)
            result.applied[name] = count
            self._log.append("  {:<6s} px={:>7d}  amount={:.2f}".format(
                name, count, amount))

        # ── Eyes: sharpen then brighten through the same alpha ──
        eye_mask = _union_mask(semantic.faces, ["l_eye", "r_eye"], (H, W))
        eye_mask = _clean_mask(eye_mask, cfg.morph_open_px)
        eye_count = int((eye_mask > 0).sum())
        if eye_count and (cfg.eye_sharpen > 0.001 or cfg.eye_brighten > 0.001):
            alpha = _feather(eye_mask, cfg.feather_px)
            tmp = out
            if cfg.eye_sharpen > 0.001:
                tmp = _proc_unsharp(tmp, cfg.eye_sharpen, sigma=1.0)
            if cfg.eye_brighten > 0.001:
                tmp = _proc_brighten(tmp, cfg.eye_brighten)
            out = _blend(out, tmp, alpha)
            result.applied["eyes"] = eye_count
            self._log.append("  eyes   px={:>7d}  sharp={:.2f} bright={:.2f}".format(
                eye_count, cfg.eye_sharpen, cfg.eye_brighten))

        # ── Lips + mouth: vibrance + warmth ──
        lip_mask = _union_mask(semantic.faces,
                               ["u_lip", "l_lip", "mouth"], (H, W))
        lip_mask = _clean_mask(lip_mask, cfg.morph_open_px)
        lip_count = int((lip_mask > 0).sum())
        if lip_count and (cfg.lip_vibrance > 0.001 or cfg.lip_warmth > 0.001):
            alpha = _feather(lip_mask, cfg.feather_px)
            tmp = out
            if cfg.lip_vibrance > 0.001:
                tmp = _proc_vibrance(tmp, cfg.lip_vibrance)
            if cfg.lip_warmth > 0.001:
                tmp = _proc_warmth(tmp, cfg.lip_warmth)
            out = _blend(out, tmp, alpha)
            result.applied["lips"] = lip_count
            self._log.append("  lips   px={:>7d}  vibr={:.2f} warm={:.2f}".format(
                lip_count, cfg.lip_vibrance, cfg.lip_warmth))

        result.image = out
        result.success = True
        return result

    def get_log(self) -> List[str]:
        return self._log.copy()


# ── Visualization helper ──────────────────────────────────────────────

def diff_amplified(before: np.ndarray, after: np.ndarray,
                   gain: float = 4.0) -> np.ndarray:
    """Pixel-wise |after - before| × gain — shows where changes landed."""
    d = cv2.absdiff(after, before).astype(np.float32) * gain
    return np.clip(d, 0, 255).astype(np.uint8)


# ── Standalone smoke test ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import sys

    from semantic_parser import FaceParser

    ap = argparse.ArgumentParser(
        description="Stage 2 smoke test: per-region enhancement")
    ap.add_argument("input",  help="input image path")
    ap.add_argument("--out",  default=None,
                    help="output path (default: <input>_enhanced.png)")
    ap.add_argument("--skin",  type=float, default=None)
    ap.add_argument("--eyes",  type=float, default=None)
    ap.add_argument("--lips",  type=float, default=None)
    ap.add_argument("--brows", type=float, default=None)
    ap.add_argument("--nose",  type=float, default=None)
    ap.add_argument("--hair",  type=float, default=None)
    ap.add_argument("--diff-gain", type=float, default=5.0,
                    help="amplification factor for difference panel")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print("[ERROR] Cannot read: {}".format(args.input))
        sys.exit(1)

    # Stage 1
    parser = FaceParser()
    sem = parser.parse(img)
    for line in sem.log:
        print(line)
    if not sem.success or not sem.faces:
        print("(no faces — nothing to do)")
        sys.exit(0)

    # Stage 2
    cfg = RegionConfig()
    if args.skin  is not None: cfg.skin_smooth   = args.skin
    if args.eyes  is not None: cfg.eye_sharpen   = args.eyes
    if args.lips  is not None: cfg.lip_vibrance  = args.lips
    if args.brows is not None: cfg.brow_contrast = args.brows
    if args.nose  is not None: cfg.nose_sharpen  = args.nose
    if args.hair  is not None: cfg.hair_denoise  = args.hair

    enh = RegionEnhancer(cfg).apply(img, sem)
    for line in enh.log:
        print(line)
    if not enh.success:
        sys.exit(2)

    # 3-panel: original | enhanced | diff×N
    diff = diff_amplified(img, enh.image, gain=args.diff_gain)
    panel = cv2.hconcat([img, enh.image, diff])

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(args.input)[0]
        out_path = base + "_enhanced.png"
    cv2.imwrite(out_path, panel)
    print("\nSaved: {}".format(out_path))

    print("\nApplied:")
    for k, v in sorted(enh.applied.items(), key=lambda kv: -kv[1]):
        print("  {:<6s} {:>7d} px".format(k, v))
