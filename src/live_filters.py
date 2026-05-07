"""
Live Filters — real-time video-stream enhancements for the webcam feed.

Implements Analysis Report §3.5.1 Use Case #4 extended ("live video filter
stream"). Provides a small family of per-frame filters that are cheap
enough to run on a CPU at 30 FPS (Beauty) and a best-effort AI filter
that runs GFPGAN on downscaled faces with frame-skipping + caching.

Design
------
- Every filter implements `apply(frame_bgr) -> frame_bgr`.
- Filters are stateless from the caller's point of view; any caching is
  encapsulated inside the instance.
- Each frame is processed in-place of the reference passed in; callers
  should not assume the original is preserved.
- AIFilter gracefully falls back to BeautyFilter if GFPGAN is not ready.

Complexity on a typical i7 + 1280x720 webcam (rough numbers):
    NoFilter      : <1 ms
    BeautyFilter  : 15-25 ms     (well within 30 FPS budget)
    AIFilter      : 1-3 s every N-th frame; cached overlay in between
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


# ── Base ──────────────────────────────────────────────────────────

class BaseFilter:
    name = "BASE"

    # Init/ready diagnostics — surfaced in the GUI status banner so the
    # user can tell whether AI/ENHANCE actually loaded or fell back.
    init_log: str = ""        # human-readable status of last init attempt
    is_ready: bool = True     # True when the filter is operating "as advertised"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def reset(self) -> None:
        """Clear any internal cache."""
        pass

    def set_params(self, **kwargs) -> None:
        """Live-tunable parameters — overridden by Beauty / Enhance."""
        pass


class NoFilter(BaseFilter):
    name = "OFF"
    init_log = "Pass-through — no processing"


# ── Beauty: fast classical filter (real-time) ─────────────────────

class BeautyFilter(BaseFilter):
    """
    Classical real-time face-friendly filter.

    Pipeline:
        1. Bilateral filter on skin regions (soften blemishes, keep edges)
        2. CLAHE on the L channel (lift local contrast in low light)
        3. Unsharp mask (gentle structural sharpening)
        4. Optional warm tone (slight R shift, B decrease — "studio" look)

    Every step is conservative; stack them with <25 ms total on 720p.
    """

    name = "BEAUTY"

    # ── static LUTs — built once ──
    _WARM_LUT_R = np.array([min(255, int(i * 1.05 + 3)) for i in range(256)],
                           dtype=np.uint8)
    _WARM_LUT_B = np.array([max(0, int(i * 0.95)) for i in range(256)],
                           dtype=np.uint8)

    init_log = "Classical real-time filter ready (15-25 ms / frame)"
    is_ready = True

    def __init__(self,
                 smooth_strength: float = 0.65,
                 clarity: float = 0.60,
                 warmth: float = 0.35) -> None:
        self.smooth  = float(np.clip(smooth_strength, 0.0, 1.0))
        self.clarity = float(np.clip(clarity, 0.0, 1.0))
        self.warmth  = float(np.clip(warmth, 0.0, 1.0))
        self._clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))

    def set_params(self, **kwargs) -> None:
        """Live-tune Beauty knobs from a slider panel.

        Accepts any of: smooth_strength, clarity, warmth (each 0..1).
        Unknown keys are ignored so we can blanket-pass GUI state.
        """
        if "smooth_strength" in kwargs:
            self.smooth  = float(np.clip(kwargs["smooth_strength"], 0.0, 1.0))
        if "smooth" in kwargs:
            self.smooth  = float(np.clip(kwargs["smooth"], 0.0, 1.0))
        if "clarity" in kwargs:
            self.clarity = float(np.clip(kwargs["clarity"], 0.0, 1.0))
        if "warmth" in kwargs:
            self.warmth  = float(np.clip(kwargs["warmth"], 0.0, 1.0))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        out = frame

        # 1) Edge-preserving bilateral smoothing
        if self.smooth > 0:
            d  = 7 if max(frame.shape[:2]) < 900 else 9
            sm = cv2.bilateralFilter(out, d=d, sigmaColor=50, sigmaSpace=12)
            a  = 0.40 + 0.50 * self.smooth
            out = cv2.addWeighted(sm, a, out, 1.0 - a, 0.0)

        # 2) CLAHE on L — stronger clip for visible contrast pop
        if self.clarity > 0:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            L_eq = self._clahe.apply(L)
            a    = 0.5 + 0.5 * self.clarity
            L    = cv2.addWeighted(L_eq, a, L, 1.0 - a, 0.0)
            out  = cv2.cvtColor(cv2.merge((L, A, B)), cv2.COLOR_LAB2BGR)

        # 3) Two-scale unsharp mask
        if self.clarity > 0:
            img_f  = out.astype(np.float32)
            b_fine = cv2.GaussianBlur(img_f, (0, 0), 0.8)
            b_mid  = cv2.GaussianBlur(img_f, (0, 0), 2.0)
            amt    = 0.35 + 0.55 * self.clarity
            sharp  = img_f + amt * (img_f - b_fine) + amt * 0.4 * (img_f - b_mid)
            out    = np.clip(sharp, 0, 255).astype(np.uint8)

        # 4) Vibrance — boost desaturated pixels
        if self.clarity > 0.2:
            hsv   = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)
            S_f   = S.astype(np.float32)
            wt    = 1.0 - S_f / 255.0
            S_new = np.clip(S_f + self.clarity * 55.0 * wt, 0, 255).astype(np.uint8)
            out   = cv2.cvtColor(cv2.merge([H, S_new, V]), cv2.COLOR_HSV2BGR)

        # 5) Warm tone via LUT
        if self.warmth > 0:
            Bc, G, R = cv2.split(out)
            R2 = cv2.LUT(R, self._WARM_LUT_R)
            B2 = cv2.LUT(Bc, self._WARM_LUT_B)
            a  = self.warmth
            R  = cv2.addWeighted(R2, a, R, 1.0 - a, 0.0)
            Bc = cv2.addWeighted(B2, a, Bc, 1.0 - a, 0.0)
            out = cv2.merge((Bc, G, R))

        return out


# ── Enhance: GlobalEnhancer — full quality pipeline, real-time ───────

class EnhanceLiveFilter(BaseFilter):
    """Full real-time global enhancement filter for the live webcam feed."""

    name = "ENHANCE"
    init_log = "(not initialised yet)"
    is_ready = False

    def __init__(self) -> None:
        self._enhancer  = None
        self._profiler  = None
        self._profile   = None
        self._frame_cnt = 0
        self._init_done = False
        self._intensity = 1.0

    def reset(self) -> None:
        if self._enhancer is not None:
            self._enhancer.reset()
        self._frame_cnt = 0
        self._profile   = None

    def set_params(self, **kwargs) -> None:
        """Tune the global stack live. Accepts: intensity 0..1."""
        if "intensity" in kwargs:
            v = float(np.clip(kwargs["intensity"], 0.0, 1.0))
            self._intensity = v
            if self._enhancer is not None:
                cfg = self._enhancer.cfg
                cfg.white_balance  = 0.85 * v
                cfg.shadow_lift    = 0.80 * v
                cfg.denoise        = 0.60 * v
                cfg.clahe_strength = 0.80 * v
                cfg.hdr_tone       = 0.70 * v
                cfg.sharpen        = 0.75 * v
                cfg.vibrance       = 0.65 * v
                cfg.film_look      = 0.60 * v

    def _ensure_init(self) -> bool:
        if self._init_done:
            return self._enhancer is not None
        self._init_done = True
        try:
            import os, sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from global_enhancer import GlobalEnhancer, GlobalEnhancerConfig
            from profiler import ImageProfiler
            cfg = GlobalEnhancerConfig(
                white_balance   = 0.85,
                shadow_lift     = 0.80,
                denoise         = 0.60,
                clahe_strength  = 0.80,
                hdr_tone        = 0.70,
                sharpen         = 0.75,
                vibrance        = 0.65,
                film_look       = 0.60,
                temporal_frames = 3,
            )
            self._enhancer = GlobalEnhancer(cfg)
            self._profiler = ImageProfiler()
            self.is_ready  = True
            self.init_log  = "GlobalEnhancer ready (~50-80 ms/frame, ~15 FPS)"
            return True
        except Exception as ex:
            self.is_ready = False
            self.init_log = "GlobalEnhancer init failed: {}".format(ex)
            return False

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        if not self._ensure_init() or self._enhancer is None:
            return frame
        self._frame_cnt += 1
        if self._frame_cnt % 30 == 1 and self._profiler is not None:
            try:
                self._profile = self._profiler.profile(frame)
            except Exception:
                pass
        try:
            res = self._enhancer.enhance(frame, self._profile, use_temporal=True)
            if res.success and res.image is not None:
                return res.image
        except Exception:
            pass
        return frame


# ── AI: GFPGAN, best-effort for live streams ──────────────────────

class AIFilter(BaseFilter):
    """
    Best-effort GFPGAN filter for live video.

    Strategy to keep the preview responsive on CPU:
        1. Downscale the incoming frame so AI sees at most `max_size` px.
        2. Run GFPGAN only every `skip_frames` frames; in-between frames
           return the last cached output (or classical beauty filter).
        3. If the pipeline is not ready (no torch, no weights), silently
           fall back to BeautyFilter forever.

    This is **experimental** on CPU — use it for short demo segments.
    A CUDA GPU or a lighter face-restoration model is the real fix.
    """

    name = "AI"
    init_log = "(not initialised yet)"
    is_ready = False

    def __init__(self,
                 max_size: int = 512,
                 skip_frames: int = 10) -> None:
        self.max_size = int(max_size)
        self.skip = max(1, int(skip_frames))
        self._frame_count = 0
        self._cached: Optional[np.ndarray] = None
        self._cached_shape = None
        self._fallback = BeautyFilter()
        self._pipeline = None
        self._init_tried = False

    def reset(self) -> None:
        self._cached = None
        self._cached_shape = None
        self._frame_count = 0

    def set_params(self, **kwargs) -> None:
        """Tune AI filter live. Accepts: skip_frames (int), max_size (int)."""
        if "skip_frames" in kwargs:
            self.skip = max(1, int(kwargs["skip_frames"]))
        if "max_size" in kwargs:
            self.max_size = int(kwargs["max_size"])

    def _try_init(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._init_tried:
            return False
        self._init_tried = True
        try:
            from pipeline import ImageEnhancementPipeline  # lazy
            self._pipeline = ImageEnhancementPipeline(fidelity_weight=0.4)
            self.is_ready  = True
            self.init_log  = ("GFPGAN pipeline ready — face-restore every "
                              "{} frames @ {}px".format(self.skip, self.max_size))
            return True
        except Exception as ex:
            self.is_ready = False
            self.init_log = ("AI pipeline failed to init -> using BeautyFilter "
                             "fallback. Reason: {}".format(ex))
            return False

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        self._frame_count += 1
        # Every frame gets a cheap classical pass for display continuity.
        live = self._fallback.apply(frame)

        run_ai = (self._frame_count % self.skip) == 0
        if not run_ai and self._cached is not None:
            # Blend the cached AI face back in so it still looks "AI'd"
            if self._cached_shape == live.shape:
                return cv2.addWeighted(self._cached, 0.55, live, 0.45, 0.0)
            return live

        if not self._try_init() or self._pipeline is None:
            return live

        try:
            h, w = frame.shape[:2]
            scale = 1.0
            if max(h, w) > self.max_size:
                scale = self.max_size / float(max(h, w))
                small = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            else:
                small = frame

            t0 = time.time()
            res = self._pipeline.restoreImage(
                small, fidelity_weight=0.4, only_center_face=True)
            _dt = time.time() - t0

            if res is not None and res.restored is not None:
                out = res.restored
                if scale < 1.0:
                    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
                self._cached = out
                self._cached_shape = out.shape
                return out
        except Exception:
            return live

        return live


# ── Registry ──────────────────────────────────────────────────────

FILTERS = {
    "OFF":     NoFilter,
    "BEAUTY":  BeautyFilter,
    "ENHANCE": EnhanceLiveFilter,
    "AI":      AIFilter,
}


def make_filter(name: str) -> BaseFilter:
    """Factory — returns the matching filter instance, or NoFilter."""
    cls = FILTERS.get((name or "").upper(), NoFilter)
    return cls()
