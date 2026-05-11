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

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def reset(self) -> None:
        """Clear any internal cache."""
        pass


class NoFilter(BaseFilter):
    name = "OFF"


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

    def __init__(self,
                 smooth_strength: float = 0.65,
                 clarity: float = 0.60,
                 warmth: float = 0.35) -> None:
        self.smooth  = float(np.clip(smooth_strength, 0.0, 1.0))
        self.clarity = float(np.clip(clarity, 0.0, 1.0))
        self.warmth  = float(np.clip(warmth, 0.0, 1.0))
        self._clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))

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
    """
    Hybrid live enhancement: classical every frame + GFPGAN face every 25 frames.

    Every frame  : medianBlur + brightness + sharpen  (~8 ms, smooth display)
    Every 25 fr  : GFPGAN restores the face crop and pastes it back in (~3-5 s)

    Result: face region shows AI quality, background is classically enhanced.
    The face cache stays visible between AI updates for smooth display.
    """

    name    = "ENHANCE"
    AI_SKIP = 25

    def __init__(self) -> None:
        self._restorer   = None
        self._cascade    = None
        self._ov         = None   # OpenVINO Intel GPU enhancer
        self._count      = 0
        self._face_patch = None
        self._init_done  = False

    def reset(self) -> None:
        self._count      = 0
        self._face_patch = None

    # ── init (lazy, first frame) ──────────────────────────────────────

    def _init(self) -> None:
        self._init_done = True
        # Haar cascade for fast face detection (~3 ms)
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            c = cv2.CascadeClassifier(path)
            self._cascade = c if not c.empty() else None
        except Exception:
            pass
        # OpenVINO Intel GPU super-resolution (primary AI path)
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from openvino_enhancer import get_enhancer
            self._ov = get_enhancer(device="AUTO")
            if self._ov.ready:
                print("[ENHANCE] OpenVINO Intel GPU SR aktif")
            else:
                self._ov = None
        except Exception as e:
            self._ov = None
            print(f"[ENHANCE] OpenVINO yok: {e}")
        # GFPGAN restorer (face restoration fallback)
        try:
            from face_restorer import FaceRestorer
            self._restorer = FaceRestorer(fidelity_weight=0.20)
            print("[ENHANCE] GFPGAN face restorer ready")
        except Exception as e:
            print(f"[ENHANCE] GFPGAN unavailable: {e}")

    # ── fast classical pass (every frame) ────────────────────────────

    @staticmethod
    def _classical(frame: np.ndarray) -> np.ndarray:
        out  = cv2.medianBlur(frame, 3)
        out  = cv2.convertScaleAbs(out, alpha=1.42, beta=28)
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        return cv2.addWeighted(out, 1.28, blur, -0.28, 0)

    # ── main ─────────────────────────────────────────────────────────

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        if not self._init_done:
            self._init()

        self._count += 1

        # OpenVINO Intel GPU path (every frame when model is ready)
        if self._ov is not None and self._ov.ready:
            try:
                base = self._ov.enhance(frame)
                # Apply classical brightness on top of SR result
                base = cv2.convertScaleAbs(base, alpha=1.20, beta=15)
            except Exception:
                base = self._classical(frame)
        else:
            base = self._classical(frame)

        # Every AI_SKIP frames: GFPGAN on the face region
        if (self._restorer is not None
                and self._cascade is not None
                and self._count % self.AI_SKIP == 1):
            try:
                h, w = frame.shape[:2]
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4,
                    minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces) > 0:
                    # Largest face
                    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                    pad = int(max(fw, fh) * 0.40)
                    x1, y1 = max(0, fx - pad), max(0, fy - pad)
                    x2, y2 = min(w, fx + fw + pad), min(h, fy + fh + pad)
                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] > 80 and crop.shape[1] > 80:
                        res = self._restorer.restore(
                            crop, fidelity_weight=0.20, only_center_face=True)
                        if res.success and res.restored is not None:
                            patch = cv2.resize(res.restored, (x2 - x1, y2 - y1))
                            self._face_patch = (x1, y1, x2, y2, patch)
            except Exception as e:
                print(f"[ENHANCE] AI error: {e}")

        # Paste cached AI face over classical base
        if self._face_patch is not None:
            x1, y1, x2, y2, patch = self._face_patch
            fh, fw = patch.shape[:2]
            if fh == y2 - y1 and fw == x2 - x1:
                roi = base[y1:y2, x1:x2]
                base[y1:y2, x1:x2] = cv2.addWeighted(patch, 0.80, roi, 0.20, 0)

        return base


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

    def _try_init(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._init_tried:
            return False
        self._init_tried = True
        try:
            from pipeline import ImageEnhancementPipeline  # lazy
            self._pipeline = ImageEnhancementPipeline(fidelity_weight=0.4)
            return True
        except Exception:
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
