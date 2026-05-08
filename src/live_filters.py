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

import queue
import threading
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
                 smooth_strength: float = 0.72,
                 clarity: float = 0.68,
                 warmth: float = 0.38) -> None:
        self.smooth  = float(np.clip(smooth_strength, 0.0, 1.0))
        self.clarity = float(np.clip(clarity, 0.0, 1.0))
        self.warmth  = float(np.clip(warmth, 0.0, 1.0))
        self._auto = True
        self._frame_cnt = 0
        self._profiler = None
        self._profile = None
        # Process BEAUTY at a smaller resolution for live FPS, then upscale and blend.
        self._proc_max = 640
        # Default CLAHE is intentionally mild; aggressive CLAHE amplifies noise
        # on low-quality webcams (exact issue in the screenshots).
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))

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
        if "auto" in kwargs:
            self._auto = bool(kwargs["auto"])

    def _ensure_profiler(self) -> None:
        if self._profiler is not None:
            return
        try:
            from profiler import ImageProfiler  # lazy, lightweight
            self._profiler = ImageProfiler()
        except Exception:
            self._profiler = None

    @staticmethod
    def _highlight_ratio(frame: np.ndarray) -> float:
        """Fraction of very bright pixels (backlit detection)."""
        try:
            small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            return float(np.mean(g >= 245))
        except Exception:
            return 0.0

    @staticmethod
    def _midtone_mask(frame: np.ndarray) -> np.ndarray:
        """Float32 mask [0..1] — apply enhancement mostly on midtones."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = ((g >= 25) & (g <= 230)).astype(np.float32)
        return cv2.GaussianBlur(m, (0, 0), 3.0)

    @staticmethod
    def _skin_mask(frame: np.ndarray) -> np.ndarray:
        """Float32 mask [0..1] — cheap skin proxy for face/hand emphasis."""
        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            _, cr, cb = cv2.split(ycrcb)
            # Classic YCrCb skin box; tuned a bit wider for webcams.
            m = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).astype(np.uint8) * 255
            # Fill small holes / remove speckles -> avoids patchy artifacts.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
            m = cv2.GaussianBlur(m, (0, 0), 7.0)
            return (m.astype(np.float32) / 255.0)
        except Exception:
            return np.ones(frame.shape[:2], dtype=np.float32)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        out = frame

        # Downscale for processing if needed (keep display resolution intact).
        h0, w0 = out.shape[:2]
        m0 = max(h0, w0)
        proc = out
        scale = 1.0
        if m0 > self._proc_max:
            scale = self._proc_max / float(m0)
            proc = cv2.resize(
                out,
                (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                interpolation=cv2.INTER_AREA,
            )

        # Dynamic tuning: adapt strength to the scene, but keep it stable.
        self._frame_cnt += 1
        if self._auto and (self._frame_cnt % 20 == 1):
            self._ensure_profiler()
            if self._profiler is not None:
                try:
                    self._profile = self._profiler.profile(proc)
                except Exception:
                    self._profile = None
        hi = self._highlight_ratio(proc) if self._auto else 0.0
        if self._auto and self._profile is not None:
            pr = self._profile
            # Low-light -> more smoothing, slightly less local-contrast.
            if pr.is_low_light:
                self.smooth = float(np.clip(0.80, 0.0, 1.0))
                self.clarity = float(np.clip(0.55, 0.0, 1.0))
                self.warmth = float(np.clip(0.42, 0.0, 1.0))
            # Noisy/blurry -> avoid crunch: more smoothing, less sharpening/CLAHE.
            if pr.is_noisy or pr.is_blurry:
                self.smooth = float(np.clip(max(self.smooth, 0.80), 0.0, 0.92))
                self.clarity = float(np.clip(min(self.clarity, 0.60), 0.40, 0.75))
            # Clean frames: allow a bit more "phone-like" pop in midtones.
            if (not pr.is_low_light) and (not pr.is_noisy) and (not pr.is_blurry) and (not getattr(pr, "is_overexposed", False)):
                self.clarity = float(np.clip(max(self.clarity, 0.70), 0.0, 0.82))

        # Backlit/highlight scenes: CLAHE + sharpen create halos/noise.
        # Reduce clarity aggressively when highlights dominate.
        if hi > 0.03:
            self.clarity = float(np.clip(min(self.clarity, 0.45), 0.30, 0.60))

        mid = self._midtone_mask(proc) if self._auto else None
        skin = self._skin_mask(proc) if self._auto else None
        # Emphasis mask: prefer skin midtones when reliable; otherwise fall back to midtones.
        em = None
        if mid is not None and skin is not None:
            try:
                cov = float(np.mean(skin > 0.25))
            except Exception:
                cov = 0.0
            if cov < 0.01:
                # Skin detector failed (common in backlit / WB shifts) -> safe fallback.
                em = mid
            else:
                # Gentle weighting (avoid hard, patchy boundaries).
                em = np.clip(mid * (0.55 + 0.45 * skin), 0.0, 1.0).astype(np.float32)
        elif mid is not None:
            em = mid

        # 1) Edge-preserving bilateral smoothing
        if self.smooth > 0:
            d  = 7 if max(proc.shape[:2]) < 900 else 9
            sm = cv2.bilateralFilter(proc, d=d, sigmaColor=50, sigmaSpace=12)
            a  = 0.40 + 0.50 * self.smooth
            proc = cv2.addWeighted(sm, a, proc, 1.0 - a, 0.0)

        # 2) CLAHE on L — stronger clip for visible contrast pop
        if self.clarity > 0:
            lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            # Stronger local contrast when clarity is high (presentation-friendly).
            # Recreate CLAHE only when the clip changes materially.
            # Cap CLAHE; never go extreme in live (noise amplification).
            clip = 1.9 + 1.8 * float(self.clarity)
            try:
                if abs(getattr(self, "_clahe_clip", 0.0) - clip) > 0.25:
                    self._clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
                    self._clahe_clip = float(clip)
            except Exception:
                pass
            L_eq = self._clahe.apply(L)
            a    = 0.5 + 0.5 * self.clarity
            L2   = cv2.addWeighted(L_eq, a, L, 1.0 - a, 0.0)
            if em is not None:
                L = (L.astype(np.float32) * (1.0 - em) + L2.astype(np.float32) * em).astype(np.uint8)
            else:
                L = L2
            proc  = cv2.cvtColor(cv2.merge((L, A, B)), cv2.COLOR_LAB2BGR)

        # 3) Two-scale unsharp mask
        if self.clarity > 0:
            img_f  = proc.astype(np.float32)
            b_fine = cv2.GaussianBlur(img_f, (0, 0), 0.8)
            b_mid  = cv2.GaussianBlur(img_f, (0, 0), 2.0)
            # Reduce sharpen amount to avoid "noisy edge" look.
            amt    = 0.30 + 0.50 * self.clarity
            sharp  = img_f + amt * (img_f - b_fine) + amt * 0.4 * (img_f - b_mid)
            sharp_u = np.clip(sharp, 0, 255).astype(np.uint8)
            if em is not None:
                proc = (proc.astype(np.float32) * (1.0 - em[..., None]) +
                        sharp_u.astype(np.float32) * em[..., None]).astype(np.uint8)
            else:
                proc = sharp_u

        # Optional: in backlit scenes, pull highlights back slightly
        # to avoid the "posterized curtain" look.
        if hi > 0.03:
            hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)
            V_f = V.astype(np.float32)
            # compress top end only
            V_f = np.where(V_f > 220.0, 220.0 + (V_f - 220.0) * 0.55, V_f)
            V2 = np.clip(V_f, 0, 255).astype(np.uint8)
            proc = cv2.cvtColor(cv2.merge((H, S, V2)), cv2.COLOR_HSV2BGR)

        # 4) Vibrance — boost desaturated pixels
        if self.clarity > 0.2:
            hsv   = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)
            S_f   = S.astype(np.float32)
            wt    = 1.0 - S_f / 255.0
            # Keep vibrance mild in live (avoid colour noise in shadows).
            S_new = np.clip(S_f + self.clarity * 55.0 * wt, 0, 255).astype(np.uint8)
            out_v = cv2.cvtColor(cv2.merge([H, S_new, V]), cv2.COLOR_HSV2BGR)
            if em is not None:
                proc = (proc.astype(np.float32) * (1.0 - em[..., None]) +
                        out_v.astype(np.float32) * em[..., None]).astype(np.uint8)
            else:
                proc = out_v

        # 5) Warm tone via LUT
        if self.warmth > 0:
            Bc, G, R = cv2.split(proc)
            R2 = cv2.LUT(R, self._WARM_LUT_R)
            B2 = cv2.LUT(Bc, self._WARM_LUT_B)
            a  = self.warmth
            R  = cv2.addWeighted(R2, a, R, 1.0 - a, 0.0)
            Bc = cv2.addWeighted(B2, a, Bc, 1.0 - a, 0.0)
            proc = cv2.merge((Bc, G, R))

        if scale < 1.0 and proc.shape[:2] != (h0, w0):
            proc = cv2.resize(proc, (w0, h0), interpolation=cv2.INTER_LINEAR)
            if em is not None:
                em = cv2.resize(em, (w0, h0), interpolation=cv2.INTER_LINEAR)

        if em is None:
            return proc
        # Final blend: only push enhancements into midtones.
        return (out.astype(np.float32) * (1.0 - em[..., None]) +
                proc.astype(np.float32) * em[..., None]).astype(np.uint8)


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
        # Process at a smaller resolution for real-time; upscale back for display.
        # This keeps the ENHANCE filter usable on CPU (target ~25-30 FPS).
        # Defaults tuned for CPU: keep preview responsive and use caching.
        # (Higher run_every / lower proc_max => higher FPS, less "live" response.)
        self._proc_max  = 480
        self._run_every = 3
        self._cached: Optional[np.ndarray] = None
        self._auto = True
        self._midtone_only = True
        # Stabilize preview: don't run the heavy path too frequently even if
        # the GUI timer ticks faster. This reduces FPS spikes/jitter.
        self._min_interval_ms: float = 60.0
        self._last_run_ts: float = 0.0
        # rembg-based person segmentation. When available, ENHANCE only
        # touches the subject so bright windows / lamps in the background
        # stay untouched. Falls back to the previous skin/midtone-only
        # behaviour if rembg or its model can't be loaded.
        self._person_segmenter = None
        self._person_mask_cache: Optional[np.ndarray] = None
        self._person_segment_enabled: bool = True

    def reset(self) -> None:
        if self._enhancer is not None:
            self._enhancer.reset()
        self._frame_cnt = 0
        self._profile   = None

    def set_params(self, **kwargs) -> None:
        """Tune the global stack live. Accepts: intensity 0..1.30."""
        if "intensity" in kwargs:
            v = float(np.clip(kwargs["intensity"], 0.0, 1.30))
            self._intensity = v
            if self._enhancer is not None:
                cfg = self._enhancer.cfg
                # Presentation mapping; capped to avoid drift / banding on webcams.
                cfg.white_balance  = min(1.00, 0.94 * v)
                cfg.shadow_lift    = min(0.98, 0.92 * v)
                cfg.denoise        = min(0.78, 0.67 * v)
                cfg.clahe_strength = min(0.98, 0.92 * v)
                cfg.hdr_tone       = min(0.88, 0.80 * v)
                cfg.sharpen        = min(0.95, 0.90 * v)
                cfg.vibrance       = min(0.92, 0.84 * v)
                cfg.film_look      = min(0.82, 0.70 * v)
        if "auto" in kwargs:
            self._auto = bool(kwargs["auto"])
        if "proc_max" in kwargs:
            try:
                self._proc_max = int(max(160, min(1280, int(kwargs["proc_max"]))))
            except Exception:
                pass
        if "run_every" in kwargs:
            try:
                self._run_every = int(max(1, min(6, int(kwargs["run_every"]))))
            except Exception:
                pass

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
            self.init_log  = (
                "GlobalEnhancer ready (cached) — proc_max={}px, every {} frames"
                .format(self._proc_max, self._run_every)
            )
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
        if self._run_every > 1 and (self._frame_cnt % self._run_every) != 0:
            return self._cached if self._cached is not None else frame

        # Time-budget guard: keep preview stable by enforcing a minimum interval.
        now = time.perf_counter()
        if self._cached is not None and self._min_interval_ms > 0:
            if self._last_run_ts > 0 and (now - self._last_run_ts) * 1000.0 < self._min_interval_ms:
                return self._cached

        if self._frame_cnt % 20 == 1 and self._profiler is not None:
            try:
                self._profile = self._profiler.profile(frame)
            except Exception:
                pass
        if self._auto and self._profile is not None and self._enhancer is not None:
            pr = self._profile
            # Scene-adaptive intensity: stronger on clean frames, safer on low-light/noisy/blurry.
            base = float(np.clip(self._intensity, 0.0, 1.0))
            if pr.is_overexposed:
                v = max(0.45, base * 0.70)
            elif pr.is_low_light or pr.is_noisy or pr.is_blurry:
                v = max(0.50, base * 0.80)
            else:
                v = min(1.30, base * 1.12)
            self.set_params(intensity=v)

        # Refresh the person matte once per heavy run. ~5-10 ms on CUDA
        # at 320 px, so it sits comfortably inside the existing per-tick
        # budget; cached/skipped frames above never reach this point.
        if self._person_segment_enabled:
            if self._person_segmenter is None:
                try:
                    from person_segmenter import PersonSegmenter
                    self._person_segmenter = PersonSegmenter(
                        model_name="u2net_human_seg", proc_max=320)
                except Exception:
                    self._person_segment_enabled = False
            if self._person_segmenter is not None:
                self._person_mask_cache = self._person_segmenter.get_mask(frame)
        try:
            img = frame
            h, w = img.shape[:2]
            m = max(h, w)
            if m > self._proc_max:
                s = self._proc_max / max(1, m)
                img_small = cv2.resize(
                    img, (max(1, int(w * s)), max(1, int(h * s))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                img_small = img

            res = self._enhancer.enhance(img_small, self._profile, use_temporal=True)
            if res.success and res.image is not None:
                out = res.image
                if out.shape[:2] != (h, w):
                    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
                if self._midtone_only:
                    # Apply enhancement mostly on midtones to avoid halos/banding in highlights
                    # (e.g. curtains / backlit windows).
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mid = ((g >= 25) & (g <= 230)).astype(np.float32)
                    mid = cv2.GaussianBlur(mid, (0, 0), 3.0)
                    # Prefer skin/face areas: reduces "background damage" in backlit scenes.
                    try:
                        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                        _, cr, cb = cv2.split(ycrcb)
                        skin = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).astype(np.uint8) * 255
                        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k, iterations=1)
                        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k, iterations=2)
                        skin = cv2.GaussianBlur(skin, (0, 0), 7.0).astype(np.float32) / 255.0
                        cov = float(np.mean(skin > 0.25))
                        em = mid if cov < 0.01 else np.clip(mid * (0.55 + 0.45 * skin), 0.0, 1.0).astype(np.float32)
                    except Exception:
                        em = mid
                    # Person-only restriction: when rembg gave us a matte,
                    # multiply it in so background pixels (alpha~0) get
                    # the original frame untouched. No matte → fall back
                    # to the unrestricted midtone+skin mask above.
                    pm = self._person_mask_cache
                    if pm is not None and pm.shape[:2] == em.shape[:2]:
                        em = (em * pm).astype(np.float32)
                    out = (img.astype(np.float32) * (1.0 - em[..., None]) +
                           out.astype(np.float32) * em[..., None]).astype(np.uint8)
                self._cached = out
                self._last_run_ts = now
                return out
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
                 max_size: int = 384,
                 skip_frames: int = 15) -> None:
        self.max_size = int(max_size)
        self.skip = max(1, int(skip_frames))
        self._frame_count = 0
        self._cached: Optional[np.ndarray] = None
        self._cached_shape = None
        self._fallback = BeautyFilter()
        self._pipeline = None
        self._init_tried = False
        self._req: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=1)
        self._busy = False
        self._stop = False
        self._th: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        self._cached = None
        self._cached_shape = None
        self._frame_count = 0
        with self._lock:
            self._busy = False

    def set_params(self, **kwargs) -> None:
        """Tune AI filter live. Accepts: skip_frames (int), max_size (int)."""
        if "skip_frames" in kwargs:
            self.skip = max(1, int(kwargs["skip_frames"]))
        if "max_size" in kwargs:
            self.max_size = int(kwargs["max_size"])

    def _ensure_thread(self) -> None:
        if self._th is not None:
            return
        self._stop = False

        # Try init immediately so GUI banner reflects reality.
        try:
            self._try_init()
        except Exception:
            pass

        def _loop() -> None:
            while not self._stop:
                try:
                    item = self._req.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None or self._stop:
                    break
                frame_small = item
                with self._lock:
                    self._busy = True
                try:
                    if not self._try_init() or self._pipeline is None:
                        continue
                    res = self._pipeline.restoreImage(
                        frame_small, fidelity_weight=0.4, only_center_face=True
                    )
                    if res is not None and res.restored is not None:
                        with self._lock:
                            self._cached = res.restored
                            self._cached_shape = res.restored.shape
                except Exception:
                    pass
                finally:
                    with self._lock:
                        self._busy = False

        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()

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

        self._ensure_thread()
        self._frame_count += 1
        # Every frame gets a cheap classical pass for display continuity.
        live = self._fallback.apply(frame)

        # If we have a cached AI frame at the current resolution, blend it in.
        with self._lock:
            cached = self._cached
            cached_shape = self._cached_shape
            busy = self._busy

        if cached is not None and cached_shape == live.shape:
            # Make the AI effect more visible when cached is available.
            live = cv2.addWeighted(cached, 0.65, live, 0.35, 0.0)

        # Trigger an async AI refresh every N frames (single-slot queue).
        run_ai = (self._frame_count % self.skip) == 0
        # On first use (no cache yet), submit immediately so the user sees
        # the AI effect quickly instead of waiting skip_frames.
        if cached is None:
            run_ai = True
        if run_ai and not busy:
            try:
                h, w = frame.shape[:2]
                if max(h, w) > self.max_size:
                    s = self.max_size / float(max(h, w))
                    small = cv2.resize(
                        frame,
                        (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = frame
                try:
                    self._req.put_nowait(small.copy())
                except queue.Full:
                    pass
            except Exception:
                pass

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
