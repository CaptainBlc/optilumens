"""
Live Capture — Stage 4

CMPE 491 Senior Design Project.

Webcam-driven live mode: holds the user still for ~3 seconds, then runs
the full enhancement pipeline (GFPGAN → semantic parse → region enhance)
on the captured frame in a background thread.  The live preview never
freezes; the result lands in the right half of a side-by-side BEFORE/
AFTER view as soon as the worker finishes.

Controls (live window has focus):
    Q / Esc      quit
    S            save the latest enhanced result to outputs/
    SPACE        toggle full-screen view of the latest enhanced result
    R            re-trigger enhancement on the current still frame
    Up / Down    increase / decrease hold duration

This module is standalone — runs without the PyQt GUI.  Stage 5 will
embed the same primitives into the main window.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline import EnhancementResult, ImageEnhancementPipeline
from region_enhancer import RegionConfig


# ── Live-tuned defaults ──────────────────────────────────────────────
# Webcam frames are noisier and lower-res than studio photos.  Stronger
# defaults so the difference is clearly visible on screen.
LIVE_REGION_CONFIG = RegionConfig(
    skin_smooth   = 0.40,    # was 0.65 — preserve face structure under bilateral
    eye_sharpen   = 0.85,    # punchier eyes (covers glasses too via eye_g)
    eye_brighten  = 0.28,
    lip_vibrance  = 0.55,
    lip_warmth    = 0.32,
    brow_contrast = 0.55,
    nose_sharpen  = 0.55,
    hair_denoise  = 0.00,
    face_clarity  = 0.55,    # whole-face detail boost — features pop
    morph_open_px = 3,
    feather_px    = 9,
)


# ── Stability detection ──────────────────────────────────────────────

@dataclass
class StabilityConfig:
    motion_threshold: float = 2.5    # mean abs-diff per pixel (uint8 scale)
    hold_seconds:     float = 3.0    # how long user must stay still
    motion_size:      int   = 160    # downscale width for motion calc
    cooldown_seconds: float = 2.0    # min gap between auto-triggers


class StabilityDetector:
    """Tracks frame-to-frame motion; emits a 'stable' event after holding still."""

    def __init__(self, cfg: StabilityConfig) -> None:
        self.cfg = cfg
        self._prev: Optional[np.ndarray] = None
        self._stable_since: Optional[float] = None
        self._last_trigger: float = 0.0
        self._motion: float = 0.0

    def reset(self) -> None:
        self._prev = None
        self._stable_since = None

    def update(self, frame_bgr: np.ndarray, now: float) -> Tuple[bool, float, float]:
        """
        Feed one frame.

        Returns
        -------
        (triggered, motion_score, stable_seconds)
            triggered       — True on the single frame that crosses the hold
            motion_score    — current mean |Δpixel| (uint8 scale)
            stable_seconds  — how long the scene has been below threshold
        """
        h, w = frame_bgr.shape[:2]
        target_w = self.cfg.motion_size
        scale = target_w / max(1, w)
        small = cv2.resize(frame_bgr, (target_w, max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev is None:
            self._prev = gray
            return False, 0.0, 0.0

        diff = cv2.absdiff(gray, self._prev)
        self._motion = float(diff.mean())
        self._prev = gray

        moving = self._motion > self.cfg.motion_threshold

        if moving:
            self._stable_since = None
            return False, self._motion, 0.0

        if self._stable_since is None:
            self._stable_since = now
        held = now - self._stable_since

        triggered = (
            held >= self.cfg.hold_seconds
            and (now - self._last_trigger) >= self.cfg.cooldown_seconds
        )
        if triggered:
            self._last_trigger = now
            self._stable_since = now      # require fresh hold for next trigger

        return triggered, self._motion, held


# ── Background enhancement worker ────────────────────────────────────

class EnhanceWorker(threading.Thread):
    """
    Single-slot background pipeline runner.

    submit(frame) drops the request if a previous one is still in flight,
    so the main thread is never blocked.
    """

    def __init__(self, pipeline: ImageEnhancementPipeline) -> None:
        super().__init__(daemon=True)
        self._pipe = pipeline
        self._req: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._result: Optional[EnhancementResult] = None
        self._busy = False
        self._stop = threading.Event()

    def submit(self, frame: np.ndarray) -> bool:
        try:
            self._req.put_nowait(frame.copy())
            return True
        except queue.Full:
            return False

    def latest(self) -> Optional[EnhancementResult]:
        with self._lock:
            return self._result

    def busy(self) -> bool:
        return self._busy

    def stop(self) -> None:
        self._stop.set()
        try:
            self._req.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._req.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            self._busy = True
            try:
                res = self._pipe.restoreImage(frame)
            except Exception as e:
                res = EnhancementResult(
                    log=["[ERROR] Live worker: {}".format(e)], success=False)
            with self._lock:
                self._result = res
            self._busy = False


# ── HUD rendering ────────────────────────────────────────────────────

def _draw_hud(frame: np.ndarray, motion: float, held: float, hold_target: float,
              state: str, busy: bool, fps: float) -> np.ndarray:
    """Draw a top status bar and bottom progress bar onto a copy of the frame."""
    out = frame.copy()
    H, W = out.shape[:2]

    # Top bar
    bar_h = 32
    cv2.rectangle(out, (0, 0), (W, bar_h), (0, 0, 0), thickness=-1)
    txt = "{}  motion={:.2f}  hold={:.1f}/{:.1f}s  fps={:.0f}{}".format(
        state, motion, held, hold_target, fps, "  [WORKING]" if busy else "")
    cv2.putText(out, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 200) if not busy else (0, 200, 255), 1, cv2.LINE_AA)

    # Bottom progress bar — fills as user holds still
    progress = max(0.0, min(1.0, held / max(0.05, hold_target)))
    pad = 12
    bw, bh = W - 2 * pad, 8
    by = H - pad - bh
    cv2.rectangle(out, (pad, by), (pad + bw, by + bh), (40, 40, 40), -1)
    fill = int(bw * progress)
    color = (60, 220, 60) if progress >= 1.0 else (60, 180, 220)
    cv2.rectangle(out, (pad, by), (pad + fill, by + bh), color, -1)

    return out


def _draw_split(live: np.ndarray, enhanced: Optional[np.ndarray],
                motion: float, held: float, hold_target: float,
                state: str, busy: bool, fps: float) -> np.ndarray:
    """Side-by-side BEFORE | AFTER layout with HUD top + progress bar bottom.

    Left half : current live frame
    Right half: latest enhanced result (or placeholder if none yet)
    """
    H, W = live.shape[:2]
    canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)

    canvas[:, :W] = live
    if enhanced is not None:
        # Match height to live frame
        eh, ew = enhanced.shape[:2]
        scale = H / max(1, eh)
        rw = max(1, int(ew * scale))
        if rw == W:
            canvas[:, W:] = cv2.resize(enhanced, (W, H),
                                       interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(enhanced, (rw, H),
                                 interpolation=cv2.INTER_AREA)
            # center-crop or pad to W
            if rw > W:
                xo = (rw - W) // 2
                canvas[:, W:] = resized[:, xo:xo + W]
            else:
                canvas[:, W:W + rw] = resized
    else:
        # Placeholder
        cv2.rectangle(canvas, (W, 0), (2 * W, H), (15, 15, 15), -1)
        msg1 = "Hold still {:.0f}s".format(hold_target)
        msg2 = "to render enhancement here"
        (tw1, th1), _ = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        (tw2, th2), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(canvas, msg1,
                    (W + (W - tw1) // 2, H // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2,
                    cv2.LINE_AA)
        cv2.putText(canvas, msg2,
                    (W + (W - tw2) // 2, H // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1,
                    cv2.LINE_AA)

    # Center divider
    cv2.line(canvas, (W, 0), (W, H), (0, 255, 200), 2)

    # Side labels
    cv2.rectangle(canvas, (8, H - 36), (110, H - 12), (0, 0, 0), -1)
    cv2.putText(canvas, "BEFORE", (16, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (W + 8, H - 36), (W + 110, H - 12), (0, 0, 0), -1)
    cv2.putText(canvas, "AFTER", (W + 16, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 200), 1, cv2.LINE_AA)

    # Top HUD bar
    bar_h = 32
    cv2.rectangle(canvas, (0, 0), (2 * W, bar_h), (0, 0, 0), -1)
    txt = "{}  motion={:.2f}  hold={:.1f}/{:.1f}s  fps={:.0f}{}".format(
        state, motion, held, hold_target, fps, "  [WORKING]" if busy else "")
    cv2.putText(canvas, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 200) if not busy else (0, 200, 255), 1, cv2.LINE_AA)

    # Bottom progress bar (under both halves)
    progress = max(0.0, min(1.0, held / max(0.05, hold_target)))
    pad = 12
    bw, bh = 2 * W - 2 * pad, 8
    by = H - pad - bh - 30   # above the BEFORE/AFTER labels
    cv2.rectangle(canvas, (pad, by), (pad + bw, by + bh), (40, 40, 40), -1)
    fill = int(bw * progress)
    color = (60, 220, 60) if progress >= 1.0 else (60, 180, 220)
    cv2.rectangle(canvas, (pad, by), (pad + fill, by + bh), color, -1)

    return canvas


def _draw_inset(frame: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """Place a small enhanced thumbnail in the bottom-right corner."""
    out = frame
    H, W = out.shape[:2]
    th_w = max(160, W // 4)
    eh, ew = enhanced.shape[:2]
    th_h = int(eh * (th_w / max(1, ew)))
    th = cv2.resize(enhanced, (th_w, th_h), interpolation=cv2.INTER_AREA)
    pad = 12
    x0, y0 = W - th_w - pad, H - th_h - pad - 24
    out[y0:y0 + th_h, x0:x0 + th_w] = th
    cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + th_w, y0 + th_h),
                  (0, 255, 200), 1)
    cv2.putText(out, "ENHANCED  (S=save  SPACE=full)",
                (x0, y0 + th_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 200), 1, cv2.LINE_AA)
    return out


# ── Main session ─────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    camera_index: int = 0
    output_dir:   str = "outputs"
    window_name:  str = "Live Enhancement — CMPE491"
    request_w:    int = 1280
    request_h:    int = 720


class LiveSession:
    """Camera capture + stability detection + background enhancement."""

    def __init__(self,
                 pipeline: Optional[ImageEnhancementPipeline] = None,
                 stability: Optional[StabilityConfig] = None,
                 session: Optional[SessionConfig] = None) -> None:
        if pipeline is None:
            # Live mode uses a more aggressive RegionConfig — webcam input
            # is noisier than photos so subtle defaults disappear.
            pipeline = ImageEnhancementPipeline(region_config=LIVE_REGION_CONFIG)
        self.pipeline = pipeline
        self.s_cfg = stability if stability is not None else StabilityConfig()
        self.cfg   = session if session is not None else SessionConfig()
        self._detector = StabilityDetector(self.s_cfg)
        self._worker   = EnhanceWorker(self.pipeline)

    def run(self) -> int:
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera index {}".format(self.cfg.camera_index))
            return 2

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.request_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.request_h)

        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        self._worker.start()

        last_enhanced_id: int = -1     # to detect new results
        latest_enhanced: Optional[np.ndarray] = None
        full_view: bool = False
        was_busy: bool = False
        fps_t = time.time()
        fps = 0.0
        n_done = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WARN] frame grab failed")
                    time.sleep(0.05)
                    continue

                now = time.time()
                busy = self._worker.busy()

                # Busy-lock: while the pipeline runs, don't accumulate hold
                # time — user must restart the hold after the result lands.
                if busy:
                    self._detector.reset()
                    triggered, motion, held = False, 0.0, 0.0
                    # Still measure motion for the HUD by feeding the detector
                    # one frame so _prev tracks; we just ignore the output.
                    _, motion, _ = self._detector.update(frame, now)
                    self._detector._stable_since = None   # never accumulate
                else:
                    if was_busy:                          # just finished
                        self._detector.reset()
                    triggered, motion, held = self._detector.update(frame, now)

                # State labelling
                if busy:
                    state = "ENHANCING"
                elif triggered:
                    state = "TRIGGERED"
                elif held >= self.s_cfg.hold_seconds * 0.5:
                    state = "HOLDING"
                else:
                    state = "MOVING"

                if triggered and not busy:
                    if self._worker.submit(frame):
                        state = "ENHANCING"

                was_busy = busy

                # Pull latest result (non-blocking)
                res = self._worker.latest()
                if res is not None and res.success and res.restored is not None:
                    rid = id(res)
                    if rid != last_enhanced_id:
                        last_enhanced_id = rid
                        latest_enhanced = res.restored
                        n_done += 1

                # FPS
                dt = now - fps_t
                fps_t = now
                if dt > 0:
                    fps = 0.85 * fps + 0.15 * (1.0 / dt)

                # Render
                if full_view and latest_enhanced is not None:
                    show = cv2.resize(
                        latest_enhanced,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.putText(show,
                                "FULL VIEW — SPACE to return",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 200), 1, cv2.LINE_AA)
                else:
                    show = _draw_split(frame, latest_enhanced, motion, held,
                                       self.s_cfg.hold_seconds, state, busy, fps)

                cv2.imshow(self.cfg.window_name, show)

                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("s") and latest_enhanced is not None:
                    fn = os.path.join(
                        self.cfg.output_dir,
                        "live_{}.png".format(time.strftime("%Y%m%d_%H%M%S")),
                    )
                    cv2.imwrite(fn, latest_enhanced)
                    print("Saved: {}".format(fn))
                elif k == ord(" "):
                    full_view = not full_view
                elif k == ord("r"):
                    if not self._worker.busy():
                        self._worker.submit(frame)
                elif k == 82 or k == ord("+"):     # up arrow / +
                    self.s_cfg.hold_seconds = min(10.0, self.s_cfg.hold_seconds + 0.5)
                elif k == 84 or k == ord("-"):     # down arrow / -
                    self.s_cfg.hold_seconds = max(0.5, self.s_cfg.hold_seconds - 0.5)

        finally:
            self._worker.stop()
            cap.release()
            cv2.destroyAllWindows()

        print("Session ended.  enhanced={}  ".format(n_done))
        return 0


# ── Entry ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Live camera enhancement (Stage 4)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hold",   type=float, default=3.0,
                    help="seconds the user must stay still to trigger")
    ap.add_argument("--motion", type=float, default=2.5,
                    help="motion threshold (mean |Δpix| on uint8)")
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    sess = LiveSession(
        stability=StabilityConfig(
            motion_threshold=args.motion,
            hold_seconds=args.hold,
        ),
        session=SessionConfig(
            camera_index=args.camera,
            request_w=args.width,
            request_h=args.height,
        ),
    )
    sys.exit(sess.run())
