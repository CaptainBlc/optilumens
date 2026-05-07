"""
Real-Time Camera Capture Module — GFPGAN Enhancement System.

Implements Scenario 4 (Analysis Report §3.5.1):
    "Real-Time Image Capture (Future Extension)"

Provides OpenCV-based webcam access with safe enumeration, preview feed,
and snapshot capture. The captured frame feeds into the standard
ImageEnhancementPipeline for GFPGAN-based enhancement.

Design notes:
    - All VideoCapture calls happen on the main thread (OpenCV requirement
      on Windows when using DSHOW). GUI timer drives the polling loop.
    - Device enumeration is best-effort: opens indices 0..4, closes
      immediately. Works on Windows / macOS / Linux.
    - Returns standard BGR uint8 frames — same contract as cv2.imread().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class CameraInfo:
    """Describes an available camera device."""
    index: int
    name: str
    width: int
    height: int
    fps: float


def _backend_name(b: int) -> str:
    """Return a short human-readable name for an OpenCV backend constant."""
    names = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF:  "MSMF",
        cv2.CAP_V4L2:  "V4L2",
        cv2.CAP_AVFOUNDATION: "AVFoundation",
        cv2.CAP_ANY:   "any",
    }
    return names.get(b, "backend{}".format(b))


class CameraCapture:
    """
    Webcam wrapper — safe open/close, frame polling, snapshot.

    Usage:
        cam = CameraCapture()
        if cam.open(0):
            frame = cam.read()           # BGR uint8 (or None)
            cam.close()

    Attributes
    ----------
    is_open : bool
        True if a device is currently active.
    index : int or None
        Currently opened camera index.
    width, height : int
        Actual frame dimensions after open().
    fps : float
        Reported device FPS (may be 0 on some drivers).
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._index: Optional[int] = None
        self._w: int = 0
        self._h: int = 0
        self._fps: float = 0.0
        # When the GUI's filter is slow, OpenCV's internal buffer fills
        # up and frames lag a second or more behind reality.  We tell
        # OpenCV to keep only the latest frame; backends that don't
        # support this just ignore the call.
        self._wants_low_latency = True

    # ── enumeration ───────────────────────────────────────────────

    # Backend probe order: try several and use whichever opens the device.
    # On Windows, DSHOW is the historical default but recent drivers
    # (especially built-in laptop cams + UVC dongles) work better with
    # MSMF or even CAP_ANY. We try them in this order and pick the first
    # that returns a valid frame.
    @staticmethod
    def _backends_to_try() -> List[int]:
        import sys as _sys
        if _sys.platform == "win32":
            # IMPORTANT (Windows stability):
            # MSMF / CAP_ANY can trigger native backend crashes on some setups
            # (e.g. obsensor/UVC depth-camera paths, driver bugs). Python
            # try/except cannot catch those. DSHOW is the most predictable
            # option for a demo app.
            return [cv2.CAP_DSHOW]
        return [cv2.CAP_ANY]

    @staticmethod
    def list_cameras(max_devices: int = 5) -> List[CameraInfo]:
        """
        Probe device indices 0..max_devices-1 across multiple backends.

        Returns one CameraInfo per device that successfully delivers a
        real frame on at least one backend. Indices are deduplicated so a
        camera reachable on both DSHOW and MSMF only appears once.
        """
        found: List[CameraInfo] = []
        seen_indices: set = set()
        for i in range(max_devices):
            if i in seen_indices:
                continue
            for backend in CameraCapture._backends_to_try():
                cap = cv2.VideoCapture(i, backend)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue
                # Backend may say "open" without delivering frames. Verify.
                ok, frame = cap.read()
                if ok and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    found.append(CameraInfo(
                        index=i,
                        name="Camera {} ({})".format(i, _backend_name(backend)),
                        width=w, height=h, fps=fps,
                    ))
                    seen_indices.add(i)
                    cap.release()
                    break
                cap.release()
        return found

    # ── lifecycle ─────────────────────────────────────────────────

    def open(self, index: int = 0,
             width: Optional[int] = None,
             height: Optional[int] = None) -> bool:
        """
        Open camera at the given index, trying multiple backends.

        Returns True on success.
        """
        if self._cap is not None:
            self.close()

        for backend in self._backends_to_try():
            cap = cv2.VideoCapture(index, backend)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue

            # Some backends "open" without producing frames -- verify.
            ok, _ = cap.read()
            if not ok:
                cap.release()
                continue

            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Low-latency hint: keep only latest frame in driver buffer.
            # No-op on backends that don't expose CAP_PROP_BUFFERSIZE.
            if self._wants_low_latency:
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            self._cap = cap
            self._index = index
            self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            return True
        return False

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._index = None

    # ── frame access ──────────────────────────────────────────────

    def read(self, drain: bool = True) -> Optional[np.ndarray]:
        """
        Grab a single BGR uint8 frame. Returns None on failure.

        Parameters
        ----------
        drain : bool
            When True (default), discard any backlogged frames in the
            driver queue and return the *latest* one. This kills the
            lag that builds up when an upstream filter is slower than
            the camera's native frame rate (e.g. ENHANCE at ~16 FPS
            against a 30 FPS sensor would otherwise drift seconds
            behind reality).
        Safe to call at GUI timer frequency (e.g. 30 FPS).
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        if drain:
            # grab() is much cheaper than read() because it skips the
            # YUV->BGR conversion. Drop any backlog, then decode the
            # *next* frame.
            for _ in range(2):
                if not self._cap.grab():
                    break
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def snapshot(self) -> Optional[np.ndarray]:
        """
        Capture a high-quality still frame.

        Reads several frames in a row to let the auto-exposure / white-balance
        stabilize, then returns the last one. Typical gain: ~1-2 stops
        sharper than an immediate single-frame grab.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        frame = None
        for _ in range(5):
            ok, f = self._cap.read()
            if ok and f is not None:
                frame = f
        return frame

    # ── status ────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def index(self) -> Optional[int]:
        return self._index

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    @property
    def fps(self) -> float:
        return self._fps

    def info(self) -> str:
        if not self.is_open:
            return "Camera: (not open)"
        return "Camera {}: {}x{} @ {:.0f} FPS".format(
            self._index, self._w, self._h, self._fps)

    def __del__(self):
        self.close()
