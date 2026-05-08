"""
Person segmentation wrapper for live enhancement.

Uses rembg's u2net_human_seg model on CUDA via onnxruntime-gpu. Returns
a soft float32 mask in [0..1] where 1 = person, 0 = background. Lets
the live enhancement pipeline scope its work to the subject so bright
windows / lamps in the background stay untouched (the actual cause of
the "washed" look in well-lit rooms).

Design notes:
    - Segmentation runs at downscaled resolution (default 320 px on the
      long edge); the produced mask is resized back to the original
      shape with bilinear interpolation. ~5-10 ms on an RTX 4070.
    - All initialization is lazy; if the model can't load, get_mask()
      returns None and the caller is expected to fall back gracefully.
    - No exceptions propagate to the caller — segmentation must never
      take down the GUI's filter chain.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class PersonSegmenter:
    """
    Lazy CUDA-backed person matte producer.

    Parameters
    ----------
    model_name : str
        rembg model id. ``u2net_human_seg`` is human-specialised (best
        for portraits / webcam framing). ``u2net`` is the generic fallback.
    proc_max : int
        Long-edge size used for inference. Lower = faster, blurrier mask.
    prefer_gpu : bool
        Try CUDAExecutionProvider first; fall back to CPU if not present.
    """

    def __init__(self,
                 model_name: str = "u2net_human_seg",
                 proc_max: int = 320,
                 prefer_gpu: bool = True) -> None:
        self._model_name = str(model_name)
        self._proc_max = int(proc_max)
        self._prefer_gpu = bool(prefer_gpu)
        self._session = None
        self._init_tried = False
        self._device_name: str = "uninit"

    # ── lifecycle ───────────────────────────────────────────────────

    def _ensure(self) -> bool:
        if self._session is not None:
            return True
        if self._init_tried:
            return False
        self._init_tried = True
        try:
            from rembg import new_session
            providers = ["CPUExecutionProvider"]
            if self._prefer_gpu:
                try:
                    import onnxruntime as ort
                    avail = set(ort.get_available_providers())
                    if "CUDAExecutionProvider" in avail:
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                except Exception:
                    pass
            self._session = new_session(self._model_name, providers=providers)
            self._device_name = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
            return True
        except Exception:
            self._session = None
            self._device_name = "failed"
            return False

    @property
    def device(self) -> str:
        """Human-readable execution provider tag, set after first init."""
        return self._device_name

    # ── inference ───────────────────────────────────────────────────

    def get_mask(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Return a float32 [H, W] mask in [0..1] (1 = person). ``None`` on
        any failure — never raises.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if not self._ensure() or self._session is None:
            return None
        try:
            from rembg import remove

            h, w = frame_bgr.shape[:2]
            scale = min(1.0, float(self._proc_max) / max(1, max(h, w)))
            if scale < 1.0:
                small = cv2.resize(
                    frame_bgr,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = frame_bgr
            # rembg expects RGB uint8
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mask_small = remove(
                rgb,
                session=self._session,
                only_mask=True,
                post_process_mask=True,
            )
            if mask_small is None:
                return None
            if mask_small.ndim == 3:
                mask_small = mask_small[..., 0]
            if mask_small.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask_small.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                mask = mask_small
            return (mask.astype(np.float32) / 255.0)
        except Exception:
            return None
