"""
GeneralRestorer — Real-ESRGAN x2plus wrapper for non-face content.

CMPE 491 Senior Design Project.

Companion to FaceRestorer (GFPGAN). Together they cover:
    - FaceRestorer:    AI restoration on detected faces only
    - GeneralRestorer: AI super-resolution + denoise on the whole image,
                       used when there is no face, when the image is
                       small/low-quality, or when arbitrary objects are
                       present (text, plates, structures...)

This module is what makes the system **device-independent** in practice
— Real-ESRGAN was trained with synthetic degradations matching real-world
camera noise, blur, and compression artefacts, so it generalises to
phones / CCTV / drones / screenshots.

Design (mirrors face_restorer.py):
    - Lazy `import torch` + `model load` so import-time DLL faults don't
      crash the GUI.
    - Tile-based inference: large frames are split into overlapping tiles
      so 4K input doesn't exhaust memory.
    - Conservative classical fallback (bilateral + unsharp) when weights
      are missing.

Public API:
    GeneralRestorer(checkpoint_path=None, scale=2, tile=512, tile_pad=16)
    .restore(image_bgr) -> RestorationResult
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


# ── Lazy import probes ────────────────────────────────────────────────

_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    pass


# ── Result dataclass (mirrors FaceRestorer's contract) ────────────────

@dataclass
class GeneralRestorationResult:
    restored:    Optional[np.ndarray] = None
    success:     bool = False
    used_ai:     bool = False
    elapsed_s:   float = 0.0
    log:         List[str] = field(default_factory=list)


# ── Default weight location ───────────────────────────────────────────

_DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "RealESRGAN_x2plus.pth",
)


# ── Wrapper ───────────────────────────────────────────────────────────

class GeneralRestorer:
    """
    Real-ESRGAN x2plus wrapper.

    Parameters
    ----------
    checkpoint_path : Optional[str]
        Path to RealESRGAN_x2plus.pth (downloaded by scripts/download_model.py).
    scale : int
        Net SR factor of the loaded model (2 = x2plus, 4 = x4plus).
    tile : int
        Tile size for chunked inference. 0 = whole image at once. 512 is
        a good default for 8 GB VRAM / 16 GB RAM machines.
    tile_pad : int
        Overlap between tiles to suppress seams.
    output_scale : float
        How much of the SR factor to keep in the final output.
        1.0 = full SR (output 2× larger), 0.5 = back to original size.
        Default 0.5 — most callers want enhancement at original size.
    half : bool
        Use FP16 inference (CUDA only). Roughly 2× faster.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        scale: int = 2,
        tile: int = 512,
        tile_pad: int = 16,
        output_scale: float = 0.5,
        half: bool = False,
        max_process_mp: float = 4.0,
    ) -> None:
        self._ckpt = checkpoint_path or _DEFAULT_CKPT
        self.scale = int(scale)
        self.tile = max(0, int(tile))
        self.tile_pad = max(0, int(tile_pad))
        self.output_scale = float(output_scale)
        self.half = bool(half)
        # Cap how much pixel volume goes through the AI. 4 MP processes
        # in ~5-15 s on CPU; anything larger is downsampled to fit, run,
        # then upscaled back so the user still gets the AI cleanup
        # without paying full-resolution latency.
        self.max_process_mp = float(max_process_mp)

        self._model = None
        self._device = None
        self._init_tried = False

    # ── lifecycle ────────────────────────────────────────────────────

    def _init_model(self) -> bool:
        if self._model is not None:
            return True
        if self._init_tried:
            return False
        self._init_tried = True

        if not _TORCH_AVAILABLE:
            return False
        if not os.path.isfile(self._ckpt):
            return False

        try:
            import torch as _torch
            from models.rrdb_arch import realesrgan_x2plus, realesrgan_x4plus

            arch = realesrgan_x2plus() if self.scale == 2 else realesrgan_x4plus()
            sd = _torch.load(self._ckpt, map_location="cpu", weights_only=True)
            # Real-ESRGAN releases store weights under either 'params' or 'params_ema'
            if isinstance(sd, dict):
                if "params_ema" in sd:
                    sd = sd["params_ema"]
                elif "params" in sd:
                    sd = sd["params"]
            arch.load_state_dict(sd, strict=True)
            arch.eval()

            self._device = _torch.device(
                "cuda" if _torch.cuda.is_available() else "cpu")
            arch = arch.to(self._device)
            if self.half and self._device.type == "cuda":
                arch = arch.half()

            self._model = arch
            return True
        except Exception:
            return False

    # ── public API ───────────────────────────────────────────────────

    def restore(self, image: np.ndarray) -> GeneralRestorationResult:
        """
        AI-enhance the entire image. Returns same-size output by default
        (output_scale=0.5 cancels the model's ×2 upscale).

        Falls back to a conservative classical pipeline if the model is
        unavailable.
        """
        result = GeneralRestorationResult(log=[])

        if image is None or image.size == 0:
            result.log.append("[ERROR] Invalid input image")
            return result

        h0, w0 = image.shape[:2]
        result.log.append(
            "GeneralRestorer: input {}x{} ({:.2f} MP)".format(
                w0, h0, w0 * h0 / 1e6))

        t0 = time.time()

        if self._init_model():
            try:
                # ── Adaptive resolution: downscale before AI if too big ──
                # AI at half-res gives the same denoise/sharpen benefit at
                # a fraction of the time, and the model's built-in x2
                # super-resolution restores us to the original size.
                input_mp = (w0 * h0) / 1e6
                if input_mp > self.max_process_mp:
                    factor = (self.max_process_mp / input_mp) ** 0.5
                    proc_w = max(64, int(w0 * factor))
                    proc_h = max(64, int(h0 * factor))
                    proc_img = cv2.resize(image, (proc_w, proc_h),
                                          interpolation=cv2.INTER_AREA)
                    result.log.append(
                        "  Downscale {}x{} -> {}x{} ({:.2f} MP) for AI "
                        "budget".format(w0, h0, proc_w, proc_h,
                                        proc_w * proc_h / 1e6))
                else:
                    proc_img = image

                sr = self._infer(proc_img)

                # Resize SR output back to the original input size (we want
                # AI cleanup at original size, not actual super-resolution).
                if (sr.shape[1], sr.shape[0]) != (w0, h0):
                    sr = cv2.resize(sr, (w0, h0),
                                    interpolation=cv2.INTER_LANCZOS4)

                result.restored = sr
                result.success = True
                result.used_ai = True
                result.elapsed_s = time.time() - t0
                result.log.append(
                    "  AI path: Real-ESRGAN x{} -> output {}x{} ({:.2f}s)".format(
                        self.scale, w0, h0, result.elapsed_s))
                return result
            except Exception as e:
                result.log.append("  [WARN] AI path failed: {}".format(e))

        # Classical fallback
        out = self._classical_fallback(image)
        result.restored = out
        result.success = True
        result.used_ai = False
        result.elapsed_s = time.time() - t0
        result.log.append(
            "  Classical fallback (no Real-ESRGAN weights or torch)  "
            "({:.2f}s)".format(result.elapsed_s))
        return result

    # ── internals ────────────────────────────────────────────────────

    def _classical_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        Three-stage classical pipeline (bilateral denoise -> CLAHE on
        L channel -> unsharp mask) tuned to be visibly better than the
        input on degraded photos but still safe on clean ones.

        Quality Guard will catch any over-processing downstream; this
        function is intentionally a bit assertive so the user actually
        sees an improvement when the AI path is unavailable.
        """
        denoise = cv2.bilateralFilter(image, d=7, sigmaColor=40, sigmaSpace=10)

        lab = cv2.cvtColor(denoise, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        L = clahe.apply(L)
        contrasted = cv2.cvtColor(cv2.merge((L, A, B)), cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(contrasted, 1.5, blur, -0.5, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """Tile-aware inference. Returns BGR uint8, scale × spatial size."""
        import torch as _torch

        # BGR uint8 → RGB float32 [0,1] → tensor [1,3,H,W]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = _torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        x = x.to(self._device)
        if self.half and self._device.type == "cuda":
            x = x.half()

        with _torch.no_grad():
            if self.tile <= 0 or max(x.shape[-2:]) <= self.tile:
                y = self._model(x)
            else:
                y = self._tiled_inference(x)

        y = y.clamp(0, 1).float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        bgr = cv2.cvtColor((y * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return bgr

    def _tiled_inference(self, x):
        """Process input in overlapping tiles; merge with linear blend."""
        import torch as _torch

        b, c, h, w = x.shape
        s = self.scale
        out = _torch.zeros((b, c, h * s, w * s),
                           dtype=x.dtype, device=x.device)
        weight = _torch.zeros_like(out)

        t = self.tile
        tp = self.tile_pad

        for ty in range(0, h, t):
            for tx in range(0, w, t):
                y0 = max(0, ty - tp); y1 = min(h, ty + t + tp)
                x0 = max(0, tx - tp); x1 = min(w, tx + t + tp)
                tile_in = x[:, :, y0:y1, x0:x1]
                tile_out = self._model(tile_in)

                # Place back, accounting for the pad on each side
                py0 = (ty - y0) * s
                py1 = py0 + (min(h, ty + t) - ty) * s
                px0 = (tx - x0) * s
                px1 = px0 + (min(w, tx + t) - tx) * s
                oy0 = ty * s; oy1 = min(h, ty + t) * s
                ox0 = tx * s; ox1 = min(w, tx + t) * s

                out   [:, :, oy0:oy1, ox0:ox1] += tile_out[:, :, py0:py1, px0:px1]
                weight[:, :, oy0:oy1, ox0:ox1] += 1.0
        return out / weight.clamp_min(1e-6)
