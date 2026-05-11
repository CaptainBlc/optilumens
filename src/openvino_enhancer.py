"""
OpenVINO Image Enhancer — Intel GPU-accelerated super-resolution.

Uses Intel OpenVINO runtime to run the single-image-super-resolution
model (Intel Open Model Zoo) on Intel Integrated GPU (Iris Plus).

On MateBook 13 (Intel Iris Plus):
    - CPU inference  : ~50-100 ms per 720p frame
    - GPU inference  : ~15-30 ms per 720p frame  (~30-60 FPS)

Model: single-image-super-resolution-1032
    - Input : 1x3x480x270 (LQ frame, half-resolution)
    - Output: 1x3x1920x1080 (4x upscaled)
    - For 720p: we feed 360x180, get 720p output

Requirements:
    py -m pip install openvino openvino-dev
    omz_downloader --name single-image-super-resolution-1032
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────────────

MODEL_NAME = "single-image-super-resolution-1032"
MODEL_DIR  = Path(__file__).parent.parent / "checkpoints" / "openvino"
MODEL_XML  = MODEL_DIR / MODEL_NAME / "FP16" / f"{MODEL_NAME}.xml"

# Fallback: smaller model for 2x upscale
MODEL_NAME2 = "single-image-super-resolution-1033"
MODEL_XML2  = MODEL_DIR / MODEL_NAME2 / "FP16" / f"{MODEL_NAME2}.xml"


# ── Model download ────────────────────────────────────────────────────

def download_model(model_name: str = MODEL_NAME) -> bool:
    """Download Intel SR model via omz_downloader."""
    try:
        import subprocess
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, "-m", "omz_tools.downloader",
             "--name", model_name,
             "--output_dir", str(MODEL_DIR),
             "--precision", "FP16"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print(f"[OpenVINO] Model indirildi: {model_name}")
            return True
        # Try alternative command
        result2 = subprocess.run(
            ["omz_downloader",
             "--name", model_name,
             "--output_dir", str(MODEL_DIR),
             "--precision", "FP16"],
            capture_output=True, text=True, timeout=120
        )
        return result2.returncode == 0
    except Exception as e:
        print(f"[OpenVINO] Model indirme hatasi: {e}")
        return False


# ── OpenVINO Enhancer ─────────────────────────────────────────────────

class OpenVINOEnhancer:
    """
    Intel GPU-accelerated image enhancement via OpenVINO super-resolution.

    Usage:
        enh = OpenVINOEnhancer()
        if enh.ready:
            enhanced = enh.enhance(frame)
    """

    def __init__(self, device: str = "AUTO") -> None:
        """
        device: "AUTO" (let OpenVINO choose), "GPU" (Intel Iris), "CPU"
        AUTO prefers GPU when available, falls back to CPU automatically.
        """
        self._compiled   = None
        self._input_name = None
        self._out_name   = None
        self._in_h       = None
        self._in_w       = None
        self._device     = device
        self.ready       = False
        self._init()

    def _init(self) -> None:
        try:
            import openvino as ov
            core = ov.Core()

            # Try to load existing model
            xml = None
            for candidate in [MODEL_XML, MODEL_XML2]:
                if candidate.exists():
                    xml = candidate
                    break

            # Download if not found
            if xml is None:
                print("[OpenVINO] Model bulunamadi, indiriliyor...")
                if download_model(MODEL_NAME):
                    xml = MODEL_XML if MODEL_XML.exists() else None
                if xml is None and download_model(MODEL_NAME2):
                    xml = MODEL_XML2 if MODEL_XML2.exists() else None

            if xml is None:
                print("[OpenVINO] Model yok. Manuel indirme icin:")
                print("  py -m pip install openvino-dev")
                print(f"  omz_downloader --name {MODEL_NAME} "
                      f"--output_dir checkpoints/openvino --precision FP16")
                return

            print(f"[OpenVINO] Model yukleniyor: {xml.name} on {self._device}")
            model    = core.read_model(str(xml))
            compiled = core.compile_model(model, self._device)

            self._compiled   = compiled
            self._input_name = compiled.input(0)
            self._out_name   = compiled.output(0)

            # Determine expected input resolution
            shape = self._input_name.shape   # [1, 3, H, W]
            self._in_h = int(shape[2])
            self._in_w = int(shape[3])

            print(f"[OpenVINO] Hazir — input {self._in_w}x{self._in_h} "
                  f"on {self._device}")
            self.ready = True

        except Exception as e:
            print(f"[OpenVINO] Init hatasi: {e}")

    # ── public API ────────────────────────────────────────────────────

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a BGR uint8 frame.

        The SR model expects a specific input resolution.  We resize the
        input to the model's LQ size, run inference, then resize output
        back to the original frame dimensions.
        """
        if not self.ready or self._compiled is None:
            return frame
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]

        try:
            # Prepare input
            lq = cv2.resize(frame, (self._in_w, self._in_h),
                            interpolation=cv2.INTER_AREA)
            # BGR → RGB → float32 [0,1] → CHW → NCHW
            inp = (lq[:, :, ::-1].astype(np.float32) / 255.0)
            inp = np.transpose(inp, (2, 0, 1))[np.newaxis]

            # Inference (Intel GPU via OpenVINO)
            result = self._compiled({self._input_name: inp})
            out    = result[self._out_name]   # [1, 3, H*4, W*4]

            # NCHW → HWC → BGR → uint8
            out = np.transpose(out[0], (1, 2, 0))
            out = np.clip(out * 255, 0, 255).astype(np.uint8)
            out = out[:, :, ::-1]   # RGB → BGR

            # Resize to original frame size
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
            return out

        except Exception as e:
            print(f"[OpenVINO] Inference hatasi: {e}")
            return frame

    def available_devices(self) -> list:
        """Return list of available OpenVINO devices."""
        try:
            import openvino as ov
            return ov.Core().available_devices
        except Exception:
            return []


# ── Singleton (lazy init) ─────────────────────────────────────────────

_instance: Optional[OpenVINOEnhancer] = None


def get_enhancer(device: str = "AUTO") -> OpenVINOEnhancer:
    """Return the shared OpenVINOEnhancer instance."""
    global _instance
    if _instance is None:
        _instance = OpenVINOEnhancer(device=device)
    return _instance


# ── Standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    enh = OpenVINOEnhancer()
    print("Devices:", enh.available_devices())

    if not enh.ready:
        print("Model hazir degil. Cikiliyor.")
        sys.exit(1)

    # Benchmark with a test frame
    test = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    # Warm-up
    enh.enhance(test)
    # Measure
    t0 = time.perf_counter()
    for _ in range(10):
        enh.enhance(test)
    dt = (time.perf_counter() - t0) / 10 * 1000
    print(f"Ortalama inference: {dt:.1f} ms  ({1000/dt:.1f} FPS)")
