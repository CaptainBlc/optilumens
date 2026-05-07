"""
Quick smoke test for OptiLumen pipeline.

- Runs the pipeline on a couple of synthetic images (low-light+blur, clean gradient)
- Saves outputs under ./outputs/
- Prints a short log tail focusing on layer decisions + QualityGuard outcomes.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _make_lowlight_blur() -> np.ndarray:
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = 18
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    noise = (np.random.randn(*img.shape) * 6).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _make_clean_grad() -> np.ndarray:
    x = np.linspace(0, 255, 320, dtype=np.uint8)
    img = np.tile(x, (240, 1))
    return cv2.merge([img, img, img])


def main() -> int:
    _ensure_src_on_path()

    from pipeline import ImageEnhancementPipeline  # noqa: PLC0415
    from scene_presets import PRESETS  # noqa: PLC0415

    os.makedirs("outputs", exist_ok=True)
    pipe = ImageEnhancementPipeline()

    def run_case(name: str, img: np.ndarray, preset_name: str) -> None:
        preset = PRESETS.get(preset_name)
        if preset is None:
            raise RuntimeError(f"Missing preset: {preset_name}")

        res = pipe.restoreImage(img, preset=preset)
        out_path = os.path.join("outputs", f"smoke_{name}_{preset.name}.png")
        cv2.imwrite(out_path, res.restored)

        focus = ("GUARD/", "Decision", "GlobalEnhancer", "GlobalEnhance", "Real-ESRGAN", "GFPGAN")
        tail = [ln for ln in res.log if any(k in ln for k in focus)][-35:]

        print(f"{name} {preset.name} -> {out_path}")
        print("---log_tail---")
        print("\n".join(tail))
        print("---")

    img1 = _make_lowlight_blur()
    img2 = _make_clean_grad()

    for pn in ("natural_plus", "modern_touch"):
        run_case("lowlight_blur", img1, pn)
        run_case("clean_grad", img2, pn)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

