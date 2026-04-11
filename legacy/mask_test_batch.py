"""
Batch test for the full ImageProcessor pipeline.

Usage:
    python mask_test_batch.py

Outputs per image:
  1. Visual: [Original | Mask | Enhanced | Difference Heatmap] + metrics bar
  2. Log:    ../outputs/pixel_pipeline_log.txt (full decision engine trace)

Follows the documented sequence (Analysis Report 3.5.4):
  analyzeImage() -> [LabelBasedProcessor] -> PixelBasedProcessor -> [GeneralEnhancer]
"""

import os
import sys

import cv2
import numpy as np

from pixel_pipeline import ImageProcessor


INPUT_DIR = "../test_mix"
OUTPUT_DIR = "../outputs/mask_test_results"
LOG_PATH = "../outputs/pixel_pipeline_log.txt"


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a text label to the top-left of an image."""
    out = img.copy()
    h = out.shape[0]
    font_scale = max(0.4, h / 1200.0)
    thickness = max(1, int(h / 600))

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(out, (0, 0), (tw + 10, th + 14), (0, 0, 0), -1)
    cv2.putText(out, text, (5, th + 8), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        print("[ERROR] Input directory not found: {}".format(INPUT_DIR))
        sys.exit(1)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(INPUT_DIR, f))
    ]

    if not files:
        print("[WARNING] No images found in {}".format(INPUT_DIR))
        sys.exit(0)

    processor = ImageProcessor()
    log_lines = []
    total = 0

    log_lines.append("=" * 70)
    log_lines.append("  PIXEL ENHANCEMENT SYSTEM - BATCH PROCESSING LOG")
    log_lines.append("  Hybrid Architecture: AI + Classical (HLD 3.1)")
    log_lines.append("  {} images to process".format(len(files)))
    log_lines.append("=" * 70)
    log_lines.append("")

    print("=" * 60)
    print("  PIXEL ENHANCEMENT SYSTEM - BATCH TEST")
    print("  {} images found in {}".format(len(files), INPUT_DIR))
    print("=" * 60)
    print()

    for name in sorted(files):
        path = os.path.join(INPUT_DIR, name)
        img = cv2.imread(path)
        if img is None:
            msg = "[SKIP] Read failed: {}".format(name)
            print(msg)
            log_lines.append(msg)
            continue

        print("--- {} ---".format(name))

        result = processor.enhanceImage(img, mask_map=None, auto_mask=True)

        # Print the full log to console so user can see decisions
        for line in result.log:
            print("  " + line)

        if not result.success or result.enhanced is None:
            msg = "[ERROR] enhanceImage failed for {}".format(name)
            print(msg)
            log_lines.extend([msg] + result.log + [""])
            print()
            continue

        h, w = img.shape[:2]

        mask = result.mask
        if mask is None:
            mask = np.ones((h, w), dtype=np.float32)
        mask_vis = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        enhanced = result.enhanced
        diff_map = result.difference_map

        panels = [img, mask_vis, enhanced, diff_map]
        for i, p in enumerate(panels):
            ph, pw = p.shape[:2]
            if (ph, pw) != (h, w):
                panels[i] = cv2.resize(p, (w, h))

        panels[0] = _add_label(panels[0], "Original")
        panels[1] = _add_label(panels[1], "Mask")
        panels[2] = _add_label(panels[2], "Enhanced")
        panels[3] = _add_label(panels[3], "Difference")

        concat = cv2.hconcat(panels)

        if result.metrics is not None:
            m = result.metrics
            bar_h = max(30, int(h * 0.04))
            bar = np.zeros((bar_h, concat.shape[1], 3), dtype=np.uint8)
            metrics_text = (
                "PSNR={:.1f}dB  SSIM={:.4f}  "
                "Entropy={:.2f}->{:.2f}  "
                "Colorfulness={:.1f}->{:.1f}".format(
                    m.psnr, m.ssim,
                    m.entropy_original, m.entropy_enhanced,
                    m.colorfulness_original, m.colorfulness_enhanced,
                )
            )
            font_scale = max(0.35, bar_h / 60.0)
            cv2.putText(bar, metrics_text, (10, bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (200, 200, 200), 1, cv2.LINE_AA)
            concat = cv2.vconcat([concat, bar])

        out_name = os.path.splitext(name)[0] + "_result.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, concat)
        total += 1

        print("  => Saved: {}".format(out_name))
        print()

        log_lines.append("=" * 60)
        log_lines.append("  {}".format(name))
        log_lines.append("=" * 60)
        log_lines.extend(result.log)
        log_lines.append("")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("=" * 60)
    print("  SUMMARY")
    print("  {} / {} images processed successfully".format(total, len(files)))
    print("  Visual results : {}".format(os.path.abspath(OUTPUT_DIR)))
    print("  Detailed log   : {}".format(os.path.abspath(LOG_PATH)))
    print("=" * 60)


if __name__ == "__main__":
    main()
