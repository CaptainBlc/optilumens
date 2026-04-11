"""
Pixel Enhancement System — Main Entry Point.

Follows the documented sequence (Analysis Report 3.5.4):
  1. User uploads image
  2. ImageProcessor receives image
  3. analyzeImage() determines enhancement needs
  4. PixelBasedProcessor: reduceNoise(), sharpenImage()
  5. Return enhanced image
"""

import os
import cv2

from pixel_pipeline import ImageProcessor


INPUT_DIR = "../test_mix"
OUTPUT_DIR = "../outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
processor = ImageProcessor()
count = 0

for name in sorted(os.listdir(INPUT_DIR)):
    if not name.lower().endswith(extensions):
        continue
    path = os.path.join(INPUT_DIR, name)
    img = cv2.imread(path)
    if img is None:
        print("Skip (read fail):", name)
        continue

    result = processor.enhanceImage(img, mask_map=None, auto_mask=True)

    if result.success and result.enhanced is not None:
        out_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(out_path, result.enhanced)
        count += 1

        m = result.metrics
        metrics_str = ""
        if m is not None:
            metrics_str = " | PSNR={:.1f} SSIM={:.4f}".format(m.psnr, m.ssim)

        print("OK: {}{}".format(name, metrics_str))
    else:
        print("ERR: {} | {}".format(name, "; ".join(result.log[-2:])))

print()
print("Done. Processed: {} | Output: {}".format(count, OUTPUT_DIR))
