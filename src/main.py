"""
GFPGAN Enhancement System — Batch Processing Entry Point.
Processes all images in inputs/ and saves to outputs/.

Usage:
    python src/main.py [--fidelity 0.5] [--upscale 1] [--center]
"""

import argparse
import os
import cv2

from pipeline import ImageEnhancementPipeline


def main():
    ap = argparse.ArgumentParser(description="GFPGAN batch face restoration")
    ap.add_argument("--input",    default="../inputs",  help="Input directory")
    ap.add_argument("--output",   default="../outputs", help="Output directory")
    ap.add_argument("--fidelity", type=float, default=0.5,
                    help="0=max AI restoration, 1=original (default 0.5)")
    ap.add_argument("--upscale",  type=int, default=1,
                    help="Output upscale factor (default 1)")
    ap.add_argument("--center",   action="store_true",
                    help="Only process center/largest face")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    files = [f for f in sorted(os.listdir(args.input))
             if f.lower().endswith(exts)]

    if not files:
        print("[WARNING] No images found in {}".format(args.input))
        return

    pipe = ImageEnhancementPipeline(upscale=args.upscale,
                                     fidelity_weight=args.fidelity)
    print("GFPGAN Enhancement System  —  {} images".format(len(files)))
    print("  fidelity = {}  upscale = {}x  center_only = {}".format(
        args.fidelity, args.upscale, args.center))
    print()

    ok = 0
    for name in files:
        path = os.path.join(args.input, name)
        img = cv2.imread(path)
        if img is None:
            print("[SKIP] {}".format(name))
            continue

        result = pipe.restoreImage(img, fidelity_weight=args.fidelity,
                                    only_center_face=args.center)
        if result.success and result.restored is not None:
            out = os.path.join(args.output, name)
            cv2.imwrite(out, result.restored)
            m = result.metrics
            r = result.restoration
            faces = r.faces_found if r else "?"
            psnr  = "{:.1f}dB".format(m.psnr) if m else "--"
            ssim  = "{:.4f}".format(m.ssim)   if m else "--"
            print("[OK] {}  faces={}  PSNR={}  SSIM={}".format(
                name, faces, psnr, ssim))
            ok += 1
        else:
            print("[ERR] {}".format(name))

    print()
    print("Done.  {}/{} processed  ->  {}".format(ok, len(files), args.output))


if __name__ == "__main__":
    main()
