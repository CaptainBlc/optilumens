"""
Download AI model weights for the OptiLumen pipeline.

Models fetched into checkpoints/:
    GFPGANv1.3.pth        ~340 MB  — face restoration (FaceRestorer)
    RealESRGAN_x2plus.pth  ~64 MB  — general SR / denoise (GeneralRestorer)

Usage:
    python scripts/download_model.py            # all required models
    python scripts/download_model.py --only gfpgan
    python scripts/download_model.py --only realesrgan
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request


# ── Model registry ────────────────────────────────────────────────────

MODELS = {
    "gfpgan": {
        "url": ("https://github.com/TencentARC/GFPGAN/releases/download/"
                "v1.3.0/GFPGANv1.3.pth"),
        "filename":   "GFPGANv1.3.pth",
        "approx_mb":  340,
        "purpose":    "Face restoration (FaceRestorer)",
    },
    "realesrgan": {
        "url": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.2.1/RealESRGAN_x2plus.pth"),
        "filename":   "RealESRGAN_x2plus.pth",
        "approx_mb":  64,
        "purpose":    "General super-resolution & denoise (GeneralRestorer)",
    },
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")


# ── Download helpers ──────────────────────────────────────────────────

def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(count * block_size / total_size * 100, 100)
    bar = "#" * int(pct / 2)
    sys.stdout.write("\r  [{:<50}] {:.1f}%".format(bar, pct))
    sys.stdout.flush()


def _download(name: str, info: dict) -> bool:
    out = os.path.join(CKPT_DIR, info["filename"])

    if os.path.isfile(out):
        sz = os.path.getsize(out) / 1e6
        print("[{}] already present: {} ({:.0f} MB)".format(
            name, info["filename"], sz))
        return True

    print("[{}] downloading {} (~{} MB)".format(
        name, info["filename"], info["approx_mb"]))
    print("       {}".format(info["purpose"]))
    print("       URL: {}".format(info["url"]))

    try:
        urllib.request.urlretrieve(info["url"], out, _progress)
        actual = os.path.getsize(out) / 1e6
        print("\n       OK ({:.0f} MB)".format(actual))
        return True
    except Exception as e:
        print("\n[{}] FAILED: {}".format(name, e))
        print("       Manual download URL: {}".format(info["url"]))
        print("       Save to: {}".format(out))
        return False


# ── Entry ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download OptiLumen model weights")
    parser.add_argument(
        "--only", choices=list(MODELS.keys()),
        help="Download only this model (default: all)")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    targets = [args.only] if args.only else list(MODELS.keys())
    failed: list[str] = []

    for name in targets:
        info = MODELS[name]
        ok = _download(name, info)
        if not ok:
            failed.append(name)
        print()

    if failed:
        print("Some downloads failed: {}".format(", ".join(failed)))
        return 1

    print("All models ready.")
    print("Next: python src/gui_main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
