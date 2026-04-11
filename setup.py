"""
Quick-setup helper.

Run:
    python setup.py

This will:
  1. Install all Python dependencies
  2. Download the GFPGANv1.3 model weights (~340 MB)
"""

import subprocess
import sys
import os


def run(cmd):
    print(">>", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("[WARN] Command exited with code", result.returncode)


def main():
    print("=" * 60)
    print("  OptiLumen — GFPGAN Enhancement System Setup")
    print("  CMPE 491 Senior Design Project")
    print("=" * 60)
    print()

    # 1. Dependencies
    print("[1/2] Installing dependencies...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print()

    # 2. Model weights
    print("[2/2] Downloading GFPGANv1.3 model weights...")
    script = os.path.join(os.path.dirname(__file__), "scripts", "download_model.py")
    run([sys.executable, script])
    print()

    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  Launch GUI:")
    print("    python src/gui_main.py")
    print()
    print("  Batch processing:")
    print("    python src/main.py --input inputs/ --output outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
