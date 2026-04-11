"""
Download GFPGANv1.3 model weights.

Usage:
    python scripts/download_model.py

Downloads GFPGANv1.3.pth (~340 MB) into the checkpoints/ folder.
"""

import os
import sys
import urllib.request

URL = ("https://github.com/TencentARC/GFPGAN/releases/download/"
       "v1.3.0/GFPGANv1.3.pth")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "checkpoints", "GFPGANv1.3.pth")


def _progress(count, block_size, total_size):
    pct = min(count * block_size / total_size * 100, 100)
    bar = "#" * int(pct / 2)
    sys.stdout.write("\r  [{:<50}] {:.1f}%".format(bar, pct))
    sys.stdout.flush()


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    if os.path.isfile(OUT):
        sz = os.path.getsize(OUT) / 1e6
        print("Already downloaded: {} ({:.0f} MB)".format(OUT, sz))
        return

    print("Downloading GFPGANv1.3.pth...")
    print("URL : {}".format(URL))
    print("Dest: {}".format(OUT))
    print()

    try:
        urllib.request.urlretrieve(URL, OUT, _progress)
        print("\n\nDone! ({:.0f} MB)".format(os.path.getsize(OUT) / 1e6))
        print("Now run:  python src/gui_main.py")
    except Exception as e:
        print("\n[ERROR] Download failed: {}".format(e))
        print("Manual download:")
        print("  {}".format(URL))
        print("  Save to: {}".format(OUT))
        sys.exit(1)


if __name__ == "__main__":
    main()
