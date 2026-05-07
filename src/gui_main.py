"""
GFPGAN Enhancement System — GUI Entry Point.
CMPE 491 Senior Design Project.

Usage:
    python gui_main.py
"""

import sys

# IMPORTANT: prime PyTorch *before* PyQt6 imports.
# On Windows, when PyQt6 loads first it pulls Qt's bundled C runtime
# DLLs into the process; torch's c10.dll then refuses to initialise
# (WinError 1114). Importing torch first reserves its DLL space and
# avoids the conflict so AI layers (GFPGAN, FaceParser, Real-ESRGAN)
# can actually run inside the GUI process.
try:
    import torch  # noqa: F401
except (ImportError, OSError) as _e:
    print("[WARN] torch could not be imported up front:", _e)
    print("       AI layers will fall back to classical processing.")

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    print("[ERROR] PyQt6 required.  pip install PyQt6")
    sys.exit(1)

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("GFPGAN Enhancement System")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
