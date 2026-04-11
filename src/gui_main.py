"""
GFPGAN Enhancement System — GUI Entry Point.
CMPE 491 Senior Design Project.

Usage:
    python gui_main.py
"""

import sys

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
