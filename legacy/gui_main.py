"""
Pixel Enhancement System — GUI Entry Point.

Part of the Hybrid Image Enhancement System (CMPE 491 Senior Design Project).
Launches the PyQt6 GUI for the Pixel Enhancement Module.

Usage:
    python gui_main.py
"""

import sys

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    print("[ERROR] PyQt6 is required for the GUI.")
    print("Install it with:  pip install PyQt6")
    sys.exit(1)

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Pixel Enhancement System")
    app.setOrganizationName("OptiLumen")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
