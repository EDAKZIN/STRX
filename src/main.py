from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from config import ensure_runtime_dirs
from ui.main_window import MainWindow


def main() -> int:
    ensure_runtime_dirs()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

