import os
import sys
import ctypes
from pathlib import Path

# Registrar AUMID ANTES de cualquier import de Qt para que Windows Shell
# asocie la ventana a este proceso y no a python.exe desde el inicio
if sys.platform == "win32":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("strx.app.1")

# --- CONFIGURACIÓN PARA EL EJECUTABLE (CUDA & DLLs) ---
if getattr(sys, "frozen", False):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # sys._MEIPASS apunta directamente a _internal/ en PyInstaller 6
    meipass = Path(sys._MEIPASS)

    # Registramos todas las carpetas con DLLs (rutas verificadas en dist/)
    dll_dirs = set()

    # 1. Raíz de _internal (donde el .spec copia DLLs manualmente)
    dll_dirs.add(meipass)
    # 2. paddle/libs (DLLs propias de paddle + CUDA)
    dll_dirs.add(meipass / "paddle" / "libs")
    # 3. Subcarpetas de nvidia/* (donde collect_all las coloca)
    nvidia_root = meipass / "nvidia"
    if nvidia_root.exists():
        for sub in nvidia_root.iterdir():
            bin_dir = sub / "bin"
            if bin_dir.exists():
                dll_dirs.add(bin_dir)

    path_entries = []
    for d in dll_dirs:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
                path_entries.append(str(d))
            except Exception:
                pass

    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries) + os.pathsep + os.environ.get("PATH", "")
# -------------------------------------------------------

# Imports de Qt después del AUMID
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from config import ensure_runtime_dirs
from ui.main_window import MainWindow

if getattr(sys, "frozen", False):
    _ASSETS_DIR = Path(sys._MEIPASS) / "assets"
else:
    _ASSETS_DIR = Path(__file__).parent / "assets"


def main() -> int:
    ensure_runtime_dirs()

    app = QApplication(sys.argv)
    # Sincronizar identidad Qt con el manifiesto PE (version.txt)
    app.setApplicationName("STRX")
    app.setOrganizationName("EDAKZIN")
    # Usar .ico para la taskbar de Windows (multi-resolución)
    icon = QIcon(str(_ASSETS_DIR / "icon.ico"))
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowIcon(icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
