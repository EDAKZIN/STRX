# -*- mode: python ; coding: utf-8 -*-
# Basado en el script oficial de PaddleOCR:
# https://www.paddleocr.ai/main/version3.x/deployment/packaging.html
import os
import subprocess
import sys
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files, copy_metadata

# Ruta absoluta al directorio donde está este .spec (independiente del CWD)
_SPEC_ROOT = os.path.dirname(os.path.abspath(SPEC))
_ICON_PATH = os.path.join(_SPEC_ROOT, 'src', 'assets', 'icon.ico')

# 1. METADATA CRÍTICA - PaddleX verifica en runtime qué paquetes están instalados
#    usando importlib.metadata. Sin esto, falla con "dependency error".
#    Lista obtenida con: paddlex.utils.deps.BASE_DEP_SPECS intersectado con el entorno.
deps_with_metadata = [
    'aistudio-sdk', 'chardet', 'colorlog', 'filelock', 'imagesize',
    'Jinja2', 'modelscope', 'numpy', 'opencv-contrib-python', 'packaging',
    'pandas', 'pillow', 'prettytable', 'pyclipper', 'pydantic', 'pypdfium2',
    'python-bidi', 'PyYAML', 'py-cpuinfo', 'requests', 'ruamel.yaml',
    'safetensors', 'scikit-image', 'scipy', 'shapely', 'tqdm', 'ujson',
    # Paquetes propios del proyecto también necesitan su metadata
    'paddleocr', 'paddlepaddle-gpu', 'paddlex', 'paddlepaddle',
]

datas = [('src/assets', 'assets')]
binaries = []
hiddenimports = ['cv2', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'pyspellchecker', 'winreg']

# Copiar metadata de todos los paquetes necesarios
for dep in deps_with_metadata:
    try:
        datas += copy_metadata(dep)
    except Exception:
        pass  # El paquete no está instalado, se ignora

# 2. RECOLECCIÓN DE DATOS Y MÓDULOS (metodo oficial: collect-data paddlex)
for pkg in ['paddlex', 'paddleocr']:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Diccionarios de pyspellchecker (spellchecker/resources/*.json.gz)
datas += collect_data_files('spellchecker')

# 3. RECOLECCIÓN DE BINARIOS DE PADDLE (método oficial: collect-binaries paddle)
binaries += collect_dynamic_libs('paddle')

# 4. CUDA: Recolección de binarios de nvidia (método oficial: --nvidia flag)
binaries += collect_dynamic_libs('nvidia')

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook', 'jedi'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='STRX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # SIEMPRE FALSE: UPX corrompe DLLs de CUDA
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_ICON_PATH,
    version='version.txt',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='STRX',
)
