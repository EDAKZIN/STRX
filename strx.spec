# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# Configurar librerías y datos de IA / GPU
datas = []
datas += collect_data_files('paddle')
datas += collect_data_files('paddleocr')
try:
    datas += collect_data_files('paddlex', include_py_files=True)
except Exception:
    pass

binaries = []
binaries += collect_dynamic_libs('paddle')
binaries += collect_dynamic_libs('nvidia')

hiddenimports = ['paddle', 'paddleocr', 'pyspellchecker', 'cv2', 'PyQt6']
hiddenimports += collect_submodules('paddle')

# Excluir módulos grandes e innecesarios de python base si es posible para optimizar
excludes = ['tkinter', 'matplotlib', 'IPython', 'notebook', 'scipy', 'jedi', 'pandas']

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='STRX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Consola oculta para la Build GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='STRX',
)
