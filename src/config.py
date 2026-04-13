from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
MODELS_DIR = RUNTIME_DIR / "models"
WHEELS_DIR = RUNTIME_DIR / "wheels"
LOGS_DIR = RUNTIME_DIR / "logs"
TEMP_DIR = RUNTIME_DIR / "temp"


def ensure_runtime_dirs() -> None:
    for directory in (MODELS_DIR, WHEELS_DIR, LOGS_DIR, TEMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_model_environment() -> dict[str, Path]:
    ensure_runtime_dirs()

    paddle_home = MODELS_DIR / "paddle"
    paddlex_cache_home = MODELS_DIR / "paddlex"
    cache_dir = TEMP_DIR / "cache"

    for directory in (paddle_home, paddlex_cache_home, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Se fuerza el uso de caché local del proyecto para evitar rutas globales.
    os.environ["PADDLE_HOME"] = str(paddle_home)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddlex_cache_home)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    return {
        "paddle_home": paddle_home,
        "paddlex_cache_home": paddlex_cache_home,
    }
