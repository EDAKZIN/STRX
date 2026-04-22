# STRX

Aplicación de escritorio en Python para extraer subtítulos con OCR desde video local, corregir tiempos/texto y exportar a formatos compatibles (como `.str` / `.srt`). Cuenta con soporte completo para aceleración por GPU (NVIDIA CUDA) y empaquetado portable.

## Requisitos

- Windows 10/11
- Python 3.11
- Drivers de NVIDIA actualizados (para soporte GPU)

## Configuración del entorno (Desarrollo)

```powershell
# Clonar el repositorio y entrar a la carpeta
git clone https://github.com/EDAKZIN/STRX.git
cd STRX

# Crear entorno virtual
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Actualizar e instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` incluye por defecto el soporte para GPU (CUDA), `paddleocr`, `paddlex` y corrección ortográfica (`pyspellchecker`).

---

## Generación de Ejecutable Portable (Distribución)

El proyecto utiliza PyInstaller para crear un ejecutable **portable ("One-Folder")** que incluye:
- Binarios de Python y dependencias.
- Las DLLs necesarias de NVIDIA (CUDA/cuDNN) inyectadas en tiempo de ejecución.
- Diccionarios locales para corrección de idiomas.

No requiere que el usuario final instale el Toolkit de CUDA, solo tener los drivers de su tarjeta de video NVIDIA.

### 1) Preparar modelos
Asegúrate de haber ejecutado la app al menos una vez en desarrollo para que los modelos de PaddleOCR se descarguen en la carpeta `runtime/models/`.

### 2) Ejecutar el Build
Usa el script de automatización para generar el paquete:

```powershell
.\scripts\build_export.ps1 -SkipZip
```

*   **Salida:** Una carpeta funcional en `dist\STRX`.
*   **Nota:** El build se mantiene como un directorio para evitar tiempos de extracción lentos debido al tamaño masivo de los binarios de Deep Learning (~3 GB). Las DLLs de CUDA están aseguradas contra corrupción por compresión (`upx=False`).

---

## Wheelhouse CUDA (Instalación Offline)

Si prefieres instalar la app en modo desarrollo en una máquina sin internet:

### 1) Descargar wheels (Máquina con internet)
```powershell
.\scripts\build_wheelhouse_cuda.ps1 -CudaChannel cu118 -GpuVersion 3.3.1 -Clean
```

### 2) Instalar offline (Máquina destino)
```powershell
.\scripts\install_wheelhouse_cuda.ps1 -CudaChannel cu118
```

---

## Arquitectura y Estructura del Proyecto

- `src/`: Código fuente.
  - `core/`: Motor OCR (`ocr_worker.py`), segmentación, corrección de texto y timecodes.
  - `ui/`: Componentes de interfaz gráfica (PyQt6).
  - `main.py`: Punto de entrada e inyección dinámica de variables de entorno para CUDA.
- `runtime/models/`: Pesos de los modelos de IA (Paddle/PaddleX).
- `runtime/logs/`: Logs de ejecución y diagnóstico CUDA (`cuda_diag.txt`).
- `scripts/`: Utilidades de build y mantenimiento de dependencias offline.
- `dist/`: (Generado) Ejecutable final portable.

## Notas importantes de Ejecución

- **Rutas Relativas:** La aplicación detecta automáticamente si está en modo "frozen" (EXE de PyInstaller) y ajusta las variables de entorno (`PATH`, `os.add_dll_directory`) para asegurar que Paddle detecte la GPU.
- **OCR y Corrección:** El OCR está optimizado con un pipeline usando `paddlex`. La corrección de texto utiliza `pyspellchecker` con soporte integrado para diccionarios locales (inglés, español, etc.).
- **Fallback a CPU:** Si la GPU NVIDIA no está disponible o las DLLs no se cargan, la app hará fallback automático a la CPU.

---

## Licencia

Este proyecto está licenciado bajo la **GNU General Public License v3.0**. Ver el archivo `LICENSE` para más detalles.
