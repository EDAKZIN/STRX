# STRX MVP

Aplicación de escritorio en Python para extraer subtítulos con OCR desde video local, corregir tiempos/texto y exportar `.str` (formato estilo SRT).

## Requisitos

- Windows
- Python 3.11

## Configuración del entorno virtual (PowerShell)

```powershell
cd C:\Users\USUARIO\Documents\Pruebadeappst\STRX
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` usa PaddleOCR como backend OCR principal.

## Wheelhouse CUDA portable (recomendado para empaquetado)

Este flujo deja los wheels en `runtime/wheels/` para poder instalar en otra máquina sin internet.

### 1) Descargar wheels (máquina con internet)

```powershell
cd C:\Users\USUARIO\Documents\Pruebadeappst\STRX
.venv\Scripts\Activate.ps1
.\scripts\build_wheelhouse_cuda.ps1 -CudaChannel cu118 -GpuVersion 3.3.1 -Clean
```

Genera:
- `runtime/wheels/base/` (dependencias generales).
- `runtime/wheels/cuda/cu118/` (`paddlepaddle-gpu` + dependencias CUDA `nvidia-*`).
- `runtime/wheels/wheelhouse-manifest.txt` (inventario de wheels).

### 2) Instalar offline desde wheelhouse (máquina destino)

```powershell
cd C:\Users\USUARIO\Documents\Pruebadeappst\STRX
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
.\scripts\install_wheelhouse_cuda.ps1 -CudaChannel cu118
```

Opcional: agregar `-UpgradePip`.

Notas:
- El wheelhouse CUDA puede superar ~2.5 GB.
- El canal (`cu118`, `cu121`, etc.) debe coincidir con la versión CUDA soportada por la GPU/driver objetivo.

## Verificar GPU en el entorno

```powershell
python -c "import paddle; print(paddle.__version__); print(paddle.is_compiled_with_cuda()); print(paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0)"
```

## Ejecución

```powershell
cd C:\Users\USUARIO\Documents\Pruebadeappst\STRX
.venv\Scripts\Activate.ps1
python src\main.py
```

## Pruebas unitarias

```powershell
cd C:\Users\USUARIO\Documents\Pruebadeappst\STRX
.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

## Estructura principal

- `src/`: código fuente de la aplicación.
- `runtime/models/`: caché local de modelos descargados (PaddleOCR/Paddle).
- `runtime/logs/`: logs de ejecución (carpeta generada automáticamente).
- `runtime/temp/`: temporales.
- `tests/`: pruebas unitarias del núcleo.

## Notas importantes

- La aplicación fuerza el almacenamiento de modelos en `runtime/models/`.
- Se permite solape de segmentos para escenarios con múltiples textos simultáneos.
- El exportador valida tiempo de inicio/fin y texto antes de generar `.str`.
- OCR fijo en inglés para esta versión.
- Puedes activar/desactivar `Corregir texto OCR` para aplicar corrección por diccionario.
