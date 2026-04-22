# STRX MVP

Aplicación de escritorio en Python para extraer subtítulos con OCR desde video local, corregir tiempos/texto y exportar `.str` (formato estilo SRT).

## Requisitos

- Windows
- Python 3.11
- Drivers de NVIDIA actualizados (para soporte GPU)

## Configuración del entorno (Desarrollo)

```powershell
cd C:\Users\USUARIO\Documents\Pruebadeappst\STRX
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` incluye por defecto el soporte para GPU (CUDA).

---

## Generación de Ejecutable Portable (Distribución)

Este proceso crea una carpeta autónoma que **incluye CUDA y los modelos de IA**. No requiere que el usuario final instale nada más que los drivers de su tarjeta de video.

### 1) Preparar modelos y entorno
Asegúrate de haber ejecutado la app al menos una vez en desarrollo para que los modelos se descarguen en `runtime/models/`, o que ya existan ahí.

### 2) Ejecutar el Build
Usa el script de automatización para generar el paquete:

```powershell
.\scripts\build_export.ps1
```

*   **Salida:** Una carpeta en `dist\STRX` y un archivo comprimido `dist\STRX_Portable.zip`.
*   **Nota:** El build es "One-Folder" para evitar tiempos de carga lentos debido al tamaño de los binarios de CUDA (~3 GB).

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

## Verificación de Hardware

```powershell
python -c "import paddle; print('CUDA:', paddle.is_compiled_with_cuda()); print('Dispositivos:', paddle.device.cuda.device_count())"
```

## Estructura del Proyecto

- `src/`: Código fuente.
- `runtime/models/`: Pesos de los modelos de IA (Paddle/PaddleX).
- `runtime/logs/`: Logs de ejecución.
- `scripts/`: Utilidades de build y mantenimiento.
- `dist/`: (Generado) Ejecutable final portable.

## Notas importantes

- **Rutas Relativas:** La aplicación detecta automáticamente si está en modo "frozen" (EXE) y ajusta las rutas para buscar los modelos en su propia carpeta `runtime/`.
- **OCR:** Optimizado para inglés en esta versión.
- **GPU:** Si no se detecta CUDA, la app hará fallback automático a CPU.
