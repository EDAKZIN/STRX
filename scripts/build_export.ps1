param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [switch]$SkipZip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distDir = Join-Path $projectRoot "dist"
$buildDir = Join-Path $projectRoot "build"
$outputAppDir = Join-Path $distDir "STRX"
$modelsSource = Join-Path $projectRoot "runtime\models"
$modelsTarget = Join-Path $outputAppDir "runtime\models"

try {
    $pythonCommand = Get-Command $PythonExe -ErrorAction Stop
}
catch {
    throw "No se encontro el interprete de Python: $PythonExe"
}

Write-Host "1. Instalando PyInstaller en el entorno virtual..."
& $pythonCommand.Source -m pip install pyinstaller

Write-Host "2. Limpiando directorios de build previos..."
if (Test-Path $distDir) { Remove-Item -Recurse -Force $distDir }
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }

Write-Host "3. Construyendo ejecutable con PyInstaller..."
& $pythonCommand.Source -m PyInstaller strx.spec --clean --noconfirm

if (-not (Test-Path $outputAppDir)) {
    throw "PyInstaller falló: no se creó el directorio destino $outputAppDir"
}

Write-Host "4. Copiando la carpeta de modelos de IA..."
if (Test-Path $modelsSource) {
    New-Item -ItemType Directory -Force -Path $modelsTarget | Out-Null
    Copy-Item -Path "$modelsSource\*" -Destination $modelsTarget -Recurse -Force
    Write-Host "Modelos copiados exitosamente."
} else {
    Write-Host "Advertencia: No se encontro $modelsSource. El ejecutable los descargará al primer uso." -ForegroundColor Yellow
}

if (-not $SkipZip) {
    Write-Host "5. Comprimiendo el build en STRX_Portable.zip..."
    $zipPath = Join-Path $distDir "STRX_Portable.zip"
    Compress-Archive -Path $outputAppDir -DestinationPath $zipPath -Force
    Write-Host "ZIP creado en: $zipPath"
}

Write-Host "Build completado con éxito. Ejecutable principal en: $(Join-Path $outputAppDir 'STRX.exe')"
