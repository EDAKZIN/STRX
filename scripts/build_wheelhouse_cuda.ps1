param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$CudaChannel = "cu118",
    [string]$GpuVersion = "3.3.1",
    [string]$RequirementsFile = "requirements.txt",
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeDir = Join-Path $projectRoot "runtime\wheels"
$baseDir = Join-Path $runtimeDir "base"
$cudaDir = Join-Path $runtimeDir ("cuda\" + $CudaChannel)
$requirementsPath = Join-Path $projectRoot $RequirementsFile
$tempRequirements = Join-Path $env:TEMP ("strx_requirements_base_" + [guid]::NewGuid().ToString("N") + ".txt")

try {
    $pythonCommand = Get-Command $PythonExe -ErrorAction Stop
}
catch {
    throw "No se encontro el interprete de Python: $PythonExe"
}

if (-not (Test-Path -Path $requirementsPath)) {
    throw "No se encontro el archivo de requerimientos: $requirementsPath"
}

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $baseDir, $cudaDir
}

New-Item -ItemType Directory -Force -Path $baseDir, $cudaDir | Out-Null

# Descarga paquetes base excepto paddlepaddle-gpu, que se toma del canal CUDA oficial.
Get-Content $requirementsPath |
    Where-Object { $_ -and ($_ -notmatch "^\s*#") -and ($_ -notmatch "^\s*paddlepaddle-gpu") } |
    Set-Content -Path $tempRequirements -Encoding ASCII

try {
    Write-Host "Descargando wheelhouse base en: $baseDir"
    & $pythonCommand.Source -m pip download --dest $baseDir --only-binary=:all: -r $tempRequirements

    $cudaIndex = "https://www.paddlepaddle.org.cn/packages/stable/$CudaChannel/"
    Write-Host "Descargando paddlepaddle-gpu==$GpuVersion desde: $cudaIndex"
    & $pythonCommand.Source -m pip download --dest $cudaDir --index-url $cudaIndex --extra-index-url "https://pypi.org/simple" "paddlepaddle-gpu==$GpuVersion"

    $manifestPath = Join-Path $runtimeDir "wheelhouse-manifest.txt"
    Get-ChildItem -Path $baseDir, $cudaDir -Filter "*.whl" -Recurse |
        Sort-Object FullName |
        ForEach-Object { $_.FullName.Replace($projectRoot + "\", "") } |
        Set-Content -Path $manifestPath -Encoding ASCII

    Write-Host "Wheelhouse listo."
    Write-Host "Base: $baseDir"
    Write-Host "CUDA: $cudaDir"
    Write-Host "Manifest: $manifestPath"
}
finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $tempRequirements
}
