param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$CudaChannel = "cu118",
    [string]$RequirementsFile = "requirements.txt",
    [switch]$UpgradePip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeDir = Join-Path $projectRoot "runtime\wheels"
$baseDir = Join-Path $runtimeDir "base"
$cudaDir = Join-Path $runtimeDir ("cuda\" + $CudaChannel)
$requirementsPath = Join-Path $projectRoot $RequirementsFile

try {
    $pythonCommand = Get-Command $PythonExe -ErrorAction Stop
}
catch {
    throw "No se encontro el interprete de Python: $PythonExe"
}

if (-not (Test-Path -Path $requirementsPath)) {
    throw "No se encontro el archivo de requerimientos: $requirementsPath"
}

if (-not (Test-Path -Path $baseDir)) {
    throw "No existe el wheelhouse base: $baseDir. Ejecuta primero scripts\build_wheelhouse_cuda.ps1."
}

if (-not (Test-Path -Path $cudaDir)) {
    throw "No existe el wheelhouse CUDA para '$CudaChannel': $cudaDir. Ejecuta primero scripts\build_wheelhouse_cuda.ps1."
}

if ($UpgradePip) {
    & $pythonCommand.Source -m pip install --upgrade pip
}

Write-Host "Instalando dependencias desde wheelhouse local..."
& $pythonCommand.Source -m pip install --no-index --find-links $baseDir --find-links $cudaDir -r $requirementsPath

Write-Host "Verificando backend Paddle..."
& $pythonCommand.Source -c "import paddle;print('paddle', paddle.__version__);print('cuda_compilado', paddle.is_compiled_with_cuda());print('cuda_devices', paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0)"

Write-Host "Instalacion offline CUDA completada."
