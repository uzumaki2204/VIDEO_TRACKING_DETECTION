param(
    [string]$VenvName = ".venv"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot $VenvName
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $repoRoot "requirements.txt"
$verifyScript = Join-Path $PSScriptRoot "verify_torch_cuda.py"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found on PATH."
}

if (-not (Test-Path -LiteralPath $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    python -m venv $venvPath
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found inside $venvPath"
}

Write-Host "Upgrading pip"
& $pythonExe -m pip install --upgrade pip

Write-Host "Installing CUDA-enabled PyTorch wheels"
& $pythonExe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.10.0 torchvision==0.25.0

Write-Host "Installing project dependencies"
& $pythonExe -m pip install -r $requirementsPath

Write-Host "Verifying CUDA runtime"
& $pythonExe $verifyScript

Write-Host "Environment ready. Activate with: $VenvName\\Scripts\\Activate.ps1"
