$condaRoot = "I:\temp\miniconda3"
$condaHook = Join-Path $condaRoot "shell\condabin\conda-hook.ps1"
$envPath = Join-Path $condaRoot "envs\lehome"

if (-not (Test-Path $condaHook)) {
    throw "Conda hook not found at $condaHook"
}

. $condaHook
conda activate $envPath
$env:OMNI_KIT_ACCEPT_EULA = "YES"

Write-Host "Activated Conda environment: lehome"
Write-Host "Isaac Sim EULA accepted for this shell session."
