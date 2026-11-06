param()
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $PSScriptRoot '.venv-experiments\Scripts\python.exe'
function Run-Checked {
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $($args -join ' ')" }
}
if (-not (Test-Path $python)) {
    Run-Checked py -3.14 -m pip install --user uv
    Run-Checked py -3.14 -m uv python install 3.13
    Run-Checked py -3.14 -m uv venv (Join-Path $PSScriptRoot '.venv-experiments') --python 3.13
    Run-Checked py -3.14 -m uv pip install --python $python torch --index-url https://download.pytorch.org/whl/cpu
}
Run-Checked py -3.14 -m uv pip install --python $python -r (Join-Path $PSScriptRoot 'requirements-controlled.txt')
Run-Checked $python -m unittest discover -s $PSScriptRoot -p test_controlled.py
Run-Checked $python (Join-Path $PSScriptRoot 'run_controlled.py')
Run-Checked $python (Join-Path $PSScriptRoot 'audit_controlled.py')
Run-Checked $python (Join-Path $PSScriptRoot 'plot_controlled.py')
