$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Use the experiment venv if present; otherwise fall back to `python` on PATH.
$python = Join-Path $PSScriptRoot 'ijsm_review\.venv-experiments\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

$grid = Join-Path $PSScriptRoot 'code\forwarding-LSTM\grid'

function Run-Experiment([string]$script, [string]$label) {
    Write-Host "Running $label experiment..."
    & $python (Join-Path $grid $script)
    if ($LASTEXITCODE -ne 0) { throw "$script failed with exit code $LASTEXITCODE" }
}

Run-Experiment 'gru.py' 'GRU'
Run-Experiment 'lstm.py' 'LSTM'
Run-Experiment 'sarima.py' 'SARIMA'
Run-Experiment 'recursive_lstm_grid.py' 'Recursive LSTM'
Run-Experiment 'transformer.py' 'Transformer'

Write-Host 'All experiments finished successfully.'
