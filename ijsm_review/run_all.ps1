param(
    [switch]$SetupOnly
)

$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $PSScriptRoot '.venv-experiments\Scripts\python.exe'

function Run-Checked {
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE : $($args -join ' ')" }
}

Run-Checked py -3.14 -m pip install --user uv
if (-not (Test-Path $python)) {
    Run-Checked py -3.14 -m uv python install 3.13
    Run-Checked py -3.14 -m uv venv (Join-Path $PSScriptRoot '.venv-experiments') --python 3.13
}

Run-Checked py -3.14 -m uv pip install --python $python torch --index-url https://download.pytorch.org/whl/cpu
Run-Checked py -3.14 -m uv pip install --python $python -r (Join-Path $PSScriptRoot 'requirements-benchmarks.txt') tensorflow scikit-learn
Run-Checked $python (Join-Path $PSScriptRoot 'setup_benchmarks.py')
Run-Checked $python -m unittest discover -s $PSScriptRoot

if ($SetupOnly) { return }

$datasets = @('geant', 'sbrc')
$reviewBbr = Get-ChildItem (Join-Path $PSScriptRoot 'data') -File -Filter '*.csv'
$sbrcBbr = Join-Path $root 'sbrc_final_results\choosen_treated_datasets_6h'
if ($reviewBbr.Count -ne (Get-ChildItem $sbrcBbr -File -Filter '*.csv').Count) {
    $datasets += 'bbr'
} else {
    foreach ($file in $reviewBbr) {
        $peer = Join-Path $sbrcBbr $file.Name
        if (-not (Test-Path $peer) -or (Get-FileHash $file.FullName).Hash -ne (Get-FileHash $peer).Hash) {
            $datasets += 'bbr'
            break
        }
    }
}

foreach ($dataset in $datasets) {
    $inputDirectory = if ($dataset -eq 'geant') { Join-Path $PSScriptRoot 'data\geant' } elseif ($dataset -eq 'sbrc') { $sbrcBbr } else { Join-Path $PSScriptRoot 'data' }
    foreach ($source in (Get-ChildItem $inputDirectory -File -Filter '*.csv' | Sort-Object Name)) {
        Run-Checked $python (Join-Path $PSScriptRoot 'forward_prediction.py') --dataset $dataset --source $source.Name --skip-existing
    }
    Run-Checked $python (Join-Path $PSScriptRoot 'run_benchmarks.py') --dataset $dataset --model all
}
