$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create the Python environment using the README setup steps first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'artifacts\pipeline.joblib'))) {
    throw 'Download the data and train the model using the README setup steps first.'
}
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
Write-Host 'ReviewSense: http://127.0.0.1:8501'
& $taskPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
