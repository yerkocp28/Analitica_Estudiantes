$ErrorActionPreference = 'Stop'
$pilotRepository = Split-Path -Parent $PSScriptRoot
$pilotPython = Join-Path $pilotRepository '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pilotPython)) {
    throw 'Falta el entorno .venv del proyecto. Consulta piloto_fan/README.md.'
}
& $pilotPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.port 8502
