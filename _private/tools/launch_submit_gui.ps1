# Launch Assignment 3 submit GUI (local only)
# Usage: powershell -ExecutionPolicy Bypass -File _private/tools/launch_submit_gui.ps1

$ErrorActionPreference = "Stop"
$TOOLS = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Split-Path -Parent (Split-Path -Parent $TOOLS)
Set-Location $ROOT

$py = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Virtual env not found. Run: python -m venv .venv && pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

& $py (Join-Path $TOOLS "tml_submit_gui.py")
