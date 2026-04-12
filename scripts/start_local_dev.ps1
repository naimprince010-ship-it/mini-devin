# Mini-Devin local stack: API (8000) + Vite (5173) + browser.
# Requires: Poetry deps installed, frontend `npm install` once, `.env` with OPENAI_API_KEY.
$ErrorActionPreference = "Stop"
# scripts/ -> mini-devin repo root
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

Write-Host "[start_local_dev] Root: $Root"

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "Poetry not found on PATH. Install Poetry first." -ForegroundColor Red
    exit 1
}

$fe = Join-Path $Root "frontend"
if (-not (Test-Path (Join-Path $fe "node_modules"))) {
    Write-Host "[start_local_dev] Running npm install in frontend..." -ForegroundColor Yellow
    Set-Location $fe
    npm install
    Set-Location $Root
}

$apiCmd = "Set-Location '$Root'; poetry run uvicorn mini_devin.api.app:app --host 127.0.0.1 --port 8000 --reload"
Start-Process powershell -WindowStyle Minimized -ArgumentList @("-NoExit", "-Command", $apiCmd)

$uiCmd = "Set-Location '$fe'; npm run dev"
Start-Process powershell -WindowStyle Minimized -ArgumentList @("-NoExit", "-Command", $uiCmd)

Start-Sleep -Seconds 6
Start-Process "http://localhost:5173/" | Out-Null
Write-Host "[start_local_dev] Opened http://localhost:5173/ — API should be http://127.0.0.1:8000" -ForegroundColor Green
