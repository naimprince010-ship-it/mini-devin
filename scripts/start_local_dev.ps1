# Plodder local stack: API (8000) + Vite (5173) + browser.
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
# Prefer IPv4 — on some Windows setups "localhost" resolves to ::1 first and the tab shows ERR_CONNECTION_REFUSED.
Start-Process "http://127.0.0.1:5173/app" | Out-Null
Write-Host "[start_local_dev] Opened http://127.0.0.1:5173/app" -ForegroundColor Green
Write-Host "  If the tab is blank or spins forever: wait ~10s, then ensure API is up at http://127.0.0.1:8000 (check the minimized API window)." -ForegroundColor Yellow
Write-Host "  Dashboard: http://127.0.0.1:5173/app  |  Landing: http://127.0.0.1:5173/" -ForegroundColor Gray
