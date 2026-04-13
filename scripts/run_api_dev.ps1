# Start Plodder API with .env applied (avoids empty shell vars blocking OPENAI_API_KEY).
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$envFile = Join-Path $Root ".env"
& $py -m uvicorn mini_devin.api:app --host 127.0.0.1 --port 8000 --reload --env-file $envFile
