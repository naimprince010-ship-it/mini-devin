#Requires -Version 5.1
<#
.SYNOPSIS
  Help set up Ollama on Windows for Plodder (pull default registry model).

.DESCRIPTION
  1) Checks for ``ollama`` on PATH (install from https://ollama.com or: winget install Ollama.Ollama).
  2) Runs ``ollama pull`` for models used in mini_devin registry (default: llama3.2).

  Then copy the Ollama-only block from .env.example into your repo-root ``.env`` and start the API.
#>
$ErrorActionPreference = "Stop"
$models = @("llama3.2")
if ($args.Count -gt 0) { $models = $args }

$exe = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $exe) {
    Write-Host "ollama not found on PATH. Install: https://ollama.com/download/windows" -ForegroundColor Yellow
    Write-Host "Or: winget install -e --id Ollama.Ollama" -ForegroundColor Yellow
    exit 1
}

foreach ($m in $models) {
    Write-Host ">>> ollama pull $m" -ForegroundColor Cyan
    & ollama pull $m
}

Write-Host ""
Write-Host "Next: merge the Ollama-only env block from .env.example into your .env (GROQ_ENABLED=false, OLLAMA_ENABLED=true, LLM_MODEL=ollama/llama3.2), then start Plodder." -ForegroundColor Green
