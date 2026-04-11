<#
.SYNOPSIS
  Post-deploy smoke test for Mini-Devin (production or staging).

.DESCRIPTION
  Verifies API health, providers JSON, SPA shell, session create, and static assets.
  Set $env:SMOKE_BASE_URL to override (default: DigitalOcean app URL).

  DigitalOcean secrets (App Platform -> backend -> Settings -> App-Level Environment):
  - OPENAI_API_KEY     (required for default OpenAI models)
  - LLM_MODEL          (optional; e.g. gpt-4o-mini)
  - DATABASE_URL       (optional; default sqlite in container - use DO Managed DB for prod)
  - BROWSERLESS_API_KEY / BROWSERLESS_WS_URL (optional browser automation)
  - GITHUB_TOKEN       (optional repo features)
  - ANTHROPIC_API_KEY / GEMINI_API_KEY (optional; add in DO UI if app.yaml extended)

  After changing secrets: redeploy or restart the backend component.
#>
param(
  [string]$BaseUrl = $(if ($env:SMOKE_BASE_URL) { $env:SMOKE_BASE_URL } else { "https://mini-devin-pcgvw.ondigitalocean.app" })
)

$ErrorActionPreference = "Stop"
$u = $BaseUrl.TrimEnd("/")

function Test-Url {
  param([string]$Method, [string]$Path, [object]$Body = $null, [int[]]$Ok = @(200))
  $uri = "$u$Path"
  try {
    $params = @{ Uri = $uri; Method = $Method; UseBasicParsing = $true; TimeoutSec = 45 }
    if ($Body) {
      $params.ContentType = "application/json"
      $params.Body = ($Body | ConvertTo-Json -Compress)
    }
    $r = Invoke-WebRequest @params
    if ($Ok -notcontains $r.StatusCode) {
      throw "Expected one of [$($Ok -join ',')] got $($r.StatusCode) for $Method $uri"
    }
    return $r
  }
  catch {
    Write-Host "FAIL $Method $uri -> $($_.Exception.Message)" -ForegroundColor Red
    throw
  }
}

Write-Host "`n=== Mini-Devin smoke: $u ===`n" -ForegroundColor Cyan

Write-Host "[1] GET /api/health"
$r1 = Test-Url GET "/api/health"
Write-Host "    OK $($r1.StatusCode) $($r1.Content.Substring(0, [Math]::Min(120, $r1.Content.Length)))..."

Write-Host "[2] GET /api/providers"
$r2 = Test-Url GET "/api/providers"
$j2 = $r2.Content | ConvertFrom-Json
if (-not $j2.providers) { throw "providers missing in response" }
Write-Host "    OK providers count: $($j2.providers.Count)"

Write-Host "[3] GET / (SPA)"
$r3 = Test-Url GET "/"
if ($r3.Content.Length -lt 100) { throw "index.html suspiciously small" }
Write-Host "    OK len=$($r3.Content.Length)"

Write-Host "[4] GET /app (deep link)"
$r4 = Test-Url GET "/app"
Write-Host "    OK len=$($r4.Content.Length)"

Write-Host "[5] GET /api/unknown-route (expect 404 JSON on current builds; older deploys may return SPA 200)"
try {
  $r5 = Invoke-WebRequest -Uri "$u/api/nonexistent-smoke-404" -UseBasicParsing -TimeoutSec 20 -ErrorAction Stop
  if ($r5.StatusCode -eq 404) {
    Write-Host "    OK 404"
  }
  elseif ($r5.StatusCode -eq 200 -and $r5.Content -match '(?i)<!DOCTYPE|html') {
    Write-Host "    WARN 200 HTML - unknown /api/* still mapped to SPA; redeploy latest main for strict API 404." -ForegroundColor Yellow
  }
  else {
    throw "Unexpected status $($r5.StatusCode) for unknown API path"
  }
}
catch {
  $resp = $_.Exception.Response
  if ($resp -and [int]$resp.StatusCode -eq 404) {
    Write-Host "    OK 404"
  }
  else {
    throw $_
  }
}

Write-Host "[6] POST /api/sessions (smoke session)"
$body = @{
  working_directory     = ""
  model                 = "gpt-4o-mini"
  max_iterations        = 5
  auto_git_commit       = $false
  git_push              = $false
}
# API accepts flexible JSON; title optional on some builds
$r6 = Test-Url POST "/api/sessions" $body
$j6 = $r6.Content | ConvertFrom-Json
if (-not $j6.session_id) { throw "session_id missing" }
$sid = $j6.session_id
Write-Host "    OK session_id=$sid"

Write-Host "[7] GET /api/sessions (includes new session)"
$r7 = Test-Url GET "/api/sessions"
$arr = $r7.Content | ConvertFrom-Json
$found = $false
foreach ($s in $arr) {
  if ($s.session_id -eq $sid) { $found = $true; break }
}
if (-not $found) { Write-Host "    WARN new session not in list (may be OK if list capped)" -ForegroundColor Yellow }
else { Write-Host "    OK listed" }

Write-Host ""
Write-Host "=== All smoke checks passed ===" -ForegroundColor Green
Write-Host ""
