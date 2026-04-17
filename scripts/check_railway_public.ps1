<#
.SYNOPSIS
  Probe the public Railway URL (root + /api/health). Use after deploy / networking changes.

.EXAMPLE
  ./scripts/check_railway_public.ps1
  ./scripts/check_railway_public.ps1 -BaseUrl "https://mini-devin-production.up.railway.app"
#>
param(
    [string]$BaseUrl = "https://mini-devin-production.up.railway.app"
)

$BaseUrl = $BaseUrl.TrimEnd("/")
$paths = @("/api/health", "/", "/app")
$ok = $true
foreach ($p in $paths) {
    $u = "$BaseUrl$p"
    try {
        $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 25 -MaximumRedirection 5
        Write-Host "OK $($r.StatusCode)  $u"
    }
    catch {
        Write-Host "FAIL  $u"
        Write-Host "       $($_.Exception.Message)"
        $ok = $false
    }
}
if (-not $ok) {
    Write-Host ""
    Write-Host "If Deploy logs show 200 from 10.x but public FAIL: Railway -> mini-devin -> Settings -> Networking -> target port = PORT (often 8080). Then Redeploy."
    exit 1
}
exit 0
