$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$runner = Join-Path $projectRoot "tools\run_due_external_monitors.py"
$logDir = Join-Path $projectRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "external_monitor_task.log"

Set-Location $projectRoot
"===== $(Get-Date -Format o) =====" | Out-File -FilePath $logPath -Append -Encoding UTF8
$output = & $python $runner --json 2>&1
$output | Out-File -FilePath $logPath -Append -Encoding UTF8
