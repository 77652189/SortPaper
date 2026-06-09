$ErrorActionPreference = "Stop"

$taskName = "PaperSort External Paper Monitor"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$runner = Join-Path $projectRoot "tools\run_due_external_monitors.py"
$taskRunner = Join-Path $projectRoot "scripts\run_external_monitor_task.ps1"
$hiddenTaskRunner = Join-Path $projectRoot "scripts\run_external_monitor_task_hidden.vbs"
$logDir = Join-Path $projectRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "external_monitor_task.log"

if (-not (Test-Path $runner)) {
    throw "Monitor runner not found: $runner"
}
if (-not (Test-Path $taskRunner)) {
    throw "Task wrapper not found: $taskRunner"
}
if (-not (Test-Path $hiddenTaskRunner)) {
    throw "Hidden task wrapper not found: $hiddenTaskRunner"
}

$action = New-ScheduledTaskAction `
    -Execute "wscript.exe" `
    -Argument "`"$hiddenTaskRunner`""

$hourlyTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

$installedWith = "Register-ScheduledTask"
try {
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger @($hourlyTrigger, $logonTrigger) `
        -Settings $settings `
        -Principal $principal `
        -Description "Run PaperSort due external paper monitors every hour without opening Streamlit." `
        -Force | Out-Null
}
catch {
    $installedWith = "schtasks.exe"
    $taskCommand = "wscript.exe `"$hiddenTaskRunner`""
    & schtasks.exe /Create /TN $taskName /SC HOURLY /MO 1 /TR $taskCommand /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw
    }
}

Write-Host "Installed scheduled task: $taskName" -ForegroundColor Green
Write-Host "Installer: $installedWith" -ForegroundColor Green
if ($installedWith -eq "Register-ScheduledTask") {
    Write-Host "Schedule: run once at logon, then check due topics every hour." -ForegroundColor Green
}
else {
    Write-Host "Schedule: check due topics every hour." -ForegroundColor Green
}
Write-Host "Log file: $logPath" -ForegroundColor Green
Write-Host ""
Write-Host "Manual test:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName 'PaperSort External Paper Monitor'"
Write-Host ""
Write-Host "Check status:" -ForegroundColor Cyan
Write-Host "  Get-ScheduledTaskInfo -TaskName 'PaperSort External Paper Monitor'"
