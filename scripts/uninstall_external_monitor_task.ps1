$ErrorActionPreference = "Stop"

$taskName = "PaperSort External Paper Monitor"

try {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Deleted scheduled task: $taskName" -ForegroundColor Green
        exit 0
    }
}
catch {
    # Fall through to schtasks.exe, which is often available for current-user tasks.
}

& schtasks.exe /Query /TN $taskName | Out-Null
if ($LASTEXITCODE -eq 0) {
    & schtasks.exe /Delete /TN $taskName /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to delete scheduled task: $taskName"
    }
    Write-Host "Deleted scheduled task: $taskName" -ForegroundColor Green
}
else {
    Write-Host "Scheduled task does not exist: $taskName" -ForegroundColor Yellow
}
