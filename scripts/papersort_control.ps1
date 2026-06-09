param(
    [ValidateSet("menu", "status", "start", "stop", "restart", "open", "logs", "firewall", "install-task", "run-task")]
    [string]$Action = "menu"
)

$ErrorActionPreference = "Stop"

try {
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
}
catch {
}

$ProjectName = "PaperSort"
$Port = 8503
$QdrantUrl = "http://127.0.0.1:6333/collections"
$TaskName = "PaperSort External Paper Monitor"
$FirewallRuleName = "PaperSort Streamlit 8503 LAN"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AppPath = Join-Path $ProjectRoot "app.py"
$LogDir = Join-Path $ProjectRoot "logs"
$StreamlitLog = Join-Path $LogDir "streamlit_8503.log"
$MonitorLog = Join-Path $LogDir "external_monitor_task.log"

function ConvertTo-PowerShellLiteral {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Get-PythonExecutable {
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

function Get-PortProcesses {
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        $processIds = @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
        foreach ($processId in $processIds) {
            Get-Process -Id $processId -ErrorAction SilentlyContinue
        }
    }
    catch {
        return @()
    }
}

function Get-LanAddresses {
    try {
        return @(
            Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.IPAddress -notlike "127.*" -and
                    $_.IPAddress -notlike "169.254.*" -and
                    $_.InterfaceAlias -notlike "*WSL*"
                } |
                Select-Object -ExpandProperty IPAddress -Unique
        )
    }
    catch {
        return @()
    }
}

function Test-StreamlitHealth {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:${Port}/_stcore/health" -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500)
    }
    catch {
        return $false
    }
}

function Test-QdrantHealth {
    try {
        $response = Invoke-WebRequest -Uri $QdrantUrl -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500)
    }
    catch {
        return $false
    }
}

function Test-FirewallRuleReady {
    try {
        $rule = Get-NetFirewallRule -DisplayName $FirewallRuleName -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $rule -or $rule.Enabled -ne "True" -or $rule.Direction -ne "Inbound" -or $rule.Action -ne "Allow") {
            return $false
        }
        $portFilter = $rule | Get-NetFirewallPortFilter
        $addressFilter = $rule | Get-NetFirewallAddressFilter
        $remoteAddresses = @($addressFilter.RemoteAddress)
        return ($portFilter.Protocol -eq "TCP" -and $portFilter.LocalPort -eq "$Port" -and $remoteAddresses -contains "LocalSubnet")
    }
    catch {
        return $false
    }
}

function Invoke-FirewallInstaller {
    $lanScript = Join-Path $ProjectRoot "scripts\start_papersort_lan.ps1"
    if (-not (Test-Path $lanScript)) {
        Write-Host "Firewall helper is missing: $lanScript" -ForegroundColor Red
        return
    }
    & $lanScript -ConfigureFirewallOnly
}

function Write-Header {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " $ProjectName Control Center" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-AccessUrls {
    Write-Host "Local URL: http://127.0.0.1:${Port}" -ForegroundColor Green
    $addresses = Get-LanAddresses
    if ($addresses.Count -gt 0) {
        Write-Host "LAN URLs:" -ForegroundColor Green
        foreach ($address in $addresses) {
            Write-Host "  http://$address`:$Port" -ForegroundColor Green
        }
    }
    else {
        Write-Host "LAN URL: no non-loopback IPv4 address was found." -ForegroundColor Yellow
    }
}

function Show-MonitorTaskStatus {
    try {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        $info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction Stop
        $action = @($task.Actions | Select-Object -First 1)[0]
        $hidden = ($action.Execute -ieq "wscript.exe")
        Write-Host "Monitor task: installed" -ForegroundColor Green
        Write-Host "Task state: $($task.State)"
        Write-Host "Silent mode: $hidden"
        Write-Host "Last run: $($info.LastRunTime)"
        Write-Host "Next run: $($info.NextRunTime)"
        Write-Host "Last result: $($info.LastTaskResult)"
    }
    catch {
        Write-Host "Monitor task: not installed" -ForegroundColor Yellow
    }
}

function Show-Status {
    Write-Header
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Port: $Port"
    Write-Host ""

    $processes = @(Get-PortProcesses)
    if ($processes.Count -gt 0) {
        Write-Host "Streamlit: running" -ForegroundColor Green
        $processes | Select-Object Id, ProcessName, Path | Format-Table -AutoSize | Out-String | Write-Host
    }
    else {
        Write-Host "Streamlit: stopped" -ForegroundColor Yellow
    }

    if (Test-StreamlitHealth) {
        Write-Host "Streamlit health: OK" -ForegroundColor Green
    }
    else {
        Write-Host "Streamlit health: not reachable yet" -ForegroundColor Yellow
    }

    if (Test-FirewallRuleReady) {
        Write-Host "Firewall rule: ready for LocalSubnet on TCP $Port" -ForegroundColor Green
    }
    else {
        Write-Host "Firewall rule: missing or incomplete" -ForegroundColor Yellow
    }

    if (Test-QdrantHealth) {
        Write-Host "Qdrant: reachable at $QdrantUrl" -ForegroundColor Green
    }
    else {
        Write-Host "Qdrant: not reachable at $QdrantUrl" -ForegroundColor Yellow
    }

    Write-Host ""
    Show-MonitorTaskStatus
    Write-Host ""
    Write-AccessUrls
    Write-Host ""
    Write-Host "Streamlit log: $StreamlitLog"
    Write-Host "Monitor log: $MonitorLog"
}

function Start-PaperSort {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    $processes = @(Get-PortProcesses)
    if ($processes.Count -gt 0) {
        Write-Host "PaperSort is already listening on port $Port." -ForegroundColor Green
        Write-AccessUrls
        return
    }

    if (-not (Test-FirewallRuleReady)) {
        Write-Host "Firewall rule is not ready. Windows may ask for administrator permission once." -ForegroundColor Yellow
        Invoke-FirewallInstaller
    }

    $python = Get-PythonExecutable
    $rootLiteral = ConvertTo-PowerShellLiteral $ProjectRoot
    $pythonLiteral = ConvertTo-PowerShellLiteral $python
    $appLiteral = ConvertTo-PowerShellLiteral $AppPath
    $logLiteral = ConvertTo-PowerShellLiteral $StreamlitLog
    $command = @"
`$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
Set-Location $rootLiteral
'===== ' + (Get-Date -Format o) + ' =====' | Out-File -FilePath $logLiteral -Append -Encoding UTF8
& $pythonLiteral -m streamlit run $appLiteral --server.address 0.0.0.0 --server.port $Port *>> $logLiteral
"@
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $process = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encodedCommand) `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "Starting PaperSort in the background. Launcher PID: $($process.Id)" -ForegroundColor Green
    Write-Host "Log: $StreamlitLog"
    for ($i = 0; $i -lt 20; $i++) {
        if (Test-StreamlitHealth) {
            Write-Host "PaperSort is ready." -ForegroundColor Green
            Write-AccessUrls
            return
        }
        Start-Sleep -Seconds 1
    }
    Write-Host "PaperSort was started, but health check is not ready yet. Check the log if it stays unavailable." -ForegroundColor Yellow
    Write-AccessUrls
}

function Stop-PaperSort {
    $processes = @(Get-PortProcesses)
    if ($processes.Count -eq 0) {
        Write-Host "No process is listening on port $Port." -ForegroundColor Yellow
        return
    }

    foreach ($process in $processes) {
        Write-Host "Stopping PID $($process.Id) ($($process.ProcessName))..." -ForegroundColor Yellow
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
    if (@(Get-PortProcesses).Count -eq 0) {
        Write-Host "PaperSort stopped." -ForegroundColor Green
    }
    else {
        Write-Host "Some process is still listening on port $Port." -ForegroundColor Red
    }
}

function Restart-PaperSort {
    Stop-PaperSort
    Start-Sleep -Seconds 1
    Start-PaperSort
}

function Open-PaperSort {
    Start-Process "http://127.0.0.1:${Port}"
}

function Open-Logs {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    Start-Process explorer.exe $LogDir
    Write-Host "Opened log folder: $LogDir" -ForegroundColor Green
    if (Test-Path $StreamlitLog) {
        Write-Host ""
        Write-Host "Recent Streamlit log:" -ForegroundColor Cyan
        Get-Content $StreamlitLog -Tail 20
    }
    if (Test-Path $MonitorLog) {
        Write-Host ""
        Write-Host "Recent monitor log:" -ForegroundColor Cyan
        Get-Content $MonitorLog -Tail 20
    }
}

function Install-MonitorTask {
    $script = Join-Path $ProjectRoot "scripts\install_external_monitor_task.ps1"
    if (-not (Test-Path $script)) {
        Write-Host "Task installer is missing: $script" -ForegroundColor Red
        return
    }
    & $script
    Write-Host ""
    Show-MonitorTaskStatus
}

function Run-MonitorTask {
    try {
        Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        Write-Host "Monitor task triggered: $TaskName" -ForegroundColor Green
        Write-Host "Log: $MonitorLog"
    }
    catch {
        Write-Host "Monitor task is not installed. Use option 6 first." -ForegroundColor Yellow
    }
}

function Invoke-SelectedAction {
    param([Parameter(Mandatory = $true)][string]$SelectedAction)
    switch ($SelectedAction) {
        "status" { Show-Status }
        "start" { Start-PaperSort }
        "stop" { Stop-PaperSort }
        "restart" { Restart-PaperSort }
        "open" { Open-PaperSort }
        "logs" { Open-Logs }
        "firewall" {
            Invoke-FirewallInstaller
            if (Test-FirewallRuleReady) {
                Write-Host "Firewall rule is ready." -ForegroundColor Green
            }
            else {
                Write-Host "Firewall rule still does not look ready. Run this option as administrator if needed." -ForegroundColor Yellow
            }
        }
        "install-task" { Install-MonitorTask }
        "run-task" { Run-MonitorTask }
    }
}

function Show-Menu {
    do {
        Write-Header
        Write-Host "[1] Status"
        Write-Host "[2] Start PaperSort"
        Write-Host "[3] Stop PaperSort"
        Write-Host "[4] Restart PaperSort"
        Write-Host "[5] Open browser"
        Write-Host "[6] Install/repair monitor task"
        Write-Host "[7] Trigger monitor task now"
        Write-Host "[8] Open logs"
        Write-Host "[9] Repair firewall rule"
        Write-Host "[0] Exit"
        Write-Host ""
        $choice = Read-Host "Choose"
        Write-Host ""
        switch ($choice) {
            "1" { Show-Status }
            "2" { Start-PaperSort }
            "3" { Stop-PaperSort }
            "4" { Restart-PaperSort }
            "5" { Open-PaperSort }
            "6" { Install-MonitorTask }
            "7" { Run-MonitorTask }
            "8" { Open-Logs }
            "9" {
                Invoke-FirewallInstaller
                if (Test-FirewallRuleReady) {
                    Write-Host "Firewall rule is ready." -ForegroundColor Green
                }
                else {
                    Write-Host "Firewall rule still does not look ready. Run this option as administrator if needed." -ForegroundColor Yellow
                }
            }
            "0" { return }
            default { Write-Host "Unknown option." -ForegroundColor Yellow }
        }
        Write-Host ""
        Read-Host "Press Enter to continue"
    } while ($true)
}

if ($Action -eq "menu") {
    Show-Menu
}
else {
    Invoke-SelectedAction -SelectedAction $Action
}
