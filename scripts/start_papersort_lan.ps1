param([switch]$ConfigureFirewallOnly)

$ErrorActionPreference = "Stop"

$projectName = "PaperSort"
$port = 8503
$ruleName = "PaperSort Streamlit 8503 LAN"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$appPath = "app.py"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-FirewallRuleReady {
    $rule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $rule -or $rule.Enabled -ne "True" -or $rule.Direction -ne "Inbound" -or $rule.Action -ne "Allow") {
        return $false
    }
    $portFilter = $rule | Get-NetFirewallPortFilter
    $addressFilter = $rule | Get-NetFirewallAddressFilter
    return ($portFilter.Protocol -eq "TCP" -and $portFilter.LocalPort -eq "$port" -and $addressFilter.RemoteAddress -eq "LocalSubnet")
}

function Ensure-FirewallRule {
    if (-not (Test-IsAdministrator)) {
        Write-Host "需要管理员权限配置 Windows 防火墙规则，正在请求 UAC..." -ForegroundColor Yellow
        $args = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$PSCommandPath`"",
            "-ConfigureFirewallOnly"
        )
        Start-Process -FilePath "powershell.exe" -ArgumentList $args -Verb RunAs -Wait
        return
    }
    $rules = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if (-not $rules) {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port -RemoteAddress LocalSubnet -Profile Any -Enabled True | Out-Null
    } else {
        $rules | Set-NetFirewallRule -Enabled True -Direction Inbound -Action Allow -Profile Any
        $rules | Get-NetFirewallPortFilter | Set-NetFirewallPortFilter -Protocol TCP -LocalPort $port
        $rules | Get-NetFirewallAddressFilter | Set-NetFirewallAddressFilter -RemoteAddress LocalSubnet
    }
}

if ($ConfigureFirewallOnly) {
    Ensure-FirewallRule
    exit
}

if (-not (Test-FirewallRuleReady)) {
    Ensure-FirewallRule
}

Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$lanAddresses = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.InterfaceAlias -notlike "*WSL*" } |
    Select-Object -ExpandProperty IPAddress

Write-Host ""
Write-Host "$projectName 将在局域网启动：" -ForegroundColor Green
foreach ($address in $lanAddresses) {
    Write-Host "  http://$address`:$port" -ForegroundColor Green
}
Write-Host ""
Write-Host "如果要停止服务，关闭这个窗口或按 Ctrl+C。" -ForegroundColor Yellow
Write-Host ""

& $python -m streamlit run $appPath --server.address=0.0.0.0 --server.port=$port
