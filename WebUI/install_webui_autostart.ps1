<#
.SYNOPSIS
Registers a per-user Windows logon task that launches Tantra Studio.
#>

$launcher = Join-Path $PSScriptRoot "start_webui.ps1"
$taskName = "TantraLLMWebUI"
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtLogOn
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Force -ErrorAction Stop | Out-Null
    Write-Host "Tantra Studio will start automatically when you sign in."
} catch {
    # Some managed Windows installations deny Task Scheduler registration.
    # The per-user Startup folder has the same sign-in behaviour and needs no
    # administrative access.
    $startupDir = [Environment]::GetFolderPath("Startup")
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "TantraLLMWebUI.cmd") `
        -Destination (Join-Path $startupDir "TantraLLMWebUI.cmd") -Force
    Write-Host "Task Scheduler is unavailable; installed Startup-folder auto-launch instead."
}
