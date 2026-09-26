# Tantra — one-line install, repair and start (Windows 10/11).
#
#   irm https://raw.githubusercontent.com/atulyaai/Tantra-LLM/main/install.ps1 | iex
#
# Safe to run again any time: it only adds what is missing (Python, Git, packages, data),
# updates the code, repairs itself with `main.py --mode doctor --fix`, then opens the WebUI.
# Options (when saved as a file):  .\install.ps1 -Dir D:\Tantra-LLM -NoStart
param([string]$Dir = "", [switch]$NoStart)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
function Say($msg) { Write-Host "  » $msg" -ForegroundColor Yellow }
function Ok($msg)  { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Refresh-Path { $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") }

Write-Host "`n  तन्त्र  TANTRA — setup`n" -ForegroundColor DarkYellow

# 1. Where: this folder if it already is Tantra, else ~/Tantra-LLM
if (-not $Dir) {
    if (Test-Path (Join-Path (Get-Location) "main.py")) { $Dir = (Get-Location).Path }
    elseif ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "main.py"))) { $Dir = $PSScriptRoot }
    else { $Dir = Join-Path $HOME "Tantra-LLM" }
}

# 2. Python 3.10+ and Git (installed with winget when missing)
function Run-Py([string]$cmd, [string[]]$rest) {
    $p = $cmd.Split(" "); $args2 = @(); if ($p.Count -gt 1) { $args2 += $p[1..($p.Count - 1)] }
    & $p[0] @args2 @rest
}
function Find-Python {
    foreach ($c in @("py -3.12", "py -3", "python")) {
        try {
            $v = Run-Py $c @("-c", "import sys; print('%d.%d' % sys.version_info[:2])") 2>$null
            if ($v -and [version]$v -ge [version]"3.10") { return $c }
        } catch {}
    }
    return $null
}
$py = Find-Python
if (-not $py) {
    Say "Installing Python 3.12 …"
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent | Out-Null
    Refresh-Path; $py = Find-Python
    if (-not $py) { throw "Python could not be installed automatically. Install it from python.org and run this again." }
}
Ok "Python ($py)"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Say "Installing Git …"
    winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements --silent | Out-Null
    Refresh-Path
}
Ok "Git"

# 3. Code: clone the first time, update later (your data and models are never touched)
if (-not (Test-Path (Join-Path $Dir "main.py"))) {
    Say "Downloading Tantra into $Dir …"
    git clone --depth 1 https://github.com/atulyaai/Tantra-LLM.git $Dir
} elseif (Test-Path (Join-Path $Dir ".git")) {
    Say "Updating code …"
    try { git -C $Dir pull --ff-only } catch { Say "Could not update (local changes?) — keeping your version." }
}
Set-Location $Dir
Ok "Code in $Dir"

# 4. Private Python environment + packages (CPU PyTorch unless an NVIDIA GPU is present)
$venvPy = Join-Path $Dir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Say "Creating a private Python environment (.venv) …"
    Run-Py $py @("-m", "venv", ".venv")
}
& $venvPy -m pip install --upgrade pip --disable-pip-version-check -q
$hasTorch = (& $venvPy -c "import importlib.util as u; print(bool(u.find_spec('torch')))") -eq "True"
if (-not $hasTorch) {
    $gpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    Say ("Installing PyTorch (" + ($(if ($gpu) { "GPU" } else { "CPU" })) + ") — the largest download, a few minutes …")
    if ($gpu) { & $venvPy -m pip install torch -q } else { & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q }
}
Say "Installing Tantra's packages …"
& $venvPy -m pip install -r requirements.txt -q --disable-pip-version-check
Ok "Packages"

# 5. Self-check and repair (speech, PDF, data, Smriti … whatever is missing)
Say "Checking every part and repairing what is missing …"
& $venvPy main.py --mode doctor --fix --auto

# 6. Desktop shortcut
try {
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Tantra.lnk"
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $s.TargetPath = Join-Path $Dir "tantra.bat"; $s.Arguments = "serve"; $s.WorkingDirectory = $Dir
    $icon = Join-Path $Dir "Assets\tantra.ico"; if (Test-Path $icon) { $s.IconLocation = $icon }
    $s.Save(); Ok "Desktop shortcut: Tantra"
} catch { Say "Could not create a desktop shortcut (not important)." }

# 7. Start
if (-not $NoStart) {
    Ok "Starting Tantra → http://127.0.0.1:8000"
    Start-Process (Join-Path $Dir "tantra.bat") -ArgumentList "serve" -WorkingDirectory $Dir
}
Write-Host "`n  Done. Next time just double-click 'Tantra' on your desktop.`n" -ForegroundColor Green
