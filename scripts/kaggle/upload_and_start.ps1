# Uploads the private dataset and starts the GPU kernel. Uses only F:\Tantra-LLM.
$ErrorActionPreference = 'Stop'
Set-Location 'F:\Tantra-LLM'
$env:KAGGLE_CONFIG_DIR = 'F:\Tantra-LLM\.kaggle'
$kaggle = '.\.venv\Scripts\kaggle.exe'
if (Test-Path .kaggle\kaggle.json) { $user = (Get-Content .kaggle\kaggle.json | ConvertFrom-Json).username }
else { $user = $env:KAGGLE_USERNAME }
if (-not $user) { throw 'Put kaggle.json in F:\Tantra-LLM\.kaggle, or set KAGGLE_USERNAME and KAGGLE_API_TOKEN' }

@{ title = 'Tantra 64K pretraining resume'; id = "$user/tantra-64k-resume"; licenses = @(@{ name = 'other' }) } |
    ConvertTo-Json -Depth 3 | Out-File -Encoding ascii runs\kaggle-bundle\tantra\dataset-metadata.json
& $kaggle datasets create -p runs\kaggle-bundle\tantra --dir-mode zip
Write-Host 'Waiting for Kaggle to finish processing the dataset...'
do { Start-Sleep 30; $s = & $kaggle datasets status "$user/tantra-64k-resume"; Write-Host $s } until ("$s" -match 'ready')

@{ id = "$user/tantra-64k-train"; title = 'Tantra 64K train'; code_file = 'train_kaggle.py'; language = 'python';
   kernel_type = 'script'; is_private = $true; enable_gpu = $true; enable_internet = $true;
   dataset_sources = @("$user/tantra-64k-resume"); competition_sources = @(); kernel_sources = @() } |
    ConvertTo-Json -Depth 3 | Out-File -Encoding ascii runs\kaggle-kernel\kernel-metadata.json
& $kaggle kernels push -p runs\kaggle-kernel
Write-Host "Started. Watch: https://www.kaggle.com/code/$user/tantra-64k-train"
