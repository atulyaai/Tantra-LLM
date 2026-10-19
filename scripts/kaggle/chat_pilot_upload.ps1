# Uploads code + approved English bundle and starts the chat pilot kernel on a GPU.
# One dataset holds code, the approved bundle and the base model (model/). Needs KAGGLE_API_TOKEN.
param([string]$Version = 'v2')
$ErrorActionPreference = 'Stop'
Set-Location 'F:\Tantra-LLM'
$env:KAGGLE_CONFIG_DIR = 'F:\Tantra-LLM\.kaggle'
$kaggle = '.\.venv\Scripts\kaggle.exe'
$user = 'tantrallm'
$bundle = 'runs\kaggle-chat-bundle\tantra'
if (Test-Path $bundle) { Remove-Item $bundle -Recurse -Force }
New-Item -ItemType Directory -Force "$bundle\Datasets" | Out-Null
robocopy Tantra "$bundle\Tantra" /E /XD __pycache__ _legacy /NFL /NDL /NJH /NJS | Out-Null
robocopy Datasets\workbench-chat-en-pilot-$Version "$bundle\Datasets\workbench-chat-en-pilot-$Version" /E /NFL /NDL /NJH /NJS | Out-Null
Copy-Item requirements.txt $bundle
New-Item -ItemType Directory -Force "$bundle\model" | Out-Null
'base.pt','tokenizer.json','report.json' | % { Copy-Item "runs\kaggle-output\base-training\$_" "$bundle\model\" }
@{ title = "Tantra chat English pilot $Version"; id = "$user/tantra-chat-en-pilot-$Version"; licenses = @(@{ name = 'other' }) } |
    ConvertTo-Json -Depth 3 | Out-File -Encoding ascii "$bundle\dataset-metadata.json"
& $kaggle datasets status "$user/tantra-chat-en-pilot-$Version" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { & $kaggle datasets version -p $bundle -m 'code + data + model in one dataset' --dir-mode zip }
else { & $kaggle datasets create -p $bundle --dir-mode zip }
Start-Sleep 90  # a new version reports the old 'ready' state until processing starts
do { Start-Sleep 20; $s = & $kaggle datasets status "$user/tantra-chat-en-pilot-$Version" 2>$null; Write-Host $s } until ("$s" -match 'ready')

$kernel = 'runs\kaggle-chat-kernel'
New-Item -ItemType Directory -Force $kernel | Out-Null
Copy-Item scripts\kaggle\chat_pilot_kaggle.py $kernel
@{ id = "$user/tantra-chat-train-$Version"; title = "Tantra chat train $Version"; code_file = 'chat_pilot_kaggle.py'; language = 'python';
   kernel_type = 'script'; is_private = $true; enable_gpu = $true; enable_internet = $true;
   dataset_sources = @("$user/tantra-chat-en-pilot-$Version"); competition_sources = @(); kernel_sources = @() } |
    ConvertTo-Json -Depth 3 | Out-File -Encoding ascii "$kernel\kernel-metadata.json"
& $kaggle kernels push -p $kernel
Write-Host "Started: https://www.kaggle.com/code/$user/tantra-chat-train-$Version"
