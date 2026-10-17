# Uploads code + approved English bundle and starts the chat pilot kernel on a GPU.
# Needs KAGGLE_API_TOKEN (or F:\Tantra-LLM\.kaggle) and the existing tantrallm/tantra-64k-base-v1 dataset.
$ErrorActionPreference = 'Stop'
Set-Location 'F:\Tantra-LLM'
$env:KAGGLE_CONFIG_DIR = 'F:\Tantra-LLM\.kaggle'
$kaggle = '.\.venv\Scripts\kaggle.exe'
$user = 'tantrallm'
$bundle = 'runs\kaggle-chat-bundle\tantra'
if (Test-Path $bundle) { Remove-Item $bundle -Recurse -Force }
New-Item -ItemType Directory -Force "$bundle\Datasets" | Out-Null
robocopy Tantra "$bundle\Tantra" /E /XD __pycache__ _legacy /NFL /NDL /NJH /NJS | Out-Null
robocopy Datasets\workbench-chat-en-pilot-v1 "$bundle\Datasets\workbench-chat-en-pilot-v1" /E /NFL /NDL /NJH /NJS | Out-Null
Copy-Item requirements.txt $bundle
@{ title = 'Tantra chat English pilot v1'; id = "$user/tantra-chat-en-pilot-v1"; licenses = @(@{ name = 'other' }) } |
    ConvertTo-Json -Depth 3 | Out-File -Encoding ascii "$bundle\dataset-metadata.json"
& $kaggle datasets create -p $bundle --dir-mode zip
do { Start-Sleep 20; $s = & $kaggle datasets status "$user/tantra-chat-en-pilot-v1" 2>$null; Write-Host $s } until ("$s" -match 'ready')

$kernel = 'runs\kaggle-chat-kernel'
New-Item -ItemType Directory -Force $kernel | Out-Null
Copy-Item scripts\kaggle\chat_pilot_kaggle.py $kernel
@{ id = "$user/tantra-chat-en-pilot"; title = 'Tantra chat en pilot'; code_file = 'chat_pilot_kaggle.py'; language = 'python';
   kernel_type = 'script'; is_private = $true; enable_gpu = $true; enable_internet = $true;
   dataset_sources = @("$user/tantra-chat-en-pilot-v1", "$user/tantra-64k-base-v1"); competition_sources = @(); kernel_sources = @() } |
    ConvertTo-Json -Depth 3 | Out-File -Encoding ascii "$kernel\kernel-metadata.json"
& $kaggle kernels push -p $kernel
Write-Host "Started: https://www.kaggle.com/code/$user/tantra-chat-en-pilot"
