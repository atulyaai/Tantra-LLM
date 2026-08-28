<#
.SYNOPSIS
  Universal Windows PowerShell launcher for Tantra-LLM.

.EXAMPLE
  .\tantra.ps1 train
  .\tantra.ps1 resume
  .\tantra.ps1 chat
  .\tantra.ps1 benchmark
  .\tantra.ps1 export
  .\tantra.ps1 webui
  .\tantra.ps1 test
#>
param([ValidateSet("train", "resume", "chat", "benchmark", "export", "webui", "test", "status", "help")][string]$Command = "help")

$Python = "python"

switch ($Command) {
  "train"     { & $Python main.py --mode auto-pilot --dataset Datasets/expert_conversation.jsonl --steps 10000 --batch-size 16 --grad-accum 2 --auto-growth --device auto }
  "resume"    { & $Python main.py --mode auto-pilot --resume --device auto }
  "chat"      { & $Python main.py --mode chat --checkpoint Model/Latest/checkpoint_latest.pt --temperature 0.3 }
  "benchmark" { & $Python main.py --mode benchmark --checkpoint Model/Latest/checkpoint_latest.pt }
  "export"    { & $Python main.py --mode export --checkpoint Model/Latest/checkpoint_latest.pt }
  "webui"     { & .\webui\start_webui.ps1 }
  "test"      { & $Python -m pytest Tests -q }
  "status"    { if (Test-Path Model\training_status.json) { Get-Content Model\training_status.json } else { Write-Host "No active training status file." } }
  default {
    Write-Host "Tantra commands: train, resume, chat, benchmark, export, webui, test, status"
  }
}
