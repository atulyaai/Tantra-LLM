<#
.SYNOPSIS
  Small Windows command runner for the maintained CPU profile.

.EXAMPLE
  .\tantra.ps1 train
  .\tantra.ps1 resume
  .\tantra.ps1 chat
  .\tantra.ps1 webui
#>
param([ValidateSet("train", "resume", "chat", "webui", "test", "status", "help")][string]$Command = "help")

$Python = "python"
$ProfileArgs = @(
  "-m", "Tantra.cpu_cli", "train", "--profile", "dense", "--attention", "causal",
  "--vocab-size", "32768", "--model-dir", "Model\CPU_Dense32K",
  "--tokenizer", "Model\tokenizer.json", "--dataset", "Datasets", "--steps", "50000",
  "--batch-size", "8", "--grad-accum", "1", "--seq-len", "128", "--data-workers", "2",
  "--checkpoint-every", "500", "--eval-every", "1000"
)

switch ($Command) {
  "train"  { & $Python @ProfileArgs }
  "resume" { & $Python @ProfileArgs --resume }
  "chat"   { & $Python -m Tantra.cpu_cli chat --model-dir Model\CPU_Dense32K --tokenizer Model\tokenizer.json }
  "webui"  { & .\webui\start_webui.ps1 }
  "test"   { & $Python -m pytest Tests -q }
  "status" { Get-Content Model\training_status.json }
  default {
    Write-Host "Tantra commands: train, resume, chat, webui, test, status"
  }
}
