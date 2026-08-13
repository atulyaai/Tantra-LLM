<#
.SYNOPSIS
  Tantra-LLM — Simple command runner
  Usage: .\tantra.ps1 <command>

COMMANDS
  finetune   Fine-tune on identity data (recommended first run)
  pretrain   Full pretraining on master dataset
  resume     Resume last training from checkpoint
  chat       Interactive CLI chat
  server     Start web UI server
  status     Show model status + loss dashboard
  generate   Quick test generation
  help       Show this help
#>

param([string]$cmd = "help")

$PYTHON = "python"
$MAIN   = "main.py"

# ── Colours ────────────────────────────────────────────────────────────────
function info  { param($m) Write-Host "  ▶ $m" -ForegroundColor Cyan   }
function ok    { param($m) Write-Host "  ✔ $m" -ForegroundColor Green  }
function err   { param($m) Write-Host "  ✖ $m" -ForegroundColor Red    }
function title { param($m) Write-Host "`n━━━ $m ━━━" -ForegroundColor Yellow }

switch ($cmd.ToLower()) {

  # ── Fine-tune on identity / chat data (start here) ──────────────────────
  "finetune" {
    title "Fine-tune  (identity + chat quality)"
    info "2000 steps · lr=5e-5 · grad-accum=8 · fast mode"
    & $PYTHON $MAIN `
      --mode dataset `
      --dataset Datasets/tantra_identity_safety_expanded.jsonl `
      --steps 2000 `
      --lr 5e-5 `
      --warmup 200 `
      --grad-accum 8 `
      --seq-len 256 `
      --log-every 50 `
      --eval-every 500 `
      --no-latent-reasoning
  }

  # ── Full pre-train on master dataset ────────────────────────────────────
  "pretrain" {
    title "Pre-train  (master dataset · 10 000 steps)"
    info "10000 steps · lr=1e-4 · grad-accum=4 · fast mode"
    & $PYTHON $MAIN `
      --mode dataset `
      --steps 10000 `
      --lr 1e-4 `
      --warmup 500 `
      --grad-accum 4 `
      --seq-len 512 `
      --log-every 100 `
      --eval-every 500 `
      --no-latent-reasoning
  }

  # ── Resume last checkpoint ───────────────────────────────────────────────
  "resume" {
    title "Resume training from last checkpoint"
    info "Picks up where you left off (Latest checkpoint)"
    & $PYTHON $MAIN `
      --mode dataset `
      --steps 5000 `
      --lr 5e-5 `
      --warmup 100 `
      --grad-accum 8 `
      --seq-len 256 `
      --log-every 50 `
      --eval-every 500 `
      --no-latent-reasoning `
      --resume
  }

  # ── Quick fine-tune (just 500 steps, good for testing) ──────────────────
  "quick" {
    title "Quick fine-tune  (500 steps · test run)"
    info "500 steps · lr=5e-5 · fast"
    & $PYTHON $MAIN `
      --mode dataset `
      --dataset Datasets/tantra_identity_safety_expanded.jsonl `
      --steps 500 `
      --lr 5e-5 `
      --warmup 50 `
      --grad-accum 4 `
      --seq-len 128 `
      --log-every 25 `
      --eval-every 250 `
      --no-latent-reasoning
  }

  # ── Chat ─────────────────────────────────────────────────────────────────
  "chat" {
    title "Interactive Chat (CLI)"
    & $PYTHON $MAIN --mode chat
  }

  # ── Server (Web UI) ──────────────────────────────────────────────────────
  "server" {
    title "Starting Web UI Server  (http://localhost:8000)"
    & $PYTHON server.py
  }

  # ── Status dashboard ─────────────────────────────────────────────────────
  "status" {
    title "Model Status Dashboard"
    & $PYTHON $MAIN --mode status
  }

  # ── Quick generation test ────────────────────────────────────────────────
  "generate" {
    title "Quick Generation Test"
    & $PYTHON $MAIN --mode generate
  }

  # ── Help ─────────────────────────────────────────────────────────────────
  default {
    Write-Host ""
    Write-Host "  ████████╗ █████╗ ███╗  ██╗████████╗██████╗  █████╗ " -ForegroundColor Cyan
    Write-Host "     ██╔══╝██╔══██╗████╗ ██║╚══██╔══╝██╔══██╗██╔══██╗" -ForegroundColor Cyan
    Write-Host "     ██║   ███████║██╔██╗██║   ██║   ██████╔╝███████║" -ForegroundColor Cyan
    Write-Host "     ██║   ██╔══██║██║╚████║   ██║   ██╔══██╗██╔══██║" -ForegroundColor Cyan
    Write-Host "     ██║   ██║  ██║██║ ╚███║   ██║   ██║  ██║██║  ██║" -ForegroundColor Cyan
    Write-Host "     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝" -ForegroundColor Cyan
    Write-Host "  LLM by Atulya AI — Local · Private · CPU-First" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  TRAINING" -ForegroundColor Yellow
    Write-Host "    .\tantra.ps1 finetune    Fine-tune on identity/chat data  ← start here"
    Write-Host "    .\tantra.ps1 pretrain    Full pretraining (10k steps, master dataset)"
    Write-Host "    .\tantra.ps1 resume      Resume last checkpoint"
    Write-Host "    .\tantra.ps1 quick       Quick 500-step test run"
    Write-Host ""
    Write-Host "  INFERENCE" -ForegroundColor Yellow
    Write-Host "    .\tantra.ps1 chat        Interactive CLI chat"
    Write-Host "    .\tantra.ps1 server      Start web UI  (http://localhost:8000)"
    Write-Host "    .\tantra.ps1 generate    Quick generation test"
    Write-Host ""
    Write-Host "  INFO" -ForegroundColor Yellow
    Write-Host "    .\tantra.ps1 status      Model status + loss dashboard"
    Write-Host ""
    Write-Host "  EXAMPLES" -ForegroundColor DarkGray
    Write-Host "    .\tantra.ps1 finetune    # first time — teach Tantra who it is"
    Write-Host "    .\tantra.ps1 pretrain    # overnight run for better general knowledge"
    Write-Host "    .\tantra.ps1 server      # launch the web UI"
    Write-Host ""
  }
}
