# GitHub Repository Configuration Script for Tantra-LLM (PowerShell)
# Run this to set up repository settings via GitHub CLI

Write-Host "Setting repository description..." -ForegroundColor Green
gh api repos/atulyaai/Tantra-LLM -X PATCH -F description="🧠 Proprietary Multimodal Cognitive Architecture - SpikingBrain + Long-VITA + Whisper with dynamic context, fusion layers, hybrid memory, and adaptive personality. Not a chatbot. A cognitive mind."

Write-Host "Setting repository topics..." -ForegroundColor Green
gh api repos/atulyaai/Tantra-LLM -X PUT -H "Accept: application/vnd.github.mercy-preview+json" -F names='["tantra-llm","multimodal-ai","spikingbrain","cognitive-architecture","fusion-layers","adaptive-personality","hybrid-memory","proprietary-ai","reasoning-system","neural-architecture"]'

Write-Host "Setting homepage URL..." -ForegroundColor Green
gh api repos/atulyaai/Tantra-LLM -X PATCH -F homepage="https://github.com/atulyaai/Tantra-LLM"

Write-Host "Setting default branch to main..." -ForegroundColor Green
gh api repos/atulyaai/Tantra-LLM -X PATCH -F default_branch="main"

Write-Host "Repository configuration complete!" -ForegroundColor Green

